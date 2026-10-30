"""CPU-only tests of actual H3 reference methods, without importing the runtime."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


RUNTIME = Path(os.environ["WANGP_TEST_RUNTIME"]).resolve()
SOURCE = RUNTIME / "models/minimax_h3/pipeline.py"


def load_pipeline_methods():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Pipeline")
    names = {"_update_tensor_digest", "_image_latent_cache_key", "_encode_image_reference",
             "_restore_image_cache_cpu_state",
             "clear_image_latent_cache", "_add_image_reference", "set_offload_handoff",
             "detach_shared_components", "_prepare_condition_rows"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"hashlib": hashlib, "os": os, "torch": torch,
                 "kernel_policy": SimpleNamespace(precision="fast", families="auto"),
                 "offload": SimpleNamespace(shared_state={"_attention": "vsa"}),
                 "H3_IMAGE_LATENT_CACHE_MAX_BYTES": 16 * 1024**2,
                 "_qwen_frames": lambda video: video.clone(),
                 "VISUAL_COND_TIMESTEP": 0.9,
                 "patchify_video": lambda latent, patch_size: latent.flatten().clone(),
                 "pack_audio": lambda latent: latent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace["MiniMaxH3Pipeline"], namespace


class ReferenceVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.quant_conv = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("pixel_mean", torch.tensor([0.5]))
        self._model_dtype = torch.float32
        self.config = {"clip_length": 17}
        self.use_tiling = True
        self.tile_sample_min_height = self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = 64
        self.upsampling = False
        self.eval()


class H3ImageLatentCacheTests(unittest.TestCase):
    def setUp(self):
        self.cls, self.namespace = load_pipeline_methods()
        self.pipe = self.cls.__new__(self.cls)
        self.pipe.vae = ReferenceVAE()
        self.pipe.video_encoder = torch.nn.ModuleDict({"encoder": self.pipe.vae.encoder,
                                                       "quant_conv": self.pipe.vae.quant_conv})
        self.pipe.fixed_prompt = None
        self.pipe._image_latent_cache = None
        self.pipe.image_latent_cache_stats = {"hits": 0, "misses": 0, "bypasses": 0}
        self.pipe._shared_offloadobj = self.pipe._private_offloadobj = None
        self.pipe.transformer = SimpleNamespace(patch_size=1)
        self.image = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2)
        self.encoded = 0
        self.abort = False
        self.failure = None
        self.pipe._prepare_image_reference = lambda image, width, height, relative_size: image
        self.pipe._check_abort = self.check_abort
        self.pipe._encode_video = self.encode
        self.env = patch.dict(os.environ, {"WGP_H3_IMAGE_LATENT_CACHE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def check_abort(self):
        if self.abort:
            raise RuntimeError("cancelled")

    def encode(self, video):
        self.check_abort()
        self.encoded += 1
        if self.failure:
            raise self.failure
        # Match the real encoder's private seed, without touching task/global RNG.
        noise = torch.randn((1, 2, 1, 2, 2), generator=torch.Generator().manual_seed(42))
        return noise + video.sum()

    def reference(self, image=None):
        presentation, latents, refs = [], [], []
        self.pipe._add_image_reference(self.image if image is None else image, 640, 384, 100,
                                       presentation, latents, refs)
        self.assertEqual(len(presentation), 1)
        self.assertEqual(refs, [{"kind": "image", "latent_h": 2, "latent_w": 2}])
        return latents[0]

    def test_repeated_image_skips_encoder_and_returns_independent_tensors(self):
        first = self.reference()
        expected = first.clone()
        first.zero_()
        second = self.reference()
        self.assertEqual(self.encoded, 1)
        self.assertTrue(torch.equal(second, expected))
        second.zero_()
        self.assertTrue(torch.equal(self.reference(), expected))
        self.assertEqual(self.pipe.image_latent_cache_stats, {"hits": 2, "misses": 1, "bypasses": 0})

    def test_default_off_and_dynamic_disable_drop_cached_entry(self):
        self.reference()
        os.environ.pop("WGP_H3_IMAGE_LATENT_CACHE")
        self.reference()
        self.reference()
        os.environ["WGP_H3_IMAGE_LATENT_CACHE"] = "1"
        self.reference()
        self.assertEqual(self.encoded, 4)
        self.assertEqual(self.pipe.image_latent_cache_stats, {"hits": 0, "misses": 2, "bypasses": 2})

    def test_same_shape_changed_content_and_one_entry_eviction(self):
        self.reference()
        self.reference(self.image + 1)
        self.reference()
        self.assertEqual(self.encoded, 3)

    def simulate_mmgp_encoder_placement(self):
        original = self.pipe.vae.encoder.weight
        encode = self.pipe._encode_video

        def encode_with_resident_copy(video):
            latent = encode(video)
            # MMGP installs a different tensor during GPU work, then restores
            # this original CPU tensor at task completion. Both are CPU here.
            self.pipe.vae.encoder.weight = torch.nn.Parameter(original.detach().clone(), requires_grad=False)
            return latent

        self.pipe._encode_video = encode_with_resident_copy
        return original

    def test_mmgp_restored_original_weights_reuse_pre_encode_guard(self):
        original = self.simulate_mmgp_encoder_placement()
        expected = self.reference()
        self.assertIsNot(self.pipe.vae.encoder.weight, original)
        self.pipe.vae.encoder.weight = original
        self.assertTrue(torch.equal(self.reference(), expected))
        self.assertEqual(self.encoded, 1)
        self.assertEqual(self.pipe.image_latent_cache_stats, {"hits": 1, "misses": 1, "bypasses": 0})

    def test_mutating_mmgp_restored_original_weights_still_invalidates(self):
        original = self.simulate_mmgp_encoder_placement()
        self.reference()
        self.pipe.vae.encoder.weight = original
        with torch.no_grad():
            original.add_(1)
        self.reference()
        self.assertEqual(self.encoded, 2)
        self.assertEqual(self.pipe.image_latent_cache_stats["hits"], 0)

    def test_same_bytes_different_tensor_dtype_or_shape_are_misses(self):
        self.reference()
        self.reference(self.image.view(torch.int32))
        self.reference(self.image.reshape(3, 1, 1, 4))
        self.assertEqual(self.encoded, 3)

    def test_weight_buffer_config_precision_and_tiling_changes_invalidate(self):
        changes = [
            lambda: self.pipe.vae.encoder.weight.add_(1),
            lambda: self.pipe.vae.pixel_mean.add_(1),
            lambda: setattr(self.pipe.vae, "_model_dtype", torch.float16),
            lambda: self.pipe.vae.config.update(clip_length=18),
            lambda: setattr(self.pipe.vae, "use_tiling", False),
            lambda: setattr(self.pipe.vae, "tile_sample_min_width", 512),
            lambda: setattr(self.pipe.vae, "tile_sample_min_overlap_height", 32),
            lambda: setattr(self.namespace["kernel_policy"], "precision", "strict"),
            lambda: setattr(self.namespace["kernel_policy"], "families", "native"),
            lambda: self.namespace["offload"].shared_state.update(_attention="sdpa"),
            lambda: setattr(self.pipe.vae, "encoder", torch.nn.Linear(1, 1, bias=False)),
        ]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                self.reference()
                before = self.encoded
                with torch.no_grad():
                    change()
                self.reference()
                self.assertEqual(self.encoded, before + 1)

    def test_new_vae_and_explicit_invalidation_reencode(self):
        self.reference()
        self.pipe.vae = ReferenceVAE()
        self.reference()
        self.pipe.clear_image_latent_cache()
        self.reference()
        self.assertEqual(self.encoded, 3)

    def test_shared_component_handoff_and_detach_clear_retained_latent(self):
        self.reference()
        shared, private = object(), object()
        self.pipe.set_offload_handoff(shared, private)
        self.reference()
        self.pipe.set_offload_handoff(shared, private)
        self.reference()
        self.assertEqual(self.encoded, 2)
        self.pipe.detach_shared_components()
        self.assertIsNone(self.pipe._image_latent_cache)

    def test_multiframe_images_never_reuse_cache(self):
        video = self.image.repeat(1, 2, 1, 1)
        self.reference(video)
        self.reference(video)
        self.assertEqual(self.encoded, 2)
        self.assertEqual(self.pipe.image_latent_cache_stats["bypasses"], 2)

    def test_non_cpu_prepared_image_bypasses_without_copying_it_to_cpu(self):
        self.reference()
        # Meta tensors exercise the non-CPU guard without querying or using a GPU.
        image = torch.empty((3, 1, 2, 2), device="meta")
        self.pipe._encode_video = lambda video: torch.zeros((1, 2, 1, 2, 2))
        self.reference(image)
        self.assertEqual(self.pipe.image_latent_cache_stats["bypasses"], 1)
        self.assertIsNone(self.pipe._image_latent_cache)

    def test_over_bound_latent_bypasses_storage(self):
        self.namespace["H3_IMAGE_LATENT_CACHE_MAX_BYTES"] = 4
        self.reference()
        self.reference()
        self.assertEqual(self.encoded, 2)
        self.assertIsNone(self.pipe._image_latent_cache)

    def test_cancelled_hit_still_checks_abort_and_failed_miss_is_not_stored(self):
        self.reference()
        self.abort = True
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.reference()
        self.abort = False
        self.failure = RuntimeError("encoder failed")
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            self.reference(self.image + 1)
        self.failure = None
        self.reference(self.image + 1)
        self.assertEqual(self.encoded, 3)

    def test_unknown_parameter_version_bypasses_instead_of_storing(self):
        with torch.inference_mode():
            weight = torch.nn.Parameter(torch.ones((1, 1)))
        self.pipe.vae.encoder.weight = weight
        self.reference()
        self.reference()
        self.assertEqual(self.encoded, 2)
        self.assertEqual(self.pipe.image_latent_cache_stats["bypasses"], 2)

    def test_cache_never_advances_task_rng_or_skips_fresh_condition_noise(self):
        state = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(5090)
        task_state = generator.get_state().clone()
        first = self.reference()
        second = self.reference()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(task_state, generator.get_state()))
        a, _ = self.pipe._prepare_condition_rows([first], [], generator)
        b, _ = self.pipe._prepare_condition_rows([second], [], generator)
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.equal(task_state, generator.get_state()))


if __name__ == "__main__":
    unittest.main()
