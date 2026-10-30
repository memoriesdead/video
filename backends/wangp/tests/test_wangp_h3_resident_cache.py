"""Restore canonical CPU VAE tensors using actual MMGP unload logic; no CUDA calls."""
import ast
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_wangp_h3_image_latent_cache as cache_tests


RUNTIME = Path(os.environ["WANGP_TEST_RUNTIME"]).resolve()


class ResidentReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = cache_tests.H3ImageLatentCacheTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.pipe = self.fixture.pipe
        self.assertTrue(hasattr(self.pipe, "_restore_image_cache_cpu_state"), "resident VAE CPU restoration is missing")
        self.pipe.vae.decoder = torch.nn.Linear(1, 1, bias=False)
        self.original_decoder = self.pipe.vae.decoder.weight
        self.original_encoder = self.pipe.vae.encoder.weight
        self.pipe._local_batch_h3_gpu_residency = True
        self.env = patch.dict(os.environ, {"WANGP_LOCAL_BATCH_GPU_RESIDENCY": "1", "WGP_H3_VSA_GRAPHS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.graph_entries = {"existing_graph": object()}
        self.clears = []
        names = ["transformer", "vae", "audio_vae", "video_encoder"]
        self.manager = SimpleNamespace(active_models_ids=[], active_models=[], state_rings=[], expert_residency={},
            ring=None, pool_models=False, preloaded_blocks_per_model={name: [] for name in names},
            loaded_blocks={name: None for name in names}, inflight={}, auto_preload=None, block_slots={}, slot_views={},
            models={name: object() for name in names}, wait_prefetch=lambda: None, _drops_pages=lambda model: False)
        self.manager.cuda_graphs = {"transformer": SimpleNamespace(clear=self.clear_graphs)}
        self.manager.gpu_unload_blocks = self.restore_block
        source = RUNTIME / "mmgp/offload.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "offload")
        fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "unload_all")
        namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)),
            "runtime": SimpleNamespace(_unload_resources=lambda manager, ids: None),
            "_pool": SimpleNamespace(make_room=lambda manager: None), "gc": SimpleNamespace(collect=lambda: None), "time": time}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        self.manager.unload_all = MethodType(namespace["unload_all"], self.manager)
        self.fixture.namespace["runtime"] = SimpleNamespace(_owner=self.owner)

    def owner(self, module):
        if module is self.pipe.vae.decoder:
            return self.manager, "vae"
        if module is self.pipe.vae.encoder:
            return self.manager, "video_encoder"
        return None

    def clear_graphs(self):
        self.clears.append(True)
        self.graph_entries.clear()

    def restore_block(self, model_id, block):
        self.assertIsNone(block)
        if model_id == "vae":
            self.pipe.vae.decoder.weight = self.original_decoder
        elif model_id == "video_encoder":
            self.pipe.vae.encoder.weight = self.original_encoder

    def install_resident_copies(self):
        with torch.inference_mode():
            self.pipe.vae.decoder.weight = torch.nn.Parameter(self.original_decoder.clone())
            self.pipe.vae.encoder.weight = torch.nn.Parameter(self.original_encoder.clone())
        self.manager.active_models_ids = ["transformer", "vae", "audio_vae", "video_encoder"]

    def test_restore_original_versioned_cpu_tensors_hits_cache_and_preserves_dit_graphs(self):
        first = self.fixture.reference()
        self.install_resident_copies()
        with self.assertRaisesRegex(RuntimeError, "version counter"):
            _ = self.pipe.vae.decoder.weight._version
        second = self.fixture.reference()
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(self.fixture.encoded, 1)
        self.assertEqual(self.pipe.image_latent_cache_stats, {"hits": 1, "misses": 1, "bypasses": 0})
        self.assertIs(self.pipe.vae.decoder.weight, self.original_decoder)
        self.assertIs(self.pipe.vae.encoder.weight, self.original_encoder)
        self.assertEqual(self.manager.active_models_ids, ["transformer", "audio_vae"])
        self.assertEqual(len(self.graph_entries), 1)
        self.assertEqual(self.clears, [])

    def test_canonical_cpu_weight_version_change_remains_a_cache_miss(self):
        self.fixture.reference()
        with torch.no_grad():
            self.original_decoder.add_(1)
        self.install_resident_copies()
        self.fixture.reference()
        self.assertEqual(self.fixture.encoded, 2)
        self.assertEqual(self.pipe.image_latent_cache_stats, {"hits": 0, "misses": 2, "bypasses": 0})
        self.assertEqual(len(self.graph_entries), 1)

    def test_default_off_retains_unknown_version_bypass_and_performs_no_unload(self):
        self.fixture.reference()
        self.install_resident_copies()
        self.pipe._local_batch_h3_gpu_residency = False
        self.fixture.reference()
        self.assertEqual(self.pipe.image_latent_cache_stats["bypasses"], 1)
        self.assertIn("vae", self.manager.active_models_ids)
        self.assertEqual(len(self.graph_entries), 1)

    def test_foreign_encoder_manager_is_never_unloaded(self):
        self.fixture.reference()
        self.install_resident_copies()
        foreign = SimpleNamespace()
        self.fixture.namespace["runtime"]._owner = lambda module: (
            (self.manager, "vae") if module is self.pipe.vae.decoder else (foreign, "video_encoder"))
        self.fixture.reference()
        self.assertEqual(self.pipe.image_latent_cache_stats["bypasses"], 1)
        self.assertIn("vae", self.manager.active_models_ids)
        self.assertEqual(len(self.graph_entries), 1)

    def test_abort_and_unload_errors_do_not_return_a_cached_latent(self):
        self.fixture.reference()
        self.install_resident_copies()
        self.fixture.abort = True
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.fixture.reference()
        self.assertIn("vae", self.manager.active_models_ids)
        self.fixture.abort = False
        self.manager.unload_all = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("restore failed"))
        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            self.fixture.reference()
        self.assertEqual(self.pipe.image_latent_cache_stats["hits"], 0)


if __name__ == "__main__":
    unittest.main()
