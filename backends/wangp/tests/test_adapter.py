"""Standard-library fixtures: no runtime import, model access, network, or GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("public_wangp_batch", Path(__file__).resolve().parents[1] / "run_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def task(seed=1, **changes):
    params = {"model_type": batch.H3_MODEL, "config": "nvfp4_awq,int8_convrot",
              "overrides": "memory_profile=0, compilation='cudagraph'",
              "num_inference_steps": 8, "seed": seed}
    params.update(changes)
    return {"id": str(seed), "params": params}


class AdapterTests(unittest.TestCase):
    def fixture(self, *, invalid=False, complete=True):
        events = []
        manager = SimpleNamespace(unload_all=lambda: events.append("unload"))
        model = SimpleNamespace(wan_model=None, offloadobj=manager)

        def validate(entry, state):
            events.append("validate:" + entry["id"])
            return (None, "invalid") if invalid and entry["id"] == "1" else (entry, None)

        model.validate_task = validate
        result = SimpleNamespace(success=True)
        job = SimpleNamespace(done=complete, events=SimpleNamespace(iter=lambda **kwargs: iter(())),
                              cancel=lambda: events.append("cancel"))

        def wait():
            events.append("wait")
            job.done = True
            return result

        job.result = wait

        def submit(entries):
            events.append("submit")
            self.assertEqual(events[:2], ["validate:1", "validate:2"])
            return job

        session = SimpleNamespace(_ensure_runtime=lambda: SimpleNamespace(module=model), _state={},
            _normalize_source=lambda entries, **kwargs: entries,
            _ensure_task_client_ids=lambda entries: None, submit_manifest=submit,
            close=lambda: events.append("close"))
        return session, job, events

    def test_all_tasks_validate_before_submission_and_release(self):
        session, job, events = self.fixture()
        self.assertTrue(batch.run_session(session, [task(), task(2)], Path("."), keep=True).success)
        self.assertEqual(events, ["validate:1", "validate:2", "submit", "wait", "unload", "close"])
        self.assertNotIn(batch.RESIDENCY_ENV, os.environ)

    def test_one_invalid_task_prevents_every_render_and_still_validates_all(self):
        session, job, events = self.fixture(invalid=True)
        with self.assertRaisesRegex(ValueError, "validation"):
            batch.run_session(session, [task(), task(2)], Path("."))
        self.assertEqual(events, ["validate:1", "validate:2", "close"])

    def test_dry_run_never_submits_or_sets_retention(self):
        session, job, events = self.fixture()
        self.assertIsNone(batch.run_session(session, [task(), task(2)], Path("."), dry_run=True))
        self.assertEqual(events, ["validate:1", "validate:2", "close"])

    def test_incomplete_job_cancel_and_join_precede_unload(self):
        session, job, events = self.fixture(complete=False)
        runtime = session._ensure_runtime()
        with patch.dict(os.environ, {batch.RESIDENCY_ENV: "1"}):
            batch.cleanup(session, runtime, job, True)
            self.assertNotIn(batch.RESIDENCY_ENV, os.environ)
        self.assertEqual(events, ["cancel", "wait", "unload", "close"])

    def test_cancel_failure_skips_parallel_unload_and_closes_under_upstream_lock(self):
        session, job, events = self.fixture(complete=False)
        job.cancel = Mock(side_effect=RuntimeError("cancel failed"))
        with self.assertRaisesRegex(RuntimeError, "cancel failed"):
            batch.cleanup(session, session._ensure_runtime(), job, True)
        self.assertEqual(events, ["close"])

    def test_failed_initialization_is_not_retried_by_close(self):
        session, job, events = self.fixture()
        session._ensure_runtime = Mock(side_effect=RuntimeError("init failed"))
        with self.assertRaisesRegex(RuntimeError, "init failed"):
            batch.run_session(session, [task()], Path("."))
        self.assertEqual(events, [])
        session._ensure_runtime.assert_called_once()

    def test_unload_failure_still_closes(self):
        session, job, events = self.fixture()
        runtime = session._ensure_runtime()
        runtime.module.offloadobj.unload_all = Mock(side_effect=RuntimeError("unload failed"))
        with self.assertRaisesRegex(RuntimeError, "unload failed"):
            batch.cleanup(session, runtime, job, True)
        self.assertEqual(events, ["close"])

    def test_submission_failure_still_releases(self):
        session, job, events = self.fixture()
        session.submit_manifest = Mock(side_effect=RuntimeError("submit failed"))
        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            batch.run_session(session, [task(), task(2)], Path("."), keep=True)
        self.assertEqual(events, ["validate:1", "validate:2", "unload", "close"])

    def test_retention_requires_supported_identical_h3_tasks_and_literal_options(self):
        batch.validate_retention([task(), task(2)], True, True)
        for tasks, graphs in [([task()], False), ([task(model_type="another_backend")], True),
                              ([task(), task(2, prompt="different")], True),
                              ([task(overrides="compilation=call()")], True),
                              ([task(overrides="compilation='cudagraph', memory_profile=0, **opts")], True),
                              ([task(num_inference_steps=9)], True),
                              ([task(image_mode=1)], True)]:
            with self.subTest(tasks=tasks, graphs=graphs), self.assertRaises(ValueError):
                batch.validate_retention(tasks, True, graphs)
        batch.validate_retention([task(model_type="another_backend")], False, False)

    def test_cpu_caps_and_offline_flags_restore_parent_environment(self):
        before = dict(os.environ)
        with batch.local_environment(True, True):
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "RAYON_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
                self.assertEqual(os.environ[key], "4")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["WGP_H3_VSA_GRAPHS"], "1")
            self.assertNotIn(batch.RESIDENCY_ENV, os.environ)
        self.assertEqual(dict(os.environ), before)

    def test_default_environment_cannot_inherit_retention_or_graph_opt_in(self):
        with patch.dict(os.environ, {batch.RESIDENCY_ENV: "1", "WGP_H3_VSA_GRAPHS": "1"}):
            with batch.local_environment():
                self.assertNotIn(batch.RESIDENCY_ENV, os.environ)
                self.assertEqual(os.environ["WGP_H3_VSA_GRAPHS"], "0")
            self.assertEqual(os.environ[batch.RESIDENCY_ENV], "1")

    def test_reentrant_session_rejected_and_lock_released_after_failure(self):
        with batch.serial_session():
            with self.assertRaisesRegex(RuntimeError, "one WanGP"):
                with batch.serial_session():
                    self.fail("reentrant session entered")
        with batch.serial_session():
            pass

    def test_missing_ffmpeg_is_local_error_before_any_runtime_import(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "no download"):
                batch.local_ffmpeg_setup(Path("local-binaries"))

    def test_manifest_ids_are_checked_before_runtime_loading(self):
        for payload in ([], {}, [task(), task()], [{"id": "x"}]):
            with patch.object(Path, "read_text", return_value=json.dumps(payload)):
                with self.assertRaises(ValueError):
                    batch.load_tasks(Path("manifest.json"))

    def test_normalization_does_not_mutate_original_tasks(self):
        session, job, events = self.fixture()
        original = [task(), task(2)]
        saved = copy.deepcopy(original)
        def normalize(entries, **kwargs):
            entries[0]["params"]["new"] = True
            return entries
        session._normalize_source = normalize
        batch.run_session(session, original, Path("."), dry_run=True)
        self.assertEqual(original, saved)

    def test_source_hash_accepts_equivalent_crlf_checkout(self):
        metadata = {"revision": "pinned", "patched_source_sha256": {
            "source.py": hashlib.sha256(b"value = 1\n").hexdigest()}}
        def read(path, **kwargs):
            return json.dumps(metadata) if path.name == "backend.json" else "value = 1\r\n"
        with patch.object(batch.subprocess, "check_output", return_value="pinned\n"), \
                patch.object(Path, "read_text", read):
            batch.verify_runtime(Path("runtime"))

    def test_wrong_source_revision_rejects(self):
        with patch.object(batch.subprocess, "check_output", return_value="wrong\n"), \
                patch.object(Path, "read_text", return_value=json.dumps({"revision": "pinned"})):
            with self.assertRaisesRegex(ValueError, "revision"):
                batch.verify_runtime(Path("runtime"))

    def test_source_mutation_rejects(self):
        metadata = {"revision": "pinned", "patched_source_sha256": {"source.py": "0" * 64}}
        def read(path, **kwargs):
            return json.dumps(metadata) if path.name == "backend.json" else "changed\n"
        with patch.object(batch.subprocess, "check_output", return_value="pinned\n"), \
                patch.object(Path, "read_text", read):
            with self.assertRaisesRegex(ValueError, "source differs"):
                batch.verify_runtime(Path("runtime"))


if __name__ == "__main__":
    unittest.main()
