"""CPU-only checks for the source patch installer; no model or runtime launch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('engine_cli', Path(__file__).parents[1] / 'tools/engine.py')
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / 'checkout'
        self.checkout.mkdir()
        self.git('init', '-q')
        (self.checkout / 'source.py').write_text('value = 1\n')
        self.git('add', 'source.py')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        self.revision = self.git('rev-parse', 'HEAD').strip()
        (self.checkout / 'source.py').write_text('value = 2\n')
        patch = self.git('diff', '--', 'source.py')
        self.git('checkout', '--', 'source.py')
        self.backend = self.root / 'backend'
        self.backend.mkdir()
        (self.backend / 'change.patch').write_text(patch, encoding='utf-8')
        self.meta = {'id': 'fixture', 'upstream_url': 'https://github.com/example/fixture.git',
                     'revision': self.revision, 'patches': [{'path': 'change.patch',
                     'sha256': hashlib.sha256((self.backend / 'change.patch').read_bytes()).hexdigest()}],
                     'patch_target_allowlist': ['source.py'], 'capabilities': ['Fixture'],
                     'weight_files_modified': False}
        (self.backend / 'backend.json').write_text(json.dumps(self.meta))

    def git(self, *args):
        result = subprocess.run(['git', '-C', str(self.checkout), *args], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_check_is_read_only_and_apply_preserves_untracked_weights(self):
        weights = self.checkout / 'model.safetensors'
        weights.write_bytes(bytes(range(256)))
        before = hashlib.sha256(weights.read_bytes()).hexdigest()
        record = engine.load_backend(self.backend)
        engine.install(record, self.checkout, apply=False)
        self.assertEqual((self.checkout / 'source.py').read_text(), 'value = 1\n')
        engine.install(record, self.checkout, apply=True)
        self.assertEqual((self.checkout / 'source.py').read_text(), 'value = 2\n')
        self.assertEqual(hashlib.sha256(weights.read_bytes()).hexdigest(), before)

    def test_wrong_revision_is_rejected_without_writes(self):
        record = engine.load_backend(self.backend)
        record['revision'] = '0' * 40
        with self.assertRaisesRegex(engine.ReleaseError, 'revision'):
            engine.install(record, self.checkout, apply=True)
        self.assertEqual((self.checkout / 'source.py').read_text(), 'value = 1\n')

    def test_dirty_source_is_preserved(self):
        record = engine.load_backend(self.backend)
        (self.checkout / 'source.py').write_text('user edit\n')
        with self.assertRaisesRegex(engine.ReleaseError, 'dirty'):
            engine.install(record, self.checkout, apply=True)
        self.assertEqual((self.checkout / 'source.py').read_text(), 'user edit\n')

    def test_changed_patch_or_unapproved_target_is_rejected(self):
        record = engine.load_backend(self.backend)
        (self.backend / 'change.patch').write_text('changed')
        with self.assertRaises(engine.ReleaseError):
            engine.install(record, self.checkout, apply=True)
        with self.assertRaises(engine.ReleaseError):
            engine.patch_targets('diff --git a/../outside.py b/../outside.py\n')
        with self.assertRaises(engine.ReleaseError):
            engine.patch_targets('diff --git a/model.safetensors b/model.safetensors\n')


if __name__ == '__main__':
    unittest.main()
