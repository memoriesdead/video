"""CPU-only release boundary fixtures; every filesystem mutation uses a temp tree."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_release.py"


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "payload"
        self.root.mkdir()
        spec = importlib.util.spec_from_file_location("release_gate", SCRIPT)
        self.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gate)
        self.paths = ["README.md", *self.gate.REQUIRED_FILES]
        for name in self.paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Public source and license notice.\n", encoding="utf-8")
        self.approved = self.base / "approved.json"

    def run_gate(self, *args):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--root", str(self.root), *args],
            capture_output=True, text=True, timeout=20,
        )
        return result, json.loads(result.stdout)

    def build(self):
        self.approved.write_text(json.dumps(self.paths), encoding="utf-8")
        return self.run_gate("--build-manifest", str(self.approved))

    def assert_category(self, data, category):
        self.assertIn(category, [finding["category"] for finding in data["findings"]])

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True, check=True)

    def init_git(self):
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Release Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")

    def commit(self, message):
        self.git("add", "--all")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def history_file(self, approved):
        path = self.base / "history.json"
        path.write_text(json.dumps(approved))
        return str(path)

    def initial_history(self):
        self.assertEqual(0, self.build()[0].returncode)
        self.init_git()
        commit = self.commit("Initial fixture")
        pin = hashlib.sha256((self.root / "release-files.json").read_bytes()).hexdigest()
        return commit, pin

    def update_payload(self):
        (self.root / "README.md").write_text("Updated public source.\n")
        self.assertEqual(0, self.build()[0].returncode)

    def unchecked_manifest(self):
        entries = [{"path": name, "size": (self.root / name).stat().st_size,
                    "sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest()}
                   for name in sorted(self.paths)]
        path = self.root / "release-files.json"
        path.write_text(json.dumps({"schema_version": 1, "files": entries}))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_explicit_list_builds_deterministic_manifest(self):
        result, data = self.build()
        self.assertEqual(0, result.returncode, data)
        before = (self.root / "release-files.json").read_bytes()
        result, data = self.build()
        self.assertEqual(0, result.returncode, data)
        self.assertEqual(before, (self.root / "release-files.json").read_bytes())
        result, data = self.run_gate("--check")
        self.assertEqual(0, result.returncode, data)

    def test_unlisted_file_is_never_implicitly_approved(self):
        self.build()
        (self.root / "extra.py").write_text("print(1)\n")
        result, data = self.run_gate()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "unlisted-file")
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "unlisted-file")

    def test_secret_findings_never_echo_secret_content(self):
        secret = "hf_" + "A" * 32
        (self.root / "README.md").write_text("credential = " + secret + "\n")
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "credential")
        self.assertNotIn(secret, result.stdout + result.stderr)
        self.assertEqual({"path", "line", "category"}, set(data["findings"][0]))

    def test_private_paths_projects_sessions_and_assignments_fail(self):
        cases = [
            ("C:" + chr(92) + "Users" + chr(92) + "privateperson", "personal-path"),
            ("/hom" + "e/privateperson/project", "personal-path"),
            ("Stock" + "AI internal data", "private-project"),
            (str(uuid.uuid4()), "session-identifier"),
            ('api_' + 'key = "privatevalue123"', "credential"),
            ('{"api_' + 'key": "privatevalue123"}', "credential"),
            ('HF_' + 'TOKEN="privatevalue123"', "credential"),
            (json.dumps({"path": "C:" + chr(92) + "Users" + chr(92) + "privateperson"}), "personal-path"),
            ("https://example.invalid/file?" + "token=privatevalue123", "signed-url"),
        ]
        for content, category in cases:
            with self.subTest(category=category):
                (self.root / "README.md").write_text(content + "\n")
                result, data = self.build()
                self.assertNotEqual(0, result.returncode)
                self.assert_category(data, category)
                self.assertNotIn(content, result.stdout)

    def test_raw_prompt_json_is_rejected(self):
        name = "request.json"
        (self.root / name).write_text(json.dumps({"prompt": "private request text"}))
        self.paths.append(name)
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "raw-prompt")

    def test_licenses_urls_authors_and_public_hashes_are_accepted(self):
        public = "Copyright 2026 Upstream Authors <author@example.org>\n"
        public += "Source: https://github.com/public/upstream/blob/" + "a" * 40 + "/LICENSE\n"
        public += "SHA256: " + "b" * 64 + "\n"
        (self.root / "README.md").write_text(public)
        result, data = self.build()
        self.assertEqual(0, result.returncode, data)

    def test_explicit_root_git_attributes_is_permitted(self):
        (self.root / ".gitattributes").write_text("* -text\n")
        self.paths.append(".gitattributes")
        result, data = self.build()
        self.assertEqual(0, result.returncode, data)

    def test_path_escape_absolute_and_case_duplicate_paths_fail(self):
        for bad in ["../outside.py", "/outside.py", "C:/outside.py", "dir\\file.py", "README.md/../file.py"]:
            with self.subTest(path=bad):
                self.approved.write_text(json.dumps(self.paths + [bad]))
                result, data = self.run_gate("--build-manifest", str(self.approved))
                self.assertNotEqual(0, result.returncode)
                self.assert_category(data, "invalid-path")
        self.approved.write_text(json.dumps(self.paths + ["readme.md"]))
        result, data = self.run_gate("--build-manifest", str(self.approved))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "duplicate-path")

    def test_forbidden_payload_names_and_binary_content_fail(self):
        for bad in [".env", "models/test.py", "logs/run.txt", ".codex/config.toml", "weights.safetensors", "preview.mp4", "plugin.dll", "request.jsonl"]:
            with self.subTest(path=bad):
                path = self.root / bad
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private\n")
                self.paths.append(bad)
                result, data = self.build()
                self.assertNotEqual(0, result.returncode)
                self.assert_category(data, "forbidden-path")
                self.paths.pop()
                path.unlink()
                if path.parent != self.root:
                    path.parent.rmdir()
        path = self.root / "hidden.py"
        path.write_bytes(b"abc\x00def")
        self.paths.append("hidden.py")
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "non-text-file")

    def test_symlink_file_or_directory_fails(self):
        target = self.base / "outside.py"
        target.write_text("private\n")
        link = self.root / "link.py"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable")
        self.paths.append("link.py")
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "linked-path")

    def test_hash_size_and_required_notice_fail_closed(self):
        self.build()
        (self.root / "README.md").write_text("changed\n")
        result, data = self.run_gate()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "hash-mismatch")
        self.paths.remove(self.gate.REQUIRED_FILES[0])
        (self.root / self.gate.REQUIRED_FILES[0]).unlink()
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "required-file")

    def test_file_size_limit_precedes_reading(self):
        with (self.root / "README.md").open("wb") as stream:
            stream.truncate(self.gate.MAX_FILE_BYTES + 1)
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "file-size-limit")

    def test_total_payload_limit_and_manifest_limit(self):
        for number in range(11):
            name = "source" + str(number) + ".py"
            (self.root / name).write_bytes(b"#" * self.gate.MAX_FILE_BYTES)
            self.paths.append(name)
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "total-size-limit")
        self.approved.write_bytes(b" " * (self.gate.MAX_FILE_BYTES + 1))
        result, data = self.run_gate("--build-manifest", str(self.approved))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "file-size-limit")

    @unittest.skipUnless(os.name == "nt", "Windows junction fixture")
    def test_windows_directory_junction_is_not_followed(self):
        import _winapi
        target = self.base / "outside-directory"
        target.mkdir()
        (target / "private.py").write_text("private\n")
        link = self.root / "linked"
        _winapi.CreateJunction(str(target), str(link))
        self.addCleanup(link.rmdir)
        result, data = self.build()
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "linked-path")
        self.assertTrue((target / "private.py").exists())

    def make_zip(self, extra=None, changed=False):
        archive = self.base / "source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name in [*self.paths, "release-files.json"]:
                contents = (self.root / name).read_bytes()
                if changed and name == "README.md":
                    contents += b"changed"
                output.writestr(name, contents)
            if extra:
                output.writestr(extra, "private")
        return archive

    def test_archive_exact_members_hash_and_no_history(self):
        self.build()
        archive = self.make_zip()
        result, data = self.run_gate("--archive", str(archive))
        self.assertEqual(0, result.returncode, data)
        for leak in ["private.txt", ".git/config", "../outside.py"]:
            archive = self.make_zip(extra=leak)
            result, data = self.run_gate("--archive", str(archive))
            self.assertNotEqual(0, result.returncode)
            self.assertTrue(data["findings"])
        archive = self.make_zip(changed=True)
        result, data = self.run_gate("--archive", str(archive))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "archive-hash-mismatch")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_git_index_exact_allowlist_and_single_commit_history(self):
        self.build()
        self.init_git()
        git = self.git
        git("add", "--all")
        git("commit", "-qm", "Initial fixture")
        result, data = self.run_gate("--check-git")
        self.assertEqual(0, result.returncode, data)
        git("commit", "--allow-empty", "-qm", "Old history fixture")
        result, data = self.run_gate("--check-git")
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history")
        (self.root / "leak.txt").write_text("private\n")
        git("add", "leak.txt")
        (self.root / "leak.txt").unlink()
        result, data = self.run_gate("--check-git")
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-index-mismatch")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_clean_index_cannot_hide_a_private_committed_tree(self):
        self.build()
        self.init_git()
        git = self.git
        (self.root / "leak.txt").write_text("private\n")
        git("add", "--all")
        git("commit", "-qm", "Private tree fixture")
        git("rm", "leak.txt")
        result, data = self.run_gate("--check-git")
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-commit-mismatch")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_approved_history_allows_staged_update_and_two_commits(self):
        old, pin = self.initial_history()
        approved = self.history_file({old: pin})
        self.update_payload()
        self.git("add", "--all")
        result, data = self.run_gate("--check-git", "--approved-history", approved)
        self.assertEqual(0, result.returncode, data)
        self.commit("Public update")
        result, data = self.run_gate("--check-git", "--approved-history", approved)
        self.assertEqual(0, result.returncode, data)
        result, data = self.run_gate("--check-git")
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_approved_history_still_checks_current_head_and_index(self):
        old, pin = self.initial_history()
        approved = self.history_file({old: pin})
        self.update_payload()
        result, data = self.run_gate("--check-git", "--approved-history", approved)
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-index-content-mismatch")
        self.commit("Public update")
        self.update_payload()
        (self.root / "README.md").write_text("Another public change.\n")
        self.assertEqual(0, self.build()[0].returncode)
        self.git("add", "--all")
        result, data = self.run_gate("--check-git", "--approved-history", approved)
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-commit-mismatch")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_approved_history_rejects_private_content_even_with_valid_pin(self):
        secret = "hf_" + "B" * 32
        (self.root / "README.md").write_text(secret + "\n")
        pin = self.unchecked_manifest()
        self.init_git()
        old = self.commit("Private content fixture")
        self.update_payload()
        self.commit("Clean current payload")
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: pin}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "credential")
        self.assertNotIn(secret, result.stdout + result.stderr)

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_approved_history_rejects_unlisted_past_file(self):
        self.build()
        pin = hashlib.sha256((self.root / "release-files.json").read_bytes()).hexdigest()
        self.init_git()
        (self.root / "private.txt").write_text("unlisted past data\n")
        old = self.commit("Unlisted historical fixture")
        (self.root / "private.txt").unlink()
        self.update_payload()
        self.commit("Clean current payload")
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: pin}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-tree-mismatch")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_history_manifest_pin_and_payload_hash_are_both_checked(self):
        old, pin = self.initial_history()
        self.update_payload()
        self.commit("Public update")
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: "0" * 64}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-manifest-hash")
        self.git("checkout", "-q", "--orphan", "forged")
        document = json.loads((self.root / "release-files.json").read_text())
        document["files"][0]["sha256"] = "0" * 64
        (self.root / "release-files.json").write_text(json.dumps(document))
        forged_pin = hashlib.sha256((self.root / "release-files.json").read_bytes()).hexdigest()
        forged = self.commit("Forged manifest fixture")
        self.git("branch", "-D", "main")
        self.update_payload()
        self.commit("Current safe payload")
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({forged: forged_pin}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-payload-hash")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_history_rejects_unapproved_reachable_side_branch(self):
        old, pin = self.initial_history()
        self.update_payload()
        current = self.commit("Public update")
        self.git("checkout", "-qb", "unapproved-side", old)
        self.git("commit", "--allow-empty", "-qm", "Unapproved side branch")
        self.git("checkout", "-q", "main")
        self.assertEqual(current, self.git("rev-parse", "HEAD").stdout.strip())
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: pin}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-unapproved")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_history_limits_objects_before_reading_them(self):
        name = "oversized.py"
        (self.root / name).write_bytes(b"#" * (self.gate.MAX_FILE_BYTES + 1))
        self.paths.append(name)
        pin = self.unchecked_manifest()
        self.init_git()
        old = self.commit("Oversized old source fixture")
        self.paths.remove(name)
        (self.root / name).unlink()
        self.update_payload()
        self.commit("Current safe source")
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: pin}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-size-limit")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_history_rejects_grafts_and_shallow_boundaries(self):
        old, pin = self.initial_history()
        self.update_payload()
        head = self.commit("Public update")
        approved = self.history_file({old: pin})
        grafts = self.root / ".git" / "info" / "grafts"
        grafts.write_text(head + "\n")
        result, data = self.run_gate("--check-git", "--approved-history", approved)
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-grafts")
        grafts.unlink()
        (self.root / ".git" / "shallow").write_text(head + "\n")
        result, data = self.run_gate("--check-git", "--approved-history", approved)
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-shallow")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_history_approval_must_be_external_and_well_formed(self):
        old, pin = self.initial_history()
        self.update_payload()
        inside = self.root / "history.json"
        inside.write_text(json.dumps({old: pin}))
        self.paths.append("history.json")
        self.assertEqual(0, self.build()[0].returncode)
        self.commit("Public update")
        result, data = self.run_gate("--check-git", "--approved-history", str(inside))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "invalid-input-or-io")
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: "invalid"}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "invalid-input-or-io")

    @unittest.skipUnless(__import__("shutil").which("git"), "git unavailable")
    def test_history_commit_count_has_a_bounded_read_limit(self):
        old, pin = self.initial_history()
        for number in range(self.gate.MAX_HISTORY_COMMITS + 1):
            self.git("commit", "--allow-empty", "-qm", "Bound fixture " + str(number))
        result, data = self.run_gate("--check-git", "--approved-history", self.history_file({old: pin}))
        self.assertNotEqual(0, result.returncode)
        self.assert_category(data, "git-history-commit-limit")


if __name__ == "__main__":
    unittest.main()
