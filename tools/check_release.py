#!/usr/bin/env python3
"""Check an explicitly approved, source-only release without exposing file contents.

Build: --build-manifest approved-paths.json (a separately reviewed JSON path array).
Check: --check [--archive source.zip] [--check-git]. No upload operations exist.
Updates: add --approved-history external.json, mapping full commit IDs to the
SHA256 of each previously approved commit's release manifest. Old trees are audited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_HISTORY_COMMITS = 64
MAX_HISTORY_ENTRIES = 4096
MAX_HISTORY_BYTES = 64 * 1024 * 1024
REQUIRED_FILES = (
    "LICENSE", "LICENSES.md", "THIRD_PARTY_NOTICES.md",
    "backends/wangp/LICENSE.txt", "backends/wangp/mmgp-LICENSE.md",
)
FORBIDDEN_PARTS = {
    "models", "weights", "outputs", "inputs", "logs", "sessions", "transcripts",
    "prompts", "repro", "grafts", "node_modules", "__pycache__", "build", "dist",
    "target", "venv", "env", "benchmark_results", "results", "credentials",
}
FORBIDDEN_SUFFIXES = {
    ".exe", ".dll", ".so", ".a", ".lib", ".o", ".obj", ".pyc", ".pyo",
    ".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".ninfer",
    ".qus", ".onnx", ".npz", ".npy", ".zip", ".tar", ".gz", ".7z",
    ".mp4", ".webm", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".wav", ".mp3", ".flac", ".jsonl", ".log", ".sqlite",
    ".db", ".pkl", ".pickle", ".pem", ".key", ".pfx", ".p12",
}
TEXT_SUFFIXES = {
    ".py", ".ps1", ".sh", ".bat", ".cmd", ".cmake", ".cpp", ".cc", ".cxx",
    ".c", ".cu", ".cuh", ".h", ".hpp", ".hh", ".hxx", ".inc", ".in",
    ".patch", ".diff", ".json", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".rs", ".lock", ".jinja", ".j2", ".js", ".ts", ".html", ".css",
    ".make", ".mk", ".rc", ".s", ".metal", ".wgsl", ".vert", ".frag",
    ".comp", ".g4", ".properties", ".text", ".tcl",
}
TEXT_NAMES = {"license", "notice", "copying", "authors", "contributors", "makefile", "dockerfile"}
PATTERNS = {
    "personal-path": re.compile(r"(?i)(?:\b[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+|/hom[e]/[^\s/]+|/User[s]/[^\s/]+|/mnt/[a-z]/User[s]/|\\\\[^\s\\]+\\+Users\\+)"),
    "private-project": re.compile(r"(?i)\b(?:StockA[I]|polymarketweathe[r]|kalshiweathe[r]|fore[x]|ETH-Fleet-Keepaliv[e])\b|\b[DE]:[\\/]+"),
    "session-identifier": re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\bsession-20\d{6}[A-Za-z0-9_-]*"),
    "credential": re.compile(r"(?i)\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "signed-url": re.compile(r"(?i)(?:X-Amz-(?:Signature|Credential|Security-Token)=|[?&](?:token|access_token|signature|sig|se)=[^\s&\"']+)"),
    "account-settings": re.compile(r"(?i)(?:\.(?:claude|codex|claudex)[\\/]settings|apiKeyHelper\s*[:=])"),
}
ASSIGNMENT = re.compile(r"(?i)\b(?:[a-z0-9]+_)*(?:api[_-]?key|auth[_-]?token|access[_-]?token|password|secret[_-]?key|hf_token|github_token|authorization)[\"']?\s*[:=]\s*([\"'])([^\"'\r\n]{8,})\1")
RAW_PROMPT = re.compile(r'(?i)"(?:prompt|system_prompt|user_prompt|messages|conversation)"\s*:\s*(?:"[^"\s]|\[)')
PLACEHOLDERS = {"local-only", "your-api-key", "your_api_key", "placeholder", "changeme", "example-only"}


def finding(path, category, line=0):
    if PATTERNS["credential"].search(path):
        path = "[redacted-path]"
    return {"path": path, "line": line, "category": category}


def valid_path(value):
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and ":" not in value and not value.startswith("/")
            and not any(ord(c) < 32 for c in value)
            and all(part not in {"", ".", ".."} and part == part.rstrip(" .")
                    for part in value.split("/")))


def linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def forbidden(path):
    parts = PurePosixPath(path).parts
    lower = [p.casefold() for p in parts]
    for index, part in enumerate(lower):
        if part.startswith("."):
            allowed = (index == 0 and part in {".gitignore", ".gitattributes"} and len(parts) == 1)
            allowed |= (index == 0 and part == ".github" and len(parts) == 3
                        and lower[1] == "workflows" and Path(parts[-1]).suffix in {".yml", ".yaml"})
            if not allowed:
                return True
        if part in FORBIDDEN_PARTS or part.startswith(".env"):
            return True
    name = lower[-1]
    if name in {"auth.json", "credentials.json", "token", "token.json", "settings.local.json", "progress.md"}:
        return True
    suffix = Path(name).suffix
    return suffix in FORBIDDEN_SUFFIXES or not (suffix in TEXT_SUFFIXES or name in TEXT_NAMES or name in {".gitignore", ".gitattributes"})


def scan_text(path, payload):
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [finding(path, "non-text-file")]
    if "\x00" in text:
        return [finding(path, "non-text-file")]
    failures = []
    for number, line in enumerate(text.splitlines(), 1):
        if path.endswith(".json") and RAW_PROMPT.search(line):
            failures.append(finding(path, "raw-prompt", number))
        for category, pattern in PATTERNS.items():
            if pattern.search(line):
                failures.append(finding(path, category, number))
        for match in ASSIGNMENT.finditer(line):
            value = match.group(2)
            if value.casefold() not in PLACEHOLDERS and not value.startswith(("${", "<")):
                failures.append(finding(path, "credential", number))
    return failures


def inspect_tree(root, approved, manifest_name):
    failures, payloads, seen = [], {}, set()
    if linked(root):
        return [finding(".", "linked-path")], payloads
    total = 0
    for directory, subdirs, files in os.walk(root, followlinks=False):
        for name in list(subdirs):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if linked(path):
                failures.append(finding(relative, "linked-path"))
                subdirs.remove(name)
            elif relative == ".git":
                subdirs.remove(name)
            elif name.startswith(".") and relative != ".github":
                failures.append(finding(relative, "forbidden-path"))
                subdirs.remove(name)
            elif name.casefold() in FORBIDDEN_PARTS:
                failures.append(finding(relative, "forbidden-path"))
                subdirs.remove(name)
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if linked(path):
                failures.append(finding(relative, "linked-path"))
                continue
            if relative == ".git":
                failures.append(finding(relative, "forbidden-path"))
                continue
            if relative == manifest_name:
                continue
            seen.add(relative)
            if relative not in approved:
                failures.append(finding(relative, "unlisted-file"))
            if not valid_path(relative) or forbidden(relative):
                failures.append(finding(relative, "forbidden-path"))
                continue
            for category, pattern in PATTERNS.items():
                if pattern.search(relative):
                    failures.append(finding(relative, category))
            if not path.is_file():
                failures.append(finding(relative, "non-regular-file"))
                continue
            size = path.stat().st_size
            total += size
            if size > MAX_FILE_BYTES:
                failures.append(finding(relative, "file-size-limit"))
                continue
            if total > MAX_TOTAL_BYTES:
                failures.append(finding(relative, "total-size-limit"))
                continue
            content = path.read_bytes()
            failures.extend(scan_text(relative, content))
            payloads[relative] = content
    for path in sorted(approved - seen):
        failures.append(finding(path, "missing-file"))
    for path in REQUIRED_FILES:
        if path not in approved or path not in payloads:
            failures.append(finding(path, "required-file"))
    return failures, payloads


def validate_paths(paths, manifest_name):
    if not isinstance(paths, list) or not paths:
        return [finding("[approval-list]", "invalid-approval-list")]
    failures, seen = [], set()
    for path in paths:
        if not valid_path(path) or path == manifest_name:
            failures.append(finding("[invalid-path]", "invalid-path"))
        elif path.casefold() in seen:
            failures.append(finding(path, "duplicate-path"))
        else:
            seen.add(path.casefold())
    return failures


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def check_archive(archive, expected):
    failures, seen = [], set()
    with zipfile.ZipFile(archive) as source:
        total = 0
        for info in source.infolist():
            name = info.filename
            if not valid_path(name):
                failures.append(finding("[invalid-path]", "archive-invalid-path"))
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                failures.append(finding(name, "linked-path"))
            if name.casefold() in {p.casefold() for p in seen}:
                failures.append(finding(name, "archive-duplicate-path"))
            seen.add(name)
            if name not in expected:
                failures.append(finding(name, "archive-unlisted-file"))
                continue
            total += info.file_size
            if info.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                failures.append(finding(name, "archive-size-limit"))
                continue
            if source.read(info) != expected[name]:
                failures.append(finding(name, "archive-hash-mismatch"))
    for name in sorted(set(expected) - seen):
        failures.append(finding(name, "archive-missing-file"))
    return failures


def check_git(root, expected, approved_history=None, manifest_name="release-files.json"):
    def git(*args):
        return subprocess.run(["git", "--no-replace-objects", "-C", str(root), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
    failures = []
    if Path(os.fsdecode(git("rev-parse", "--show-toplevel")).strip()).resolve() != root:
        return [finding(".git", "git-root-mismatch")]
    grafts = Path(os.fsdecode(git("rev-parse", "--git-path", "info/grafts")).strip())
    if not grafts.is_absolute():
        grafts = root / grafts
    if grafts.exists() or os.environ.get("GIT_GRAFT_FILE"):
        return [finding(".git", "git-history-grafts")]
    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        return [finding(".git", "git-history-shallow")]
    algorithm = os.fsdecode(git("rev-parse", "--show-object-format")).strip()
    hashes = {path: hashlib.new(algorithm, b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
              for path, content in expected.items()}
    staged = {}
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.split()
            path = os.fsdecode(path)
            if stage != b"0" or path in staged or mode not in {b"100644", b"100755"}:
                failures.append(finding(path, "git-index-mismatch"))
            staged[path] = object_id.decode("ascii")
    if set(staged) != set(expected):
        failures.append(finding(".git", "git-index-mismatch"))
    else:
        for path in staged:
            if staged[path] != hashes[path]:
                failures.append(finding(path, "git-index-content-mismatch"))
    commits = git("rev-list", "--all", "--max-count=" + str(MAX_HISTORY_COMMITS + 2)).decode("ascii").splitlines()
    if len(commits) > MAX_HISTORY_COMMITS + 1:
        return failures + [finding(".git", "git-history-commit-limit")]
    if approved_history is None and len(commits) > 1:
        failures.append(finding(".git", "git-history"))
        return failures
    approved = approved_history or {}
    if len(approved) > MAX_HISTORY_COMMITS or any(len(commit) != hashlib.new(algorithm).digest_size * 2 for commit in approved):
        return failures + [finding(".git", "invalid-history-approval")]
    cache, read_bytes = {}, 0

    def read_object(object_id, kind, path):
        nonlocal read_bytes
        if object_id in cache:
            return cache[object_id]
        size = int(git("cat-file", "-s", object_id))
        if size > MAX_FILE_BYTES or read_bytes + size > MAX_HISTORY_BYTES:
            failures.append(finding(path, "git-history-size-limit"))
            return None
        content = git("cat-file", kind, object_id)
        if len(content) != size:
            raise ValueError("object-size-mismatch")
        read_bytes += size
        cache[object_id] = content
        return content

    def tree_files(commit):
        pending = [("", git("rev-parse", commit + "^{tree}").decode("ascii").strip())]
        files, entries = {}, 0
        oid_bytes = hashlib.new(algorithm).digest_size
        while pending:
            prefix, object_id = pending.pop()
            content = read_object(object_id, "tree", ".git")
            if content is None:
                return None
            position = 0
            while position < len(content):
                space = content.index(b" ", position)
                end = content.index(b"\0", space)
                mode = content[position:space]
                path = prefix + content[space + 1:end].decode("utf-8")
                object_id = content[end + 1:end + 1 + oid_bytes].hex()
                position = end + 1 + oid_bytes
                entries += 1
                if entries > MAX_HISTORY_ENTRIES:
                    failures.append(finding(".git", "git-history-entry-limit"))
                    return None
                if position > len(content) or not valid_path(path) or len(path) > 4096:
                    failures.append(finding("[invalid-path]", "git-history-invalid-path"))
                    return None
                if mode == b"40000":
                    pending.append((path + "/", object_id))
                elif mode in {b"100644", b"100755"}:
                    if path in files:
                        raise ValueError("duplicate-tree-path")
                    files[path] = object_id
                else:
                    failures.append(finding(path, "git-history-file-type"))
                    return None
        return files

    def audit_history(commit, files):
        if manifest_name not in files:
            failures.append(finding(manifest_name, "git-history-manifest"))
            return
        manifest = read_object(files[manifest_name], "blob", manifest_name)
        if manifest is None:
            return
        if digest(manifest) != approved[commit]:
            failures.append(finding(manifest_name, "git-history-manifest-hash"))
            return
        document = json.loads(manifest.decode("utf-8-sig"))
        if not isinstance(document, dict) or set(document) != {"schema_version", "files"} or type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("invalid-history-manifest")
        entries = document["files"]
        if not isinstance(entries, list) or any(not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"} for entry in entries):
            raise ValueError("invalid-history-manifest")
        paths = [entry["path"] for entry in entries]
        invalid = validate_paths(paths, manifest_name)
        failures.extend(invalid)
        if invalid:
            return
        if set(files) != set(paths) | {manifest_name}:
            failures.append(finding(".git", "git-history-tree-mismatch"))
            return
        for path in REQUIRED_FILES:
            if path not in paths:
                failures.append(finding(path, "required-file"))
        total = len(manifest)
        for entry in entries:
            path = entry["path"]
            if forbidden(path):
                failures.append(finding(path, "forbidden-path"))
                continue
            for category, pattern in PATTERNS.items():
                if pattern.search(path):
                    failures.append(finding(path, category))
            if type(entry["size"]) is not int or entry["size"] < 0 or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                failures.append(finding(path, "invalid-manifest-entry"))
                continue
            content = read_object(files[path], "blob", path)
            if content is None:
                return
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                failures.append(finding(path, "git-history-size-limit"))
                return
            if len(content) != entry["size"] or digest(content) != entry["sha256"]:
                failures.append(finding(path, "git-history-payload-hash"))
            failures.extend(scan_text(path, content))

    if commits:
        head = git("rev-parse", "HEAD").decode("ascii").strip()
        current = tree_files(head)
        if current is None:
            return failures
        historical = [commit for commit in commits if commit != head]
        if current != hashes:
            if head in approved:
                historical.append(head)
            else:
                failures.append(finding(".git", "git-commit-mismatch"))
        for commit in historical:
            if commit not in approved:
                failures.append(finding(".git", "git-history-unapproved"))
                continue
            files = current if commit == head else tree_files(commit)
            if files is not None:
                audit_history(commit, files)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="release-files.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check only (the default)")
    mode.add_argument("--build-manifest", type=Path, metavar="APPROVED_PATHS_JSON")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--check-git", action="store_true")
    parser.add_argument("--approved-history", type=Path, metavar="EXTERNAL_JSON",
                        help="map approved historical commit IDs to their manifest SHA256")
    args = parser.parse_args(argv)
    failures = []
    try:
        if not valid_path(args.manifest) or "/" in args.manifest or forbidden(args.manifest):
            raise ValueError("invalid-manifest-path")
        root = args.root.absolute()
        if not root.is_dir() or linked(root):
            raise ValueError("invalid-root")
        root = root.resolve()
        approved_history = None
        if args.approved_history:
            history_path = args.approved_history.resolve()
            if not args.check_git or history_path.is_relative_to(root) or linked(args.approved_history) or history_path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("invalid-history-approval-file")
            approved_history = json.loads(history_path.read_text(encoding="utf-8-sig"))
            if not isinstance(approved_history, dict) or len(approved_history) > MAX_HISTORY_COMMITS or any(
                not isinstance(commit, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
                or not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin)
                for commit, pin in approved_history.items()
            ):
                raise ValueError("invalid-history-approval")
        manifest_path = root / args.manifest
        if manifest_path.exists() and linked(manifest_path):
            raise ValueError("linked-manifest")
        json_path = args.build_manifest or manifest_path
        if json_path.stat().st_size > MAX_FILE_BYTES:
            print(json.dumps({"ok": False, "findings": [finding("[manifest-or-list]", "file-size-limit")]}, sort_keys=True))
            return 1
        if args.build_manifest:
            paths = json.loads(args.build_manifest.read_text(encoding="utf-8-sig"))
            entries = None
        else:
            document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if set(document) != {"schema_version", "files"} or document["schema_version"] != 1:
                raise ValueError("invalid-manifest")
            entries = document["files"]
            if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != {"path", "size", "sha256"} for e in entries):
                raise ValueError("invalid-manifest")
            paths = [entry["path"] for entry in entries]
        failures.extend(validate_paths(paths, args.manifest))
        if not failures:
            tree_failures, payloads = inspect_tree(root, set(paths), args.manifest)
            failures.extend(tree_failures)
            if entries is not None:
                for entry in entries:
                    path = entry["path"]
                    if type(entry["size"]) is not int or entry["size"] < 0 or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                        failures.append(finding(path, "invalid-manifest-entry"))
                    elif path in payloads and (len(payloads[path]) != entry["size"] or digest(payloads[path]) != entry["sha256"]):
                        failures.append(finding(path, "hash-mismatch"))
            if not failures and args.build_manifest:
                document = {"schema_version": 1, "files": [
                    {"path": p, "size": len(payloads[p]), "sha256": digest(payloads[p])}
                    for p in sorted(paths)]}
                manifest_path.write_text(json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
            if not failures:
                expected = {**payloads, args.manifest: manifest_path.read_bytes()}
                if len(expected[args.manifest]) > MAX_FILE_BYTES:
                    failures.append(finding(args.manifest, "file-size-limit"))
                if sum(map(len, expected.values())) > MAX_TOTAL_BYTES:
                    failures.append(finding(args.manifest, "total-size-limit"))
                if args.archive:
                    failures.extend(check_archive(args.archive, expected))
                if args.check_git:
                    failures.extend(check_git(root, expected, approved_history, args.manifest))
    except (OSError, ValueError, TypeError, KeyError, RecursionError, zipfile.BadZipFile, subprocess.SubprocessError):
        failures.append(finding("[gate]", "invalid-input-or-io"))
    failures.sort(key=lambda f: (f["path"], f["line"], f["category"]))
    print(json.dumps({"ok": not failures, "findings": failures}, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
