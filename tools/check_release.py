#!/usr/bin/env python3
"""Check an explicitly approved, source-only release without exposing file contents.

Build: --build-manifest approved-paths.json (a separately reviewed JSON path array).
Check: --check [--archive source.zip] [--check-git]. No upload operations exist.
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


def check_git(root, expected):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout
    failures = []
    if Path(os.fsdecode(git("rev-parse", "--show-toplevel")).strip()).resolve() != root:
        return [finding(".git", "git-root-mismatch")]
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
    count = int(git("rev-list", "--all", "--count").strip() or b"0")
    if count > 1:
        failures.append(finding(".git", "git-history"))
    elif count == 1:
        committed = {}
        for record in git("ls-tree", "-rz", "HEAD").split(b"\0"):
            if record:
                metadata, path = record.split(b"\t", 1)
                mode, kind, object_id = metadata.split()
                path = os.fsdecode(path)
                if kind != b"blob" or mode not in {b"100644", b"100755"}:
                    failures.append(finding(path, "git-commit-mismatch"))
                committed[path] = object_id.decode("ascii")
        if committed != hashes:
            failures.append(finding(".git", "git-commit-mismatch"))
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
    args = parser.parse_args(argv)
    failures = []
    try:
        if not valid_path(args.manifest) or "/" in args.manifest or forbidden(args.manifest):
            raise ValueError("invalid-manifest-path")
        root = args.root.absolute()
        if not root.is_dir() or linked(root):
            raise ValueError("invalid-root")
        root = root.resolve()
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
                    failures.extend(check_git(root, expected))
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError):
        failures.append(finding("[gate]", "invalid-input-or-io"))
    failures.sort(key=lambda f: (f["path"], f["line"], f["category"]))
    print(json.dumps({"ok": not failures, "findings": failures}, sort_keys=True))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
