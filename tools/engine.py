"""Apply pinned video engine source patches; stored model weights remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {'.py', '.cu', '.cuh', '.cpp', '.c', '.h', '.hpp', '.cmake'}


class ReleaseError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def relative_path(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value,
            'Use a portable relative source path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in ('', '.', '..') for p in value.split('/')),
            'Source path escapes the checkout')
    require(not any(p.startswith('.') for p in path.parts), 'Hidden metadata is outside patch scope')
    return path


def ordinary_path(root, value):
    path = relative_path(value)
    current = Path(root)
    for part in path.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'A source path contains a symlink or reparse point')
    require(current.resolve().is_relative_to(Path(root).resolve()), 'Source path escapes its root')
    return current


def patch_targets(text):
    require('GIT binary patch' not in text and '\nBinary files ' not in text,
            'Only readable source patches are supported')
    targets = set()
    for line in text.splitlines():
        if line.startswith('diff --git '):
            parts = shlex.split(line)
            require(len(parts) == 4 and parts[2].startswith('a/') and parts[3].startswith('b/'),
                    'Malformed source patch header')
            old, new = parts[2][2:], parts[3][2:]
            require(old == new, 'Renames are outside the source-only patch scope')
            path = relative_path(new)
            require(path.suffix in SOURCE_SUFFIXES or path.name == 'CMakeLists.txt',
                    'Patch target is not an approved source file type')
            targets.add(new)
        elif line.startswith(('new file mode ', 'old mode ', 'new mode ')):
            require(line.split()[-1] in ('100644', '100755'), 'Only regular source files are supported')
    require(targets, 'Patch contains no source files')
    return targets


def load_backend(directory):
    directory = Path(directory).resolve()
    data = json.loads((directory / 'backend.json').read_text(encoding='utf-8-sig'))
    require(re.fullmatch(r'[a-z0-9-]+', data.get('id', '')) is not None, 'Invalid backend ID')
    require(re.fullmatch(r'[0-9a-f]{40}', data.get('revision', '')) is not None, 'Invalid pinned revision')
    require(data.get('weight_files_modified') is False, 'Backend must preserve model files')
    require(data.get('upstream_url', '').startswith('https://github.com/'), 'Expected a public upstream URL')
    require(isinstance(data.get('patches'), list) and data['patches'], 'Backend has no source patches')
    allowlist = data.get('patch_target_allowlist', [])
    require(isinstance(allowlist, list) and len(set(allowlist)) == len(allowlist) and allowlist,
            'Backend needs an exact source target allowlist')
    for value in allowlist:
        relative_path(value)
    data['_directory'] = directory
    verify_patches(data)
    return data


def verify_patches(record):
    patches, targets = [], set()
    for item in record['patches']:
        path = ordinary_path(record['_directory'], item['path'])
        require(path.suffix == '.patch' and path.is_file(), 'Missing source patch')
        value = path.read_bytes()
        require(hashlib.sha256(value).hexdigest() == item['sha256'], 'Source patch SHA256 changed')
        targets.update(patch_targets(value.decode('utf-8')))
        patches.append(path)
    require(targets == set(record['patch_target_allowlist']), 'Patch targets differ from the approved allowlist')
    return patches


def git(checkout, *args):
    result = subprocess.run(['git', '-C', str(checkout), '-c', 'core.fsmonitor=false', *args],
                            capture_output=True, text=True, encoding='utf-8', timeout=120)
    if result.returncode:
        raise ReleaseError(result.stderr.strip() or result.stdout.strip() or 'Git command failed')
    return result.stdout.strip()


def install(record, checkout, apply=False):
    checkout = Path(checkout).resolve()
    require(checkout.is_dir(), 'Backend checkout directory does not exist')
    require(Path(git(checkout, 'rev-parse', '--show-toplevel')).resolve() == checkout,
            'Use the backend repository root')
    require(git(checkout, 'rev-parse', 'HEAD') == record['revision'], 'Checkout revision differs from the pinned revision')
    require(not git(checkout, 'status', '--porcelain', '--untracked-files=no'),
            'Checkout has dirty tracked source; existing edits are preserved')
    patches = verify_patches(record)
    for target in record['patch_target_allowlist']:
        ordinary_path(checkout, target)
    git(checkout, 'apply', '--check', '--whitespace=error-all', *map(str, patches))
    if apply:
        git(checkout, 'apply', '--whitespace=error-all', *map(str, patches))
    return {'backend': record['id'], 'revision': record['revision'], 'applied': apply,
            'source_targets': record['patch_target_allowlist'], 'weight_files_modified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'check', 'apply'))
    parser.add_argument('backend', nargs='?', choices=('wangp',))
    parser.add_argument('--checkout', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == 'list':
            records = [load_backend(p) for p in sorted((ROOT / 'backends').iterdir()) if p.is_dir()]
            print(json.dumps([{k: r[k] for k in ('id', 'revision', 'capabilities')} for r in records], indent=2))
            return 0
        require(args.backend and args.checkout, 'Specify a backend and --checkout')
        record = load_backend(ROOT / 'backends' / args.backend)
        print(json.dumps(install(record, args.checkout, apply=args.action == 'apply'), indent=2))
        return 0
    except (ReleaseError, OSError, KeyError, ValueError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'error': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
