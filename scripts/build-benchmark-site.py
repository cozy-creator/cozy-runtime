#!/usr/bin/env python3
"""Assemble only the public benchmark files and verified original media."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

STATIC_FILES = ('index.html', 'styles.css', 'app.js', 'data.json', 'scaling.svg', 'media-manifest.json')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'docs/benchmarks/site'
    manifest = json.loads((source / 'media-manifest.json').read_text())
    if args.media_archive.stat().st_size != manifest['archive_bytes'] or digest(args.media_archive) != manifest['archive_sha256']:
        raise SystemExit('Media archive does not match the publication manifest')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory must be empty; existing files are never removed')
    output.mkdir(parents=True, exist_ok=True)
    expected = {f['path']: f for f in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise SystemExit('Duplicate manifest path')
    with tarfile.open(args.media_archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise SystemExit('Archive members do not match the explicit media list')
        for member in members:
            name = PurePosixPath(member.name)
            row = expected[member.name]
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.size != row['bytes']:
                raise SystemExit(f'Invalid media member: {member.name}')
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open('wb') as dest:
                shutil.copyfileobj(stream, dest)
            if digest(target) != row['sha256']:
                raise SystemExit(f'Media hash mismatch: {member.name}')
    for name in STATIC_FILES:
        shutil.copyfile(source / name, output / name)
    (output / '.nojekyll').touch()
    data = json.loads((output / 'data.json').read_text())
    for artifact in data['artifacts']:
        for key in ('video', 'poster'):
            if artifact[key] not in expected:
                raise SystemExit(f'Unpublished artifact reference: {artifact[key]}')
        if expected[artifact['video']]['sha256'] != artifact['sha256']:
            raise SystemExit(f'Artifact index hash mismatch: {artifact["id"]}')
    print(f'Built {len(data["artifacts"])} original clips and {len(STATIC_FILES)} static files in {output}')


if __name__ == '__main__':
    main()
