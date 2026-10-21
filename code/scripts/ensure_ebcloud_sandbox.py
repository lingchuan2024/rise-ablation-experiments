#!/usr/bin/env python3
"""Restore a saved sandbox image after an ephemeral Docker daemon restart."""
import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True, help='Pinned sha256 image ID')
    p.add_argument('--archive', type=Path, default=Path('sandbox-image.tar'))
    args = p.parse_args()
    if not args.image.startswith('sha256:') or len(args.image) != 71:
        p.error('--image must be a full sha256 image ID')
    def exists():
        result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image], capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == args.image
    if not exists():
        if not args.archive.is_file():
            raise SystemExit(f'Sandbox image absent and archive missing: {args.archive}')
        subprocess.run(['docker', 'load', '--input', str(args.archive)], check=True)
        if not exists():
            raise SystemExit('Restored archive does not contain the pinned sandbox image')
    print(f'Sandbox image ready: {args.image}')


if __name__ == '__main__':
    main()
