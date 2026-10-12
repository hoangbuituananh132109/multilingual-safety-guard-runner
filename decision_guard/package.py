"""Package only this new code and its two design documents, never models/data."""
import argparse
from pathlib import Path
import zipfile

from .data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    files = sorted(p for p in (root / 'decision_guard').rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.sh', '.md', '.json')
                   and '__pycache__' not in p.parts)
    files += [root / f'docs/superpowers/{folder}/2026-10-11-decision-guard-bridge.md'
              for folder in ('specs', 'plans')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Choose a new package filename, do not overwrite a transferred bundle')
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            content = file.read_bytes()
            if file.suffix == '.sh':
                content = content.replace(b'\r\n', b'\n')
            archive.writestr(file.relative_to(root).as_posix(), content)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Package CRC verification failed')
    print(f'files={len(files)} bytes={args.output.stat().st_size} sha256={sha256(args.output)}')


if __name__ == '__main__':
    main()
