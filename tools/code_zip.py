#!/usr/bin/env python3
"""Pack the course's Rust code (code/) into downloads/imageanalysis-code.zip for students.

The zip holds the library (imgcourse/), the starter project and one project per unit with
coding exercises, as a Cargo workspace in a folder imageanalysis-code/. Build products
(target/, out/) are left out. The zip is deterministic (sorted names, fixed timestamps),
so an unchanged code/ gives a byte-identical zip and no change in git.
Run by `make code-zip` (and by `make render`, which copies downloads/ into docs/).
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'code')
OUT = os.path.join(ROOT, 'downloads', 'imageanalysis-code.zip')
TOP = 'imageanalysis-code'
SKIP_DIRS = {'target', 'out', '.git'}
SKIP_FILES = {'.DS_Store'}


def files():
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f not in SKIP_FILES:
                yield os.path.join(dirpath, f)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    n = 0
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files():
            rel = os.path.relpath(path, SRC).replace(os.sep, '/')
            info = zipfile.ZipInfo(f'{TOP}/{rel}', date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(path, 'rb') as fh:
                z.writestr(info, fh.read())
            n += 1
    old = open(OUT, 'rb').read() if os.path.exists(OUT) else None
    new = open(tmp, 'rb').read()
    if old == new:
        os.remove(tmp)
    else:
        os.replace(tmp, OUT)
    print(f'{os.path.relpath(OUT, ROOT)}: {n} files, {len(new) // 1024} KB')


if __name__ == '__main__':
    sys.exit(main())
