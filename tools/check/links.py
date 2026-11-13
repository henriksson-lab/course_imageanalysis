#!/usr/bin/env python3
"""Check the rendered site (docs/) for broken local links, anchors and resources.

Every href/src in every .html page under docs/ that points to a local file must exist; an
href with a #fragment into an .html page must name an id in that page (reveal.js slide
fragments "#/..." are skipped). Links that were left as .qmd (an unresolved link target at
render time) count as broken. External URLs (http, https, mailto, //) are not checked.

Files that CI adds at deploy time (demos/dist/wasm/, demos/dist/models/, git-ignored) are
loaded by the demos at run time, not linked from the HTML, so they do not appear here.

Usage: python3 tools/check/links.py [docs-dir]     (make check runs it; exit 1 on problems)
"""
import html
import os
import re
import sys
from urllib.parse import unquote

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'docs'
ATTR = re.compile(r'\b(href|src)="([^"]*)"')
IDS = re.compile(r'\bid="([^"]+)"')
SKIP = ('http:', 'https:', 'mailto:', '//', 'data:', 'javascript:', 'tel:', 'blob:', 'about:')


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f'{ROOT}/ does not exist: render the site first (make render)')
    pages = sorted(os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs if f.endswith('.html'))
    text, ids = {}, {}
    for p in pages:
        with open(p, encoding='utf-8', errors='replace') as f:
            text[p] = f.read()
        ids[p] = set(html.unescape(i) for i in IDS.findall(text[p]))
    bad = []
    n = 0
    for p in pages:
        s = text[p]
        # Skip code inside <script> blocks (generated strings, templates) but keep <script src=...>.
        s = re.sub(r'(<script\b[^>]*>).*?(</script>)', r'\1\2', s, flags=re.S)
        for attr, raw in ATTR.findall(s):
            v = html.unescape(raw).strip()
            if not v or v.startswith(SKIP) or v.startswith('{') or '${' in v:
                continue
            n += 1
            url, _, frag = v.partition('#')
            url = unquote(url.split('?', 1)[0])
            if url:
                tgt = os.path.normpath(os.path.join(os.path.dirname(p), url))
                if os.path.isdir(tgt):
                    tgt = os.path.join(tgt, 'index.html')
                if not os.path.exists(tgt):
                    bad.append((p, v, 'missing file'))
                    continue
            else:
                tgt = p
            if attr == 'href' and frag and not frag.startswith('/') and tgt.endswith('.html'):
                if unquote(frag) not in ids.get(tgt, ()):
                    bad.append((p, v, 'missing anchor'))
    for p, v, why in bad:
        print(f'{os.path.relpath(p, ROOT)}: {v} ({why})')
    print(f'links: {len(pages)} pages, {n} local links/resources checked, {len(bad)} broken')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
