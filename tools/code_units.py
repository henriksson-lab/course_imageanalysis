#!/usr/bin/env python3
"""Write the list of units with coding exercises into code.qmd (between the markers
BEGIN/END GENERATED CODE UNITS), from days.yml and the crates in code/<unit>/.
Also checks that every unit crate is a member of the code/ workspace.
Run by `make code-zip`.
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, 'code.qmd')
BEGIN = '<!-- BEGIN GENERATED CODE UNITS (tools/code_units.py) -->'
END = '<!-- END GENERATED CODE UNITS -->'


def main():
    d = yaml.safe_load(open(os.path.join(ROOT, 'days.yml')))
    toml = open(os.path.join(ROOT, 'code', 'Cargo.toml')).read()
    members = set(re.findall(r'"([^"]+)"', re.search(r'members\s*=\s*\[(.*?)\]', toml, re.S).group(1)))
    lines, n, missing = [], 0, []
    for part in d['parts']:
        for day in part['days']:
            folder = day['id']
            units = [u for u in day['units']
                     if os.path.isfile(os.path.join(ROOT, 'code', folder, 'src', u['folder'].replace('-', '_'), 'mod.rs'))]
            if not units:
                continue
            links = ', '.join(f"[{u['title']}]({u['folder']}/exercise.qmd#code)" for u in units)
            lines.append(f"- **Day {day['number']}** (`{folder}`): {day['title']}. {links}.")
            n += len(units)
            if folder not in members:
                missing.append(folder)
    if missing:
        sys.exit('not in the code/ workspace (the members list in code/Cargo.toml): ' + ', '.join(missing))
    text = open(PAGE).read()
    a, b = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:a] + '\n' + '\n'.join(lines) + '\n' + text[b:]
    if new != text:
        open(PAGE, 'w').write(new)
    print(f'code.qmd: {n} units with coding exercises')


if __name__ == '__main__':
    main()
