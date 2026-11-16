#!/usr/bin/env python3
"""Add unit crates to the members of the code/ workspace (code/Cargo.toml), safely when
several people or agents do it at the same time (file lock). Members are kept in course
order (days.yml), after imgcourse and starter.   Usage: tools/code_add_member.py <unit> ...
"""
import fcntl
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOML = os.path.join(ROOT, 'code', 'Cargo.toml')


def course_order():
    d = yaml.safe_load(open(os.path.join(ROOT, 'days.yml')))
    return [u['folder'] for p in d['parts'] for day in p['days'] for u in day['units']]


def main(names):
    for n in names:
        if not os.path.isfile(os.path.join(ROOT, 'code', n, 'Cargo.toml')):
            sys.exit(f'code/{n}/Cargo.toml does not exist: create the crate first')
    order = course_order()
    with open(TOML, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        text = f.read()
        m = re.search(r'members\s*=\s*\[(.*?)\]', text, re.S)
        current = re.findall(r'"([^"]+)"', m.group(1))
        units = [u for u in current if u not in ('imgcourse', 'starter')]
        units += [n for n in names if n not in units]
        units.sort(key=lambda u: order.index(u) if u in order else len(order))
        body = '\n'.join(f'    "{u}",' for u in ['imgcourse', 'starter'] + units)
        text = text[:m.start()] + f'members = [\n{body}\n]' + text[m.end():]
        f.seek(0)
        f.truncate()
        f.write(text)
    print('members:', ', '.join(units))


if __name__ == '__main__':
    main(sys.argv[1:])
