#!/usr/bin/env python3
"""The course structure database (days.yml): check it and generate the sidebar from it.

    python3 tools/days.py check     # validate days.yml   (make check-prereqs)
    python3 tools/days.py sidebar   # rewrite the sidebar in _quarto.yml and the
                                    # part colours in styles.css   (make sidebar)
                                    # and the overview page's data (tools/days_json.py)

days.yml: parts -> days -> units. A prerequisite is a day id, or a unit folder written with a
trailing slash ("pca/"). Hard prerequisites must come earlier in the course order; soft ones
are back-references and may point anywhere. Units and days with `status: planned` are not
built yet: their folders may be missing and they are left out of the sidebar.
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYS_YML = os.path.join(ROOT, 'days.yml')
QUARTO_YML = os.path.join(ROOT, '_quarto.yml')
STYLES = os.path.join(ROOT, 'styles.css')

SIDEBAR_BEGIN = '      # BEGIN GENERATED SIDEBAR (tools/days.py from days.yml; edit days.yml, then make sidebar)'
SIDEBAR_END = '      # END GENERATED SIDEBAR'
CSS_BEGIN = '/* BEGIN GENERATED PART COLOURS (tools/days.py from days.yml; make sidebar) */'
CSS_END = '/* END GENERATED PART COLOURS */'


def load():
    with open(DAYS_YML, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    days, units = [], []
    for part in data['parts']:
        for day in part['days']:
            day['part'] = part
            days.append(day)
            for k, unit in enumerate(day['units']):
                unit['day'] = day
                unit['index'] = k
                units.append(unit)
    return data, days, units


def planned(x):
    return x.get('status') == 'planned' or ('day' in x and x['day'].get('status') == 'planned')


def check():
    data, days, units = load()
    errors = []
    err = errors.append
    day_by_id, unit_by_folder = {}, {}
    for d in days:
        if d['id'] in day_by_id:
            err(f"day id {d['id']} used twice")
        day_by_id[d['id']] = d
    for u in units:
        if u['folder'] in unit_by_folder:
            err(f"unit {u['folder']} listed twice")
        unit_by_folder[u['folder']] = u

    # Global numbers 1..N in order.
    for k, d in enumerate(days, 1):
        if d.get('number') != k:
            err(f"day {d['id']}: number {d.get('number')}, expected {k} (numbers run 1..N in order)")

    # Folders exist with the three pages (unless planned).
    for u in units:
        if planned(u):
            continue
        for page in ('lec.qmd', 'quiz.qmd', 'exercise.qmd'):
            if not os.path.isfile(os.path.join(ROOT, u['folder'], page)):
                err(f"unit {u['folder']}: {u['folder']}/{page} missing")

    # Every unit folder on disk is in the database.
    for name in sorted(os.listdir(ROOT)):
        if os.path.isfile(os.path.join(ROOT, name, 'lec.qmd')) and name not in unit_by_folder:
            err(f"folder {name}/ has a lec.qmd but is not in days.yml")

    # Resolve prerequisites: node = ('day', id) or ('unit', folder).
    def resolve(ref, where):
        ref = str(ref)
        if ref.endswith('/'):
            if ref[:-1] not in unit_by_folder:
                err(f"{where}: unknown unit {ref}")
                return None
            return ('unit', ref[:-1])
        if ref not in day_by_id:
            err(f"{where}: unknown day {ref}")
            return None
        return ('day', ref)

    # Position of each node in the course order: a day at its first unit, a unit at itself.
    pos = {}
    for k, u in enumerate(units):
        pos[('unit', u['folder'])] = k
        pos.setdefault(('day', u['day']['id']), k)
    end_of_day = {}
    for k, u in enumerate(units):
        end_of_day[u['day']['id']] = k

    edges = {}  # node -> list of hard prerequisite nodes (for the cycle check)
    for d in days:
        node = ('day', d['id'])
        edges[node] = []
        for kind in ('hard', 'soft'):
            for ref in d.get(kind) or []:
                r = resolve(ref, f"day {d['id']} {kind}")
                if r is None:
                    continue
                if r == node or (r[0] == 'unit' and unit_by_folder[r[1]]['day'] is d):
                    err(f"day {d['id']}: {kind} prerequisite {ref} is the day itself")
                    continue
                if kind == 'hard':
                    edges[node].append(r)
                    # the whole prerequisite day must be finished before this day starts
                    last = end_of_day[r[1]] if r[0] == 'day' else pos[r]
                    if last >= pos[node]:
                        err(f"order: day {d['number']} {d['id']} needs {ref}, which comes later")
                    if planned(r[0] == 'day' and day_by_id[r[1]] or unit_by_folder.get(r[1], {})) and not planned(d):
                        err(f"day {d['id']}: hard prerequisite {ref} is only planned")
    for u in units:
        node = ('unit', u['folder'])
        edges[node] = [('day', u['day']['id'])]
        for kind in ('hard', 'soft'):
            for ref in u.get(kind) or []:
                r = resolve(ref, f"unit {u['folder']} {kind}")
                if r is None:
                    continue
                if r == node or r == ('day', u['day']['id']):
                    err(f"unit {u['folder']}: {kind} prerequisite {ref} is the unit or its own day")
                    continue
                if kind == 'hard':
                    edges[node].append(r)
                    last = end_of_day[r[1]] if r[0] == 'day' else pos[r]
                    if last >= pos[node]:
                        err(f"order: unit {u['folder']} (day {u['day']['number']}) needs {ref}, which comes later")

    # Cycle check over hard edges (a day depends on its prerequisites; a unit on its day too).
    state = {}

    def visit(n, stack):
        state[n] = 1
        for m in edges.get(n, []):
            if state.get(m) == 1:
                err('cycle: ' + ' -> '.join(f'{a}:{b}' for a, b in stack + [n, m]))
            elif m not in state:
                visit(m, stack + [n])
        state[n] = 2

    for n in list(edges):
        if n not in state:
            visit(n, [])

    # The sidebar in _quarto.yml is the generated one.
    with open(QUARTO_YML, encoding='utf-8') as f:
        q = f.read()
    if SIDEBAR_BEGIN in q:
        current = q.split(SIDEBAR_BEGIN, 1)[1].split(SIDEBAR_END, 1)[0]
        if current.strip('\n') != (head_block() + sidebar_text(days)).strip('\n'):
            err('_quarto.yml: the sidebar differs from days.yml; run make sidebar')
    else:
        err('_quarto.yml: no generated sidebar block; run make sidebar')

    # The overview page's data (days.json and the list in overview.qmd) is current.
    import days_json
    errors += days_json.stale()

    if errors:
        print('\n'.join(errors))
        print(f'{len(errors)} problem(s) in days.yml')
        return 1
    n_planned = sum(1 for u in units if planned(u))
    print(f"days.yml OK: {len(data['parts'])} parts, {len(days)} days, {len(units)} units "
          f'({n_planned} planned); prerequisites exist, no cycles, order respects hard prerequisites')
    return 0


EXPLORE = re.compile(r'^(#{2,6})\s+(.*?)\s*\{#(explore[\w-]*)[^}]*\}\s*$', re.M)


def demo_entries(unit):
    """(anchor, label) for every demo anchor on the exercise page, in page order.

    A heading with an explore anchor counts when a demo (data-demo=...) is embedded before the
    next explore heading; a bare "## Explore {#explore}" that only groups sub-sections is skipped.
    """
    path = os.path.join(ROOT, unit['folder'], 'exercise.qmd')
    with open(path, encoding='utf-8') as f:
        text = f.read()
    heads = list(EXPLORE.finditer(text))
    labels = unit.get('demos') or {}
    out = []
    for k, m in enumerate(heads):
        stop = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        if 'data-demo=' not in text[m.end():stop]:
            continue
        anchor = m.group(3)
        if anchor in labels:
            label = labels[anchor]
        else:
            rest = re.sub(r'^Explore\s*:?\s*', '', m.group(2)).strip()
            label = 'Demo: ' + rest if rest else 'Demo'
        out.append((anchor, label))
    return out


def yq(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def sidebar_text(days):
    lines = []
    ind = ' ' * 6
    parts = []
    for d in days:
        if not parts or parts[-1][0] is not d['part']:
            parts.append((d['part'], []))
        parts[-1][1].append(d)
    for part, pdays in parts:
        lines.append(f"{ind}- section: {yq('Part ' + part['id'] + ': ' + part['title'])}")
        lines.append(f'{ind}  contents:')
        for d in pdays:
            built = [u for u in d['units'] if not planned(u)]
            if not built:
                continue
            lines.append(f"{ind}    - section: {yq('Day ' + str(d['number']) + ': ' + d['title'])}")
            lines.append(f'{ind}      contents:')
            for u in built:
                f = u['folder']
                lines.append(f"{ind}        - section: {yq(u['title'])}")
                lines.append(f'{ind}          contents:')
                for text, href in [('Lecture (slides)', f'{f}/lec.qmd'), ('Quiz', f'{f}/quiz.qmd'),
                                   ('Exercise', f'{f}/exercise.qmd')]:
                    lines.append(f'{ind}            - text: {yq(text)}')
                    lines.append(f'{ind}              href: {href}')
                for anchor, label in demo_entries(u):
                    lines.append(f'{ind}            - text: {yq(label)}')
                    lines.append(f'{ind}              href: {f}/exercise.qmd#{anchor}')
    return '\n'.join(lines) + '\n'


def head_items():
    """Top-level sidebar items before the parts."""
    items = [('Home', 'index.qmd'), ('All interactive demos', 'demos.qmd')]
    if os.path.isfile(os.path.join(ROOT, 'overview.qmd')):
        items.append(('Overview: prerequisites and learning paths', 'overview.qmd'))
    return items


def topic_items():
    """The topic pages (topics/*.qmd): the index first, then the others by title."""
    d = os.path.join(ROOT, 'topics')
    if not os.path.isfile(os.path.join(d, 'index.qmd')):
        return []
    items = []
    for name in sorted(os.listdir(d)):
        if not name.endswith('.qmd') or name == 'index.qmd':
            continue
        with open(os.path.join(d, name), encoding='utf-8') as f:
            m = re.search(r'^title:\s*"?(.*?)"?\s*$', f.read(), re.M)
        title = m.group(1) if m else name[:-4]
        title = re.sub(r'^Topic:\s*', '', title)
        items.append((title[:1].upper() + title[1:], 'topics/' + name))
    items.sort(key=lambda t: t[0].lower())
    return [('All topics', 'topics/index.qmd')] + items


def head_block():
    out = ''.join(f'      - text: {yq(t)}\n        href: {h}\n' for t, h in head_items())
    topics = topic_items()
    if topics:
        out += '      - section: "Topics"\n        contents:\n'
        out += ''.join(f'          - text: {yq(t)}\n            href: {h}\n' for t, h in topics)
    return out


def n_head():
    """Number of top-level sidebar entries before the parts."""
    return len(head_items()) + (1 if topic_items() else 0)


def css_text(data):
    first = n_head() + 1
    n = len(data['parts'])
    names = [t for t, _ in head_items()] + (['Topics'] if topic_items() else [])
    lines = [CSS_BEGIN,
             '/* One colour per part: top-level sidebar items are ' +
             ', '.join(f'{t} ({k})' for k, t in enumerate(names, 1)) +
             f', then the parts ({first}–{first + n - 1}). */']
    sel = f'.sidebar-menu-container > ul > li.sidebar-item-section:nth-child(n+{first})'
    lines += [f'{sel} {{', '  border-left: 4px solid var(--week-color);', '  padding-left: 0.35rem;',
              '  margin: 0.4rem 0;', '}',
              f'{sel} > .sidebar-item-container {{', '  background: var(--week-tint);',
              '  border-radius: 0 4px 4px 0;', '  padding-left: 0.3rem;', '}',
              f'{sel} > .sidebar-item-container .menu-text {{', '  color: var(--week-color);', '}']
    for k, part in enumerate(data['parts']):
        lines.append(f'.sidebar-menu-container > ul > li:nth-child({first + k}) '
                     f"{{ --week-color: {part['colour']}; --week-tint: {part['tint']}; }} "
                     f"/* Part {part['id']}: {part['title']} */")
    lines.append(CSS_END)
    return '\n'.join(lines) + '\n'


def write_sidebar():
    data, days, units = load()
    with open(QUARTO_YML, encoding='utf-8') as f:
        q = f.read()
    block = SIDEBAR_BEGIN + '\n' + head_block() + sidebar_text(days) + SIDEBAR_END + '\n'
    if SIDEBAR_BEGIN in q:
        pre, rest = q.split(SIDEBAR_BEGIN, 1)
        post = rest.split(SIDEBAR_END + '\n', 1)[1]
    else:
        # First run: replace everything under "sidebar: ... contents:" up to the next top-level key.
        m = re.search(r'(\n    contents:\n)(.*?)(\n\S)', q, re.S)
        if not m:
            sys.exit('could not find the sidebar contents in _quarto.yml')
        pre, post = q[:m.end(1)], q[m.start(3) + 1:]
        post = '\n' + post
    with open(QUARTO_YML, 'w', encoding='utf-8') as f:
        f.write(pre + block + post)

    with open(STYLES, encoding='utf-8') as f:
        css = f.read()
    if CSS_BEGIN in css:
        pre, rest = css.split(CSS_BEGIN, 1)
        post = rest.split(CSS_END + '\n', 1)[1]
    else:
        k = css.index('/* One colour per part in the sidebar')
        pre, post = css[:k], ''
    with open(STYLES, 'w', encoding='utf-8') as f:
        f.write(pre + css_text(data) + post)
    built = sum(1 for u in units if not planned(u))
    print(f"sidebar written to _quarto.yml: {len(data['parts'])} parts, "
          f"{sum(1 for d in days if any(not planned(u) for u in d['units']))} days, {built} units; "
          'part colours written to styles.css')
    import days_json   # the overview page's data: days.json and the list in overview.qmd
    days_json.write()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if cmd == 'check':
        sys.exit(check())
    if cmd == 'sidebar':
        write_sidebar()
        sys.exit(0)
    sys.exit(__doc__)
