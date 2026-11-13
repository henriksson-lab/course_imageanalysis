#!/usr/bin/env python3
"""Generate the data of the course overview page (overview.qmd) from days.yml.

    python3 tools/days_json.py           # write both outputs
    python3 tools/days_json.py --check   # exit 1 if either output is out of date

Called by `make sidebar` (tools/days.py sidebar) after the sidebar is written, and checked by
`make check-prereqs` (tools/days.py check). Two outputs, both committed:

- demos/src/coursegraph/days.json: parts, days and units with their prerequisites and topics,
  bundled into the interactive dependency graph (demos/build.sh coursegraph);
- the plain list view in overview.qmd, between the GENERATED LIST markers: parts -> days ->
  units with their prerequisites as links (for print and screen readers; it needs no script).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import days as daysdb  # noqa: E402  (tools/days.py)

ROOT = daysdb.ROOT
JSON_OUT = os.path.join(ROOT, 'demos', 'src', 'coursegraph', 'days.json')
OVERVIEW = os.path.join(ROOT, 'overview.qmd')
LIST_BEGIN = '<!-- BEGIN GENERATED LIST (tools/days_json.py from days.yml; edit days.yml, then make sidebar) -->'
LIST_END = '<!-- END GENERATED LIST -->'


def refs(x):
    return [str(r) for r in (x or [])]


def json_text():
    data, days, units = daysdb.load()
    out = {'parts': [], 'days': []}
    for k, part in enumerate(data['parts']):
        out['parts'].append({'id': part['id'], 'title': part['title'],
                             'colour': part['colour'], 'tint': part['tint']})
        for d in part['days']:
            out['days'].append({
                'id': d['id'], 'number': d['number'], 'title': d['title'], 'part': k,
                'planned': d.get('status') == 'planned',
                'hard': refs(d.get('hard')), 'soft': refs(d.get('soft')),
                'owns': [str(o) for o in d.get('owns') or []],
                'units': [{'folder': u['folder'], 'title': u['title'],
                           'planned': daysdb.planned(u),
                           'hard': refs(u.get('hard')), 'soft': refs(u.get('soft'))}
                          for u in d['units']],
            })
    return json.dumps(out, ensure_ascii=False, indent=1) + '\n'


def md_escape(s):
    return str(s).replace('[', '\\[').replace(']', '\\]').replace('*', '\\*')


def list_text():
    data, days, units = daysdb.load()
    unit_by = {u['folder']: u for u in units}
    day_by = {d['id']: d for d in days}

    def link(ref):
        ref = str(ref)
        if ref.endswith('/'):
            u = unit_by[ref[:-1]]
            if daysdb.planned(u):
                return md_escape(u['title']) + ' (in preparation)'
            return f"[{md_escape(u['title'])}]({u['folder']}/lec.qmd) (day {u['day']['number']})"
        d = day_by[ref]
        return f"[Day {d['number']}: {md_escape(d['title'])}](#day-{d['id']})"

    lines = []
    for part in data['parts']:
        lines += [f"### Part {part['id']}: {part['title']} {{#part-{part['id']}}}", '']
        for d in part['days']:
            tag = ' (in preparation)' if d.get('status') == 'planned' else ''
            lines += [f"#### Day {d['number']}: {md_escape(d['title'])}{tag} {{#day-{d['id']}}}", '']
            hard = refs(d.get('hard'))
            soft = refs(d.get('soft'))
            lines.append('**Needs:** ' + (', '.join(link(r) for r in hard) if hard else 'nothing earlier') + '.')
            if soft:
                lines += ['', '**See also:** ' + ', '.join(link(r) for r in soft) + '.']
            if d.get('owns'):
                lines += ['', '**Topics:** ' + '; '.join(md_escape(o) for o in d['owns']) + '.']
            lines.append('')
            for k, u in enumerate(d['units'], 1):
                f = u['folder']
                if daysdb.planned(u):
                    item = f"{k}. {md_escape(u['title'])} (in preparation)"
                else:
                    item = (f"{k}. [{md_escape(u['title'])}]({f}/lec.qmd) "
                            f"([quiz]({f}/quiz.qmd), [exercise]({f}/exercise.qmd))")
                extra = []
                if u.get('hard'):
                    extra.append('also needs ' + ', '.join(link(r) for r in refs(u['hard'])))
                if u.get('soft'):
                    extra.append('see also ' + ', '.join(link(r) for r in refs(u['soft'])))
                if extra:
                    item += '; ' + '; '.join(extra)
                lines.append(item)
            lines.append('')
    return '\n'.join(lines).rstrip('\n') + '\n'


def overview_with_list(text):
    if LIST_BEGIN not in text or LIST_END not in text:
        raise SystemExit(f'overview.qmd: markers missing ({LIST_BEGIN} ... {LIST_END})')
    pre, rest = text.split(LIST_BEGIN, 1)
    post = rest.split(LIST_END, 1)[1]
    return pre + LIST_BEGIN + '\n\n' + list_text() + '\n' + LIST_END + post


def read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None


def stale():
    """Problems as strings (empty when both outputs match days.yml)."""
    errs = []
    if read(JSON_OUT) != json_text():
        errs.append('demos/src/coursegraph/days.json differs from days.yml; run make sidebar')
    ov = read(OVERVIEW)
    if ov is not None and LIST_BEGIN in ov and ov != overview_with_list(ov):
        errs.append('overview.qmd: the generated list differs from days.yml; run make sidebar')
    return errs


def write():
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        f.write(json_text())
    msg = 'days.json written'
    ov = read(OVERVIEW)
    if ov is not None:
        new = overview_with_list(ov)
        if new != ov:
            with open(OVERVIEW, 'w', encoding='utf-8') as f:
                f.write(new)
        msg += '; list in overview.qmd written'
    print(msg + ' (rebuild the graph with demos/build.sh coursegraph)')


if __name__ == '__main__':
    if '--check' in sys.argv[1:]:
        e = stale()
        print('\n'.join(e) if e else 'overview data current')
        sys.exit(1 if e else 0)
    write()
