#!/bin/bash
# Screenshots of the real Cellpose demo (cellposenet) for the cellpose/ slide figures.
# Needs `make wasm` and `make models`. usage: tools/figs/cellpose/shots.sh <outdir>
# Then: python3 -I tools/figs/cellpose/lec_figs.py <outdir>
set -e
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
RD="document.querySelector('[data-demo=cellposenet] .qb-readout').textContent"
DONE="$RD.includes('Decoding:') && document.querySelector('.cp-busy').classList.contains('cp-hide')"
tab() { echo "{\"js\":\"[...document.querySelectorAll('.cp-tabs button')].find(b=>b.textContent==='$1').click()\"},{\"sleep\":${2:-1}}"; }
shot() { echo "{\"shot\":\"$OUT/$1\",\"selector\":\".cp-stage\",\"pad\":0}"; }
btn() { echo "{\"js\":\"[...document.querySelectorAll('[data-demo=cellposenet] button')].find(b=>b.textContent==='$1').click()\"},{\"sleep\":1.5},{\"wait\":\"$DONE\",\"timeout\":300}"; }
run() { # <query> <steps...>
  local q=$1; shift
  local s="[{\"wait\":\"$DONE\",\"timeout\":600},{\"print\":\"$RD\"}"
  for x in "$@"; do s="$s,$x"; done
  echo "$s]" > "$OUT/steps.json"
  "$HERE/demo_shot.sh" "$OUT" "$q" "$OUT/steps.json" --size 1220x900 --scale 1
}
run "d=cellposenet&sample=cells&view=input&autorun=1" "$(shot cells-input.png)" "$(tab Flows)" "$(shot cells-flows.png)" \
  "$(tab Tracking 4)" "$(shot cells-track.png)" "$(tab 'Flow check')" "$(shot cells-check.png)" "$(tab Masks)" "$(shot cells-masks.png)" \
  "$(btn 'Only 10 steps')" "{\"print\":\"$RD\"}" "$(tab 'Flow check')" "$(shot cells-10-check.png)" "$(tab Masks)" "$(shot cells-10-masks.png)"
for d in 8 15 29 58 100; do
  run "d=cellposenet&sample=nuclei&view=masks&autorun=1&diameter=$d" "$(shot nuclei-d$d-masks.png)"
done
run "d=cellposenet&sample=tissue&view=masks&autorun=1" "$(shot tissue-inv-masks.png)" \
  "{\"js\":\"[...document.querySelectorAll('[data-demo=cellposenet] label')].find(l=>l.textContent.includes('invert')).querySelector('input').click()\"},{\"sleep\":2},{\"wait\":\"$DONE\",\"timeout\":300}" \
  "{\"print\":\"$RD\"}" "$(shot tissue-noinv-masks.png)"
