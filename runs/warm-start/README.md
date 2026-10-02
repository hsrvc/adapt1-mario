# Warm start: 1-1 cleared on 5/5 seeds (2026-10-01)

Two Adapt-1 domains, fed recorded rows and then evaluated frozen:

| domain | decides | features | rows fed |
|---|---|---|---|
| takeoff | which of 8 macros to play when Mario is on the ground | `full_v3` (23 numbers) | 640 |
| apex | at the top of each jump: keep going, brake, or pull back | `apex_v3` (18 numbers) | 377 |

Both are rewarded the same way: Mario's progress per frame over the next 128 frames (`reward_rate`). Both use the
`extra_trees` model, pinned with `learning.training.model_type`. Frozen evaluation (exploration off,
`selection_mode: exploit`): **3161, the flag, on seeds 777, 1, 2, 3 and 4**, 62–63 decisions each, every one picked by
the server. On this takeoff policy's jumps the apex domain always answers "keep going", so the takeoff domain carries
the clear.

## Files

| file | what |
|---|---|
| `w28count-v3-takeoff.jsonl.gz`, `w28count-v3-apex.jsonl.gz` | the exact rows fed (gunzip before ingesting) |
| `mario-gate4-takeoff-*.json`, `mario-gate4-apex-*.json` | ingest records: the domain config sent, rows fed, reward field; with provenance |
| `1-1-seed*.trace.jsonl`, `1-1-seed*.gif`, `eval.log` | the five frozen evaluations, decision by decision |
| `index.json` | sha256 of every file (`scripts/pub/verify_runs.py`) |

## Rebuilding the rows from code (offline, 0 Records)

Both pools below regenerate byte for byte from this repo (checked 2026-10-02):

```bash
MACROS=noop,right,right_run,left,right_run_jump_short,right_run_jump_mid,right_run_jump_full,right_jump_full
D=artifacts/demos
# demonstrations: the scripted teacher with 10 % random moves, then returns, then curation
python scripts/record_demos.py --cadence grounded --reward progress --epsilon 0.10 --episodes 30 --max-decisions 400 \
  --stall-timeout 40 --apex --features full_v3 --macros $MACROS --out $D/demos.jsonl
python scripts/kstep_returns.py --dataset $D/demos.jsonl --out $D/demos-r.jsonl --k 6 --gamma 0.9 --frames-horizon 128
python scripts/curate_demos.py --dataset $D/demos-r.jsonl --manifest $D/demos.manifest.jsonl --out $D/curated.jsonl \
  --cap 450 --max-flag-episodes 2 --min-episodes 1 --max-episode-transitions 90
# coverage: from states along the teacher's path, try every macro and every apex choice
python scripts/historical/apex_coverage_7631050_v2.py --out $D/coverage.jsonl
python scripts/kstep_returns.py --dataset $D/coverage.jsonl --out $D/coverage-r.jsonl --k 6 --gamma 0.9 --frames-horizon 128
# the apex feed: identical to w28count-v3-apex.jsonl (sha256 18b116aa…)
python scripts/assemble_apex_feed.py --demos $D/curated.jsonl --coverage $D/coverage-r.jsonl --out $D/apex-feed.jsonl
```

**Takeoff feed.** Every one of its 640 rows is an exact row of these pools: 341 curated demonstration rows, 121
coverage rows, 55 demonstration rows with a random move, and 123 rows for rarely used macros from a second coverage
pool (`scripts/historical/apex_coverage_d6faa03_v2.py --follow best --write-mainline --stride 3 --macros $MACROS`,
then the same returns). Each segment rotates through the macros. The exact choice of coverage rows was not recorded,
so this file has no assembly script: use the shipped file.

## Caveat

The server fits its own model. The same rows on a different server build can fit differently and play a different
route, so rerun it and compare, rather than expecting a bit-exact match.
