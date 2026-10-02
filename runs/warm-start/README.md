# Warm start: 1-1 cleared on 5/5 seeds (2026-10-01)

Two Adapt-1 domains, fed recorded rows and then evaluated frozen:

| domain | decides | features | rows fed | reward |
|---|---|---|---|---|
| takeoff | which of 8 macros to play when Mario is on the ground | `full_v3` (23 numbers) | 640 | progress per frame over the next 128 frames (`reward_rate`) |
| apex | at the top of each jump: keep going, brake, or pull back | `apex_v3` (18 numbers) | 788 | 6-step discounted return (`reward`) |

Both use the `extra_trees` model, pinned with `learning.training.model_type`. Frozen evaluation (exploration off,
`selection_mode: exploit`): **3161, the flag, on seeds 777, 1, 2, 3 and 4**. Each run takes 62 decisions and 20 apex
looks, and the server picked every one (`selected` 310/310). The traces are identical across seeds: the level is
deterministic, and so is the server given its evidence.

## Files

| file | what |
|---|---|
| `w28count-v3-takeoff.jsonl.gz`, `w28v3-apex-k6.jsonl.gz` | the exact rows fed (gunzip before ingesting) |
| `mario-gate*-takeoff-*.json`, `mario-gate*-apex-*.json` | ingest records: the domain config sent, the rows fed, the reward field; with provenance |
| `model-state-at-eval.json` | each learner's state at evaluation time (model type, version, validation skill) |
| `1-1-seed*.trace.jsonl`, `1-1-seed*.gif`, `eval.log` | the five frozen evaluations, decision by decision |
| `index.json` | sha256 of every file (`scripts/pub/verify_runs.py`) |

## Where the rows came from

All offline, 0 Records:

- **Takeoff (640 rows)**: 341 curated teacher demonstrations, 55 demonstrations with 10 % random exploration, 121
  coverage rows (every macro tried from states along the teacher's path), 123 rows for rarely used macros. Generated
  with `record_demos.py --cadence grounded --reward progress --epsilon 0.10 --episodes 30 --max-decisions 400
  --stall-timeout 40 --apex --features full_v3`, the coverage generators in `scripts/historical/`, then
  `kstep_returns.py --k 6 --gamma 0.9 --frames-horizon 128` and `curate_demos.py --cap 450 --max-flag-episodes 2
  --min-episodes 1 --max-episode-transitions 90`, taking the first N rows per macro.
- **Apex (788 rows)**: 107 curated, 405 coverage, and 276 rows branched (`branch_from_trace.py`) from the trace of an
  earlier evaluation that is not shipped here.

The fed files are the reference: their sha256 is in `index.json`, so a regenerated file can be checked against them.

## Caveat

The server fits its own model. The same rows on a different server build can fit differently and play a different
route, so rerun it and compare, rather than expecting a bit-exact match.
