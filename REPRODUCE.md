# Reproduce

Run everything from the repository root. The steps go from free to paid.

## 1. Install and test: free

Python 3.13 or newer. The emulator and the game come with the `gym-super-mario-bros` package; using them is subject
to your local law.

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock.txt          # the exact versions our runs used
pip install -e ".[mario,dev]"
pytest -q                                     # offline: never touches the API
python scripts/pub/verify_runs.py runs        # every evidence file against its sha256 index
```

## 2. Before any live run

- Get an Adapt-1 API key: sign up at [app.reilabs.org/adapt-1](https://app.reilabs.org/adapt-1), where your keys and your
  Records/Queries balance live. Adapt-1's docs: [docs.reilabs.org](https://docs.reilabs.org)
  ([quickstart](https://docs.reilabs.org/docs/neuroadapt/quickstart), [full index](https://docs.reilabs.org/llms.txt)).
- Put the key in the environment only: `export REI_KEY=...`. Rei's own examples call it `ADAPT1_API_KEY`; these
  scripts read `REI_KEY`.
- Adapt-1 bills **Records** (every row, result or feedback you send) and **Queries** (every decision you ask for).
  Check your balance at [app.reilabs.org/adapt-1](https://app.reilabs.org/adapt-1) first; the API doesn't report it.
- Use a **new domain name** for every run. Reusing one mixes your rows with old ones, and `--recreate` deletes any
  existing domain with that name first.
- Run each script that spends Records (`machina_acquire.py`, `ingest_demos.py`, `zero_start_duo.py`) with `--dry-run`
  first. It runs the same code against an offline stub and spends nothing. The evaluation scripts spend Queries only.

## 3. Machina: ≈ 470 Records, 15 minutes

```bash
python scripts/machina_acquire.py --domain-id my-machina-1 --attempts 1024 --horizon 256 --capacity 96 \
  --profile coherent_edits,global_recall,progress_guided_mutation,sequence_compaction,outcome_tier_exploration,retain_prefix_failures=false
python scripts/machina_frozen.py --domain-id my-machina-1                         # Queries only
```

**Expect:** the first flag at attempt **403**, a stop 64 attempts later (467 Records), then a frozen 1-1 of 3161 on
all 3 repeats. Machina is deterministic, so your run should match ours attempt for attempt:
`runs/machina/logs/*-attempts.jsonl.gz` has `max_x`, `flag` and `source` for each attempt. If yours diverges, compare
the server's resolved config (written near the top of your journal in `artifacts/machina/`) with ours.

## 4. Warm start: 1,017 Records

Also ≈ 1,000 Queries for the ingest (one per row) and ≈ 400 for the evaluation.

The two domains are fed the exact rows from our run, which ship in `runs/warm-start/`:

```bash
gunzip -k runs/warm-start/*.jsonl.gz
MACROS=noop,right,right_run,left,right_run_jump_short,right_run_jump_mid,right_run_jump_full,right_jump_full

python scripts/ingest_demos.py --dataset runs/warm-start/w28count-v3-apex.jsonl --domain-id my-apex-1 \
  --kind apex --features apex_v3 --cadence grounded --bandit \
  --reward-field reward_rate --reward-max 1.0 --model-type extra_trees --recreate     # 377 Records
python scripts/ingest_demos.py --dataset runs/warm-start/w28count-v3-takeoff.jsonl --domain-id my-takeoff-1 \
  --kind takeoff --takeoff-under teacher --features full_v3 --cadence grounded --bandit --macros $MACROS \
  --reward-field reward_rate --reward-max 1.0 --model-type extra_trees --recreate     # 640 Records

python scripts/frozen_eval.py --domain-id my-takeoff-1 --features full_v3 \
  --apex-domain my-apex-1 --apex-features apex_v3 --cadence grounded --macros $MACROS \
  --seeds 777,1,2,3,4 --wait-installed 900                                            # Queries only
```

`--wait-installed` matters. For a few minutes after rows arrive, the server retrains and answers from a fallback
without saying so. An evaluation in that window measures the fallback, not the policy.

**Expect:** 3161 on all five seeds. The server's fit can vary between builds, so a near miss is possible on a
different server version. How the two datasets are rebuilt from code is in
[`runs/warm-start/README.md`](runs/warm-start/README.md).

## 5. Zero start: ≈ 8,500 Records, ≈ 16 hours

Two empty domains play 1-1 and learn online. We ran it in stages of 2,000 Records each, which makes natural
checkpoints:

```bash
python scripts/zero_start_duo.py --takeoff-domain my-takeoff-z --apex-domain my-apex-z --recreate \
  --max-samples 16384 --features full_v3 --apex-features apex_v3 --credit frame_horizon --horizon 128 \
  --model-type extra_trees --exploration-mode ucb --episodes 1000 --seed 0 --max-records 2000 \
  --eval-every 25 --eval-seeds 777 --stop-flat 100 --out-dir artifacts/zero-s1 --run-id zero-s1
```

For each later stage, drop `--recreate`, set `--seed` to the number of episodes played so far (ours: 45, 72, 98, 122),
and use a new `--out-dir` and `--run-id`. Each stage's settings are in `runs/zero-start/stage-N/*.provenance.json`.

**Two settings decide whether this works:**
- `--max-samples` is the learner's row budget. Keep it above the Records you will ever send. Past the budget the
  learner silently drops old rows, and the policy gets worse (we saw 2757 → 710).
- The script checkpoints with a frozen evaluation every 25 episodes and waits for an installed model before each one.

**Expect:** a rising online clear rate and a frozen clear somewhere around 8,000 Records. Ours went 1/45 → 11/27 →
18/26 → 19/24 clears per stage, with the first frozen clear at the start of stage 5. Your exact numbers will differ
(see the note on refits above).

## Regenerating the datasets

Every dataset regenerates from the code, except the warm-start takeoff file (see
[`runs/warm-start/README.md`](runs/warm-start/README.md)). Each `*.provenance.json` records the command, the package versions and the
input hashes. Its `git.commit` refers to our private development repository, not this one. `scripts/historical/` keeps
the exact versions of generators that have changed since they produced data.
