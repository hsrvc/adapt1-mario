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
- Adapt-1 bills **credits** from one pool: "a query and a record each cost one credit", reads included (since
  November 2026; our earlier runs were billed as separate Records and Queries, and their costs below use those names).
  Check your balance at
  [app.reilabs.org/adapt-1](https://app.reilabs.org/adapt-1) before and after a run; the API doesn't report it.
- Record the server build: `GET https://rei-neuroadapt-api.reilabs.org/version` (host root, not `/api/v1`) returns
  `commit_sha`. `zero_start_duo.py` and `rebuild_domains.py` write it to the run log at the start and the end. The
  other scripts do not, so note it yourself.
- Use a **new domain name** for every run. Reusing one mixes your rows with old ones, and `--recreate` deletes any
  existing domain with that name first.
- Run each script that sends rows (`machina_acquire.py`, `ingest_demos.py`, `zero_start_duo.py`,
  `rebuild_domains.py`, `refine_frontier.py`) with `--dry-run` first. It runs the same code against an offline stub and spends nothing. The evaluation scripts spend Queries only.

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

## 3b. Machina on 2-1: ≈ 350 Records

Attempt 1 plays the 1-1 sequence from `runs/machina/` on 2-1. Then Machina acquires as in §3.

```bash
SEED=runs/machina/logs/mario-machina-z7r-frozen-SuperMarioBros-1-1-v0-20261002T044720Z.jsonl.gz
python scripts/machina_acquire.py --domain-id my-machina-2-1 --env SuperMarioBros-2-1-v0 --level-length 3193 \
  --attempts 1227 --horizon 256 --capacity 96 --seed-sequence $SEED \
  --profile coherent_edits,global_recall,progress_guided_mutation,sequence_compaction,outcome_tier_exploration,retain_prefix_failures=false
python scripts/machina_frozen.py --domain-id my-machina-2-1 --env SuperMarioBros-2-1-v0 --level-length 3193   # Queries only
```

**Expect:**

- Attempt 1 stops at x = 735.
- From attempt 170, Mario is at the flagpole when the 256 commands end. The game does not count these attempts as
  clears (see [`runs/machina-2-1/`](runs/machina-2-1/#attempt-170-and-attempt-278)).
- The first counted flag is at attempt **278**. The run stops 64 attempts later, at 342 Records.
- The frozen replay reaches the flag (x = 3193) on 3 of 3 repeats.

Compare your attempts with `runs/machina-2-1/logs/*-attempts.jsonl.gz`. We ran this setup once. Machina was
deterministic on 1-1, so we expect your run to match ours. If it does not match, please report it.

Checks without a key (0 Records):

```bash
J=runs/machina-2-1/logs/mario-machina-2-1-seeded-b-20261003T132903Z.jsonl.gz
# attempt 278 reaches the flag
python scripts/machina_replay.py --journal $J --index 277 --use-executed --env SuperMarioBros-2-1-v0 --out 278.gif
# attempt 1 beside attempt 278 (the GIF on the 2-1 page)
python scripts/machina_before_after.py $J before-after.mp4
# the frozen replay with the command panel
python scripts/machina_dashboard.py --env SuperMarioBros-2-1-v0 --first-flag 278 --attempts 342 \
  --origin "started from its own 1-1 sequence" --out machina-2-1.mp4 \
  --journal runs/machina-2-1/logs/mario-machina-2-1-seeded-b-frozen-SuperMarioBros-2-1-v0-20261003T134046Z.jsonl.gz
```

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

## 5b. Correctors for 2-1 on a frozen 1-1 base: ≈ 7,300 credits, plus the base

A base that clears 1-1 stays frozen. Two new domains learn when to replace one of its moves, until the frozen system
reaches the 2-1 flag and still clears 1-1. Read [`runs/refine-2-1/`](runs/refine-2-1/) first: it says what our code
does and what Adapt-1 does.

**The base.** The base is the 1-1 player: a takeoff domain (moves on the ground) and an apex domain (choices in
mid-air) that clear 1-1 frozen, for example your pair from §5. Ours was built in another way, because by our account
the server no longer held the October pair's model:

```bash
# the 8,498 rows of the October zero-start pair, fed again in the original order: 8,512 credits
gunzip -kf runs/base-rebuild/rows-base.jsonl.gz
python scripts/rebuild_domains.py --account mine --rows runs/base-rebuild/rows-base.jsonl \
  --config-dir runs/base-rebuild --which base --create --route explicit --max-credits 9500
# then 1,000 more rows of online play, as in §5: 2,420 credits
python scripts/zero_start_duo.py --takeoff-domain mario-zero2r-takeoff --apex-domain mario-zero2r-apex \
  --max-samples 16384 --features full_v3 --apex-features apex_v3 --credit frame_horizon --horizon 128 \
  --model-type extra_trees --exploration-mode ucb --episodes 1000 --seed 200 --max-records 1000 --eval-every 5 \
  --eval-seeds 777 --stop-flat 100 --max-credits 3200 --out-dir artifacts/base2-s1 --run-id base2-s1
```

The two config files name the domains `mario-zero2r-takeoff` and `mario-zero2r-apex`. Domains belong to one account,
so these names are free on yours the first time. For a second attempt, change `domain_id` in both config files. After
the feed alone our pair reached 2737 on 1-1, with no flag. After the 1,000
online rows it reached the flag at three checkpoints in a row. Do not send the base any rows after this point.

**The correctors.** Leg 1 creates them and works forward from the first place where Mario fails:

```bash
python scripts/refine_frontier.py --account mine --base-takeoff-domain mario-zero2r-takeoff \
  --base-apex-domain mario-zero2r-apex --create-refiners --model-max-weight 0.9 --min-context-observations 4 \
  --abstain-on-ood --refine-takeoff-domain my-takeoff-fix-2-1 --refine-apex-domain my-apex-fix-2-1 \
  --keep-dominance --keep-anchor 1 --anchor-1-1 1 --max-override-ratio 1.0 --leak-patience 6 --rounds 12 \
  --repeat-cap 100 --lead-repeats 6 --block-repeats 4 --fight-patience 3 --archive-records 0 --workers 4 \
  --max-records 15000 --max-credits 15000 --reward-max 0.002 --override-cost 0.05 \
  --out-dir artifacts/fix --run-id fix
```

Add `--dry-run` first: it runs the same loop against an offline stub and spends nothing. The stub has no trained
player, so the dry run ends with `RETENTION: 1-1 frozen 315 (no flag)` and `"outcome": "balance_unmet"`. That is the
expected end of a dry run. `--account` is only a label that the log records. The script refuses to send a row to the
base domains. Keep `--reward-max 0.002` and `--override-cost 0.05`: the script's defaults differ, and our three tries
without both of them did not work.

If a leg stops before the flag, start the next leg on the same correctors: drop `--create-refiners`, and add
`--depth 4 --keep-fixes-from <the frontier.jsonl of every earlier leg>`. Our run took two legs (4,365 and 1,733
credits) and two checks after the flag (582 and 577 credits: the same command with `--rounds 0` and smaller caps, and for the
live check a new `--out-dir`). The exact command of each is in `runs/refine-2-1/refine12/*.provenance.json`. This script does not record the server build.

**Expect:**

- Each round prints one frozen 2-1 game and one frozen 1-1 game. The 1-1 game must show the flag and 0 overrides.
- A correction can drop out after later rows. The script then sends a few repair rows and plays the game again.
- Our frozen 2-1 reach went 736 → 1654 → 1817 → 2098 → 3026 → 3193 (the flag) in 5 corrections.
- **Your numbers will differ.** The server does not fit the same model twice from the same rows, so your base plays
  its own route through 2-1 and needs its own corrections. The run can also stop with no flag: the method fixes one
  move at a time, and a place that needs several new moves together is out of its reach. On one of our earlier bases
  the tower at x = 3026 was such a place (our analysis, and its evidence is not in this repo).
- Reproducible here means the rows, the code and the evaluations. The server's model fits are not bit-identical.

Checks without a key (0 credits):

```bash
L=runs/refine-2-1/refine12/leg2-frontier.jsonl.gz
python scripts/refine_replay.py --log $L --tag final --out 2-1-flag.gif    # the frozen 2-1 game: 46 decisions, flag
python scripts/refine_replay.py --log $L --tag r01-1-1                     # the frozen 1-1 game: flag, 0 corrections
python scripts/pub/verify_runs.py runs                                      # every bundle against its hashes
```

## 6. Watch your run

Film a run on the dashboard, with Adapt-1's score for every move at every decision:

```bash
python scripts/dashboard_live.py --domain-id my-takeoff-1 --apex-domain my-apex-1 --out my-run.mp4
```

It plays one frozen run (learning off, no feedback) on your two domains, waits until both models are installed, and
saves every decision with its scores to `artifacts/dashboard-live/`. Cost: about 80 Queries, 0 Records. To render a
saved run again for free, pass `--from-evidence <that file>` instead of the domain names.

For Machina, after `machina_frozen.py`, film the sequence it played (offline, nothing spent):

```bash
python scripts/machina_dashboard.py \
  --journal "$(ls -t artifacts/machina/my-machina-1-frozen-SuperMarioBros-1-1-v0-*.jsonl | head -1)" --out machina.mp4
```

## Regenerating the datasets

Every dataset regenerates from the code, except the warm-start takeoff file (see
[`runs/warm-start/README.md`](runs/warm-start/README.md)). Each `*.provenance.json` records the command, the package versions and the
input hashes. Its `git.commit` refers to our private development repository, not this one. `scripts/historical/` keeps
the exact versions of generators that have changed since they produced data.
