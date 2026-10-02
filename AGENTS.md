# Notes for coding agents

You are probably here to reproduce a result. [`REPRODUCE.md`](REPRODUCE.md) has the commands. These rules keep a run
cheap and valid:

1. **The API key lives in `REI_KEY` only.** Never write it into a file, a command you echo, a log or a commit.
2. **Live runs cost the user money.** Records are spent per row sent. Run offline first (`pytest -q`, then every script
   that spends Records with `--dry-run`) and tell the user the Record cost from REPRODUCE.md before you start a live run.
3. **Use a new domain name for every live run.** Don't reuse or delete the user's existing domains.
4. **A `200` response means the server accepted the request, not that anything learned.** Judge a run by a frozen
   evaluation: `max_x` 3161 means the flag. `frozen_eval.py --wait-installed` avoids evaluating while the model is
   retraining.
5. **Rewards must be in [0, 1]** (and, for sequential learning, add up to at most 1 per episode). The server silently
   clips anything else, and the learner can then refuse to choose (`abstained` on every query). The scripts already
   respect this; keep it if you change them.
6. **Retry 502, 503 and 504** with backoff (the client does this). Don't retry 409 or 413: fix the request instead.
7. The emulator is deterministic (`reset(seed=777)`), so one seed per evaluation is enough. The server's model fit is
   not, except for Machina.

## Where things are

| need | file |
|---|---|
| call the Adapt-1 API (auth, the required browser User-Agent, retries) | `src/typesafe_mario/adapt1_client.py` (stdlib only; key from `REI_KEY`, base URL overridable with `REI_API_BASE`) |
| check request shapes without spending | `src/typesafe_mario/offline_client.py` (what `--dry-run` uses) |
| query a domain and send feedback for a decision | `src/typesafe_mario/adapt1_policy.py` |
| Machina (`/domains/{id}/trajectory/*`) | `src/typesafe_mario/machina.py`, `scripts/machina_acquire.py` |
| the zero-start loop (credit, Record budget, wait for an installed model) | `src/typesafe_mario/online_duo.py`, `scripts/zero_start_duo.py` |
| the numbers sent per decision | `src/typesafe_mario/adapt1_policy.py` (feature sets), `src/typesafe_mario/cadence.py` |
| feed recorded rows to a domain | `scripts/ingest_demos.py` |
| evaluate with learning off | `scripts/frozen_eval.py`, `scripts/machina_frozen.py` |
| the warm-start rows | `runs/warm-start/*.jsonl.gz` |
