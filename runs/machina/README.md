# Machina from zero: first flag at attempt 403 (2026-10-02)

Machina is Adapt-1's trajectory engine (`/domains/{id}/trajectory/*`). Each attempt it proposes a whole sequence of
up to 256 controller commands (8 frames each) for the same start state. The harness plays it on the emulator and
reports what ran and how far Mario got. Machina keeps the best attempts and edits them for the next proposal.

- State: 8 numbers per step (position, speed, grounded, gap / obstacle / enemy distance). Actions: 4 channels decoded
  into NES buttons.
- Config: horizon 256, capacity 96, the option profile in [`REPRODUCE.md`](../../REPRODUCE.md), `seed: 1`, emulator
  `reset(seed=777)`.
- Cost: 1 Record per attempt (467) plus 2 to create and configure; 1 Query per proposal.

**Result:** first flag at attempt **403**, and 42 flags in the 467 attempts. The run stops 64 attempts after the first
flag. Frozen replay: **1-1 = 3161 on 3/3** (244 commands); 2-1 = 735 (it does not transfer).

**It is deterministic.** We ran this setup first on 2026-09-22 and again on 2026-10-02 on a different account. All 467
proposals were byte-identical. A rerun with the same config should match `logs/*-attempts.jsonl.gz` attempt for attempt.

## Files

| file | what |
|---|---|
| `logs/mario-machina-z7r-*.jsonl.gz` (largest) | the full journal: resolved config, every proposal, execution and response |
| `logs/*-attempts.jsonl.gz` | one line per attempt: `max_x`, `flag`, `source`, `improved_parent` |
| `logs/*-frozen-*.jsonl.gz` | the frozen replays on 1-1 and 2-1 |
| `replays/z7r-403-first-flag.gif` | attempt 403, rendered offline from the journal (`machina_replay.py`) |
| `*.provenance.json` | command, package versions, input hashes |
| `index.json` | sha256 of every file. The `session_id` values in the journals are blanked for publication |
