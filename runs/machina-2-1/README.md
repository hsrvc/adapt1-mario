# Machina on 2-1, started from its own 1-1 sequence: first flag at attempt 278 (2026-10-03)

Run it: [`REPRODUCE.md` §3b](../../REPRODUCE.md#3b-machina-on-2-1--350-records) (≈ 350 Records).

![attempt 278 reaches the 2-1 flag](replays/seeded-b-278-first-flag.gif)

Same engine, config and decoder as the [1-1 run](../machina/), on World 2-1. The first attempt does not use
Machina's proposal: it plays the sequence Machina found on 1-1 from zero (attempt 403 of that run), which reaches
x = 735 on 2-1 and dies at a piranha pipe. Machina then edits from there, like any other attempt. No demonstration,
no scripted player: the starting point is Machina's own earlier result.

| attempt | reach |
|---|---|
| 1 (the 1-1 sequence) | 735 |
| 5 | 1736 |
| 154 | 3026, the springboard under the 9-block tower |
| 170 | 3197, over the tower, out of commands in mid-air |
| **278** | **flag** (x = 3193) |

- Flags: 33 of the 65 attempts from the first flag on. The run stops 64 attempts after the first flag.
- Frozen replay: **2-1 = 3193 flag on 3/3** (all 256 commands). The same sequence on 1-1 reaches 1262: one Machina
  sequence is one level.
- Cost: 1 Record per attempt (342) plus 2 to create and configure.
- 2-1's flag is at x = 3193 (1-1's at 3161), so the goal and the outcome are scaled by 3193. An attempt that misses
  the flag scores at most 0.995, below a clear's 1.0. Without that cap, a first run scored an attempt that flew over
  the pole (x 3206) above a clear and stopped improving.

Machina was deterministic on 1-1 across two accounts. We have run this 2-1 setup once, so a rerun matching it
attempt for attempt is an expectation, not yet a result.

## Files

| file | what |
|---|---|
| `logs/mario-machina-2-1-seeded-b-2026*.jsonl.gz` | the full journal: resolved config, every proposal, execution and response |
| `logs/*-attempts.jsonl.gz` | one line per attempt: `max_x`, `flag`, `source`, `improved_parent` |
| `logs/*-frozen-*.jsonl.gz` | the frozen replays on 2-1 and 1-1 |
| `replays/seeded-b-278-first-flag.gif` | attempt 278, rendered offline from the journal (`machina_replay.py`) |
| `*.provenance.json` | command, package versions, input hashes |
| `index.json` | sha256 of every file. The `session_id` values in the journals are blanked for publication |
