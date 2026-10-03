# Machina clears World 2-1

Machina, Adapt-1's trajectory engine, cleared World 2-1 by starting from the button sequence it had found for 1-1.
The first clear came at attempt 278. With learning switched off, the sequence it kept clears 2-1 on 3 of 3 replays.
The run took 342 attempts on 2026-10-03, at 1 Record each.

Run it yourself: [`REPRODUCE.md` §3b](../../REPRODUCE.md#3b-machina-on-2-1--350-records).

## Watch it play

Attempt 1 next to attempt 278, at twice the speed. Attempt 1 plays the 1-1 sequence on 2-1 and dies at a piranha
plant at x = 735. Attempt 278 reaches the flag.

![attempt 1 dies at x = 735; attempt 278 reaches the flag](videos/before-after-attempt1-vs-278.gif)

The sequence Machina kept, replayed with learning off. The panel shows the command now playing, how long each button
is held, and the whole 256-command sequence with a playhead:
[machina-2-1-frozen-clear.mp4](videos/machina-2-1-frozen-clear.mp4) (36 s).

## How it started from 1-1

The setup is the [1-1 run](../machina/)'s, with one change. Attempt 1 ignores Machina's proposal and plays the
sequence Machina replays on 1-1 with learning off. From then on Machina edits its best attempts, as it does on 1-1.
No demonstration or scripted player is involved; the starting point is Machina's own earlier result.

| attempt | furthest x | what happened |
|---|---|---|
| 1 | 735 | the 1-1 sequence dies at a piranha plant |
| 5 | 1736 | past the first wall |
| 154 | 3026 | at the springboard below the 9-block tower |
| 170 | 3197 | over the tower and at the flagpole when the attempt ends |
| 278 | 3193 | first counted clear |

After the first clear, 33 of the next 65 attempts also cleared. The run stops 64 attempts after its first flag.

## Attempt 170 and attempt 278

An attempt holds at most 256 commands of 8 frames, about 34 seconds of game time. That is the longest sequence
Machina accepts, so the harness stops the game when the commands run out and scores the attempt.

From attempt 170 on, Mario is in the air at the flagpole when the commands run out, 1 to 5 frames before the game
registers the grab. We let the emulator run those attempts a few frames longer, offline: all 57 of them, from 170 to
277, then get the flag. The level was in reach from attempt 170. Attempt 278 is the first to grab the pole within
the 256 commands, and that is the number we report.

The same near-misses explain a rule in how attempts are scored. An attempt scores the distance it reached divided by
the flag's position, 3193 on 2-1, and a clear scores 1.0. Mario in the air just past the pole is at x = 3206, which
would score 1.004 and beat a real clear. In our first run on 2-1 that happened: the near-misses became Machina's best
attempts and it stopped improving. The harness now caps an attempt without the flag at 0.995. This run used the cap
from its first attempt. It makes no difference on 1-1, where no failed attempt came within 690 px of the flag.

## What the sequence can and can't do

The sequence clears 2-1 only. Played on 1-1 it reaches x = 1262, and the 1-1 clear stays in
[`../machina/`](../machina/). Machina sees Mario's position and keeps one sequence per level, so it does not react
to what is on screen the way the domain recipes do.

Machina replayed 1-1 identically on two accounts. We have run this 2-1 setup once, so we expect a rerun to match it
attempt for attempt but have not shown it.

## Files

| file | what |
|---|---|
| `logs/mario-machina-2-1-seeded-b-2026*.jsonl.gz` | the full journal: resolved config, every proposal, execution and response |
| `logs/*-attempts.jsonl.gz` | one line per attempt: `max_x`, `flag`, `source`, `improved_parent` |
| `logs/*-frozen-*.jsonl.gz` | the replays with learning off, on 2-1 and on 1-1 |
| `videos/` | the clips above, rendered offline from the journals |
| `*.provenance.json` | command, package versions, input hashes |
| `index.json` | sha256 of every file. The `session_id` values in the journals are blanked for publication |
