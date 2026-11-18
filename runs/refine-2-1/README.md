# Correctors clear World 2-1 and keep 1-1

Run it: [`REPRODUCE.md` §5b](../../REPRODUCE.md#5b-correctors-for-2-1-on-a-frozen-1-1-base--7300-credits-plus-the-base)
(≈ 7,300 credits for the correctors, plus the player).

The player is a pair of domains like the [zero start](../zero-start/): one picks the moves on the ground (takeoff),
the other the choices in mid-air (apex). It knows only World 1-1. On 2-1 it dies at the first piranha pipe (x = 736).
In this run (2026-11-16 and 17) it is frozen, so it gets no more feedback rows. Two small new domains, the correctors, sit on top of it,
one for each half of the pair. At each decision a corrector keeps the player's move or replaces it with one other
move.

With learning switched off everywhere, the player and its correctors reach the 2-1 flag (x = 3193) in 46 decisions.
5 of those are corrections. On 1-1 the correctors change nothing and the player still reaches the flag.
[Machina](../machina-2-1/) also clears 2-1, but the sequence it ends with no longer clears 1-1.

## Watch it play

The frozen 2-1 game, replayed from the log. The panel shows Adapt-1's scores for keeping the player's move and for
the best replacements, on the ground and at the top of each jump, and how each move was picked. The picture stops
for a moment at each of the 5 corrections.

<!-- VIDEO: in the GitHub web editor, drag correctors-2-1.mp4 onto this line -->

Render it yourself from the log, with no key:
`python scripts/refine_dashboard.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag final --out correctors-2-1.mp4`

## The corrections

| step | correction (decision, x) | frozen 2-1 after it | frozen 1-1 |
|---|---|---|---|
| start | none | 736 | flag |
| 1 | brake in mid-air (9, x 724) | 1654 | flag, 0 corrections |
| 2 | plain jump, A only (24, x 1548) | 1817 | flag, 0 corrections |
| 3 | pull back in mid-air (28, x 1668) | 2098 | flag, 0 corrections |
| 4 | pull back in mid-air (31, x 1941) | 3026, stuck at a tall tower near the end | flag, 0 corrections |
| 5 | short running jump (41, x 2967) | **3193, the flag** | flag, 0 corrections |

## Who does what

Our code finds each correction. It replays the frozen game to just before the place where Mario fails. At each of
the last few decisions it tries every other move once and reports each trial to Adapt-1 with its reward. Then it
repeats the best trial until that move plays in the frozen game.

Adapt-1 learns where each correction applies. A corrector sees what the player sees, the player's own move and a few
extra values. It sees no position and no level name. It has to apply the 5 corrections in the right situations on
2-1 and none on 1-1. The correctors' rows include "keep" rows from 1-1 (316 of their 673 keep rows), which is how
they learn to stay quiet there.

A rule in our client turns those scores into a move. Each score comes with an interval that says how sure Adapt-1
is, and we set the correctors to choose only when they are sure. So Adapt-1 often answers with scores and no
choice: 51 of the 83 corrector answers in the flag game. The client then plays the player's move, with one
exception. If Adapt-1 held back because its intervals overlap, and its interval for a replacement lies above its
interval for keeping, the client plays that replacement. 4 of the 5 corrections play through this rule and 1 is
Adapt-1's own choice. The exact rule and a worked example are in
[`NOTES.md`](NOTES.md#how-a-move-gets-played-in-a-frozen-game).

A frozen game sends no rows, and no move in it is scripted.

## Reading the result

- We tuned the method on this level. This run, `refine12`, is the fourth corrector pair we tried. The three retired
  pairs are here with their logs.
- The method fixes one move at a time. A place that needs several new moves together is out of its reach. By our
  account that happened in October: on the original zero-start pair the same method stopped at this same tower.
- This is one player on one level. Frozen games are deterministic, so more seeds would repeat the same game.
- Later rows can undo a correction. After step 4 one dropped out, and 6 more rows put it back.
- The reward is our choice: Mario's progress over the next 512 frames, capped at 0.002. A correction's reward is
  reduced by 5%.
- The player is a rebuilt copy of the zero-start pair with 1,000 more rows of play (`base2/`).
  [`../base-rebuild/`](../base-rebuild/) has the rebuild.
- A rerun gives a different player, because the server does not fit the same model twice from the same rows. It may
  need other corrections, or stop at a place one move cannot pass.

The correctors that work cost about 7,300 credits. With the player and the three retired pairs the total is about
25,500.

## Check it yourself

These replay the two frozen games from the logged moves, with no key, in a few seconds:

```bash
python scripts/refine_replay.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag final --out 2-1.gif
python scripts/refine_replay.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag r01-1-1
```

## Files

| path | what |
|---|---|
| `refine12/` | the run that works: two runs of the script, two checks after the flag, and one aborted start |
| `refine9/`, `refine10/`, `refine11/` | the three retired corrector pairs |
| `base2/` | the player's last 1,000 online rows and its frozen games on 1-1 and 2-1 |
| `*.provenance.json` | the exact command of each run |
| `index.json` | sha256 of every run file |
| [`NOTES.md`](NOTES.md) | the full record: what went wrong on the way, the checks, every limit, the cost by part |
