# Notes on the corrector run: the full record

[`README.md`](README.md) is the short version. These notes hold the detail behind it: how a move gets played in a
frozen game, what went wrong on the way, the checks, the three retired tries, every limit we know of, and the cost.
Each number here was checked against the logs in this folder. Where a statement is our own account and no log
holds it, the text says so.

## Who does what

- **Our code finds each correction.** It replays the frozen game to just before the place where Mario fails. At
  each of the last few decisions it tries every other move once, as a trial that it reports to Adapt-1 with its
  reward. It repeats the best trial until that move plays in the frozen game.
- **Adapt-1 learns when to apply each correction.** A corrector sees the player's numbers, the player's own move,
  the jump that started the arc (mid-air corrector only), three flags for what is ahead (gap, obstacle, enemy), and a few products of these.
  It sees no position and no level name. It must
  apply the 5 corrections in the right situations on 2-1 and none on 1-1.
- **In a frozen game a fixed rule in our client turns Adapt-1's scores into a move.** The next section shows how.
  4 of the 5 corrections play through this rule. 1 is Adapt-1's own choice.
- **A frozen game sends no rows.** The client forces no move by decision number, and no emulator state enters a
  decision. All 42 frozen games in these logs show 0 rows sent and no forced move.

## How a move gets played in a frozen game

At each decision the client asks a corrector: keep the player's move, or replace it? Adapt-1 answers with a score
for each option and an interval that says how sure it is of each score. Then one of two things happens.

- **Adapt-1 chooses** (`selected` in the logs). The client plays that choice.
- **Adapt-1 does not choose** (`abstained`). We switch on two settings of Adapt-1 for the correctors. With
  `require_credible_dominance`, by our reading, Adapt-1 chooses only when the interval of one option lies above
  the interval of every other option. When two replacements both score far above "keep" and their intervals
  overlap, it names no winner. With `abstain_on_ood` it also abstains in a situation unlike its rows. An answer
  with no choice normally means: play the player's move. In 44 of the 47 abstentions in the third row of the table
  below, "keep" has the top score anyway. The lines of a frozen game do not say why Adapt-1 abstained.

Our client adds one rule for the second case. It applies only when Adapt-1 abstained and gave the reason
`credible_intervals_overlap`. The client takes the replacement whose interval starts highest. If that start is
above the end of the interval for "keep", the client plays that replacement. In every other case it plays the
player's move. The choice is by where the interval starts, not by the score: in one logged game the rule played
the replacement with the second-highest score. The rule uses only numbers from Adapt-1's answer. The code is `keep_dominance_pick` in
`src/typesafe_mario/refinement.py`.

An example from the flag game, the mid-air choice at decision 9 (the first piranha pipe). Adapt-1's scores:

| option | score |
|---|---|
| keep the player's move (here "keep going") | 0.12 |
| replace it with "brake" | 0.94 |
| replace it with "keep going" (the same move, scored as a replacement) | 0.79 |
| replace it with "pull back" | 0.71 |

All three replacements score far above "keep". The log marks this answer `keep_dominance`: Adapt-1 abstained, and
the client rule played "brake". Without the rule the player's own move plays here, and Mario dies at this pipe.
The line of a frozen game holds the scores and this mark. It does not hold the intervals or the reason. Those are
in the `probe` lines, for example the probe when this correction was first accepted (`refine12/leg1`, round 1,
24 repeats, 794 rows before the flag game): keep [0, 0.30], brake [0.79, 1].

All 83 corrector answers of the flag game (46 on the ground, 37 in mid-air):

| Adapt-1's answer | count | what played |
|---|---|---|
| chose "keep the player's move" | 31 | the player's move |
| chose a replacement | 1 | the replacement (correction 5) |
| abstained, and the client rule did not apply | 47 | the player's move |
| abstained, and the client rule applied | 4 | the replacement (corrections 1 to 4) |

So the scores are Adapt-1's: it learned from the trials that these replacements score high in these situations
and not on 1-1. The last step, from scores to a move, is our rule for 4 of the 5 corrections. By our account, our
October attempt used the same rule.

## The corrections

| step | correction (decision, x) | frozen 2-1 after it | frozen 1-1 |
|---|---|---|---|
| start | none | 736 | flag |
| 1 | brake in mid-air (9, x 724) | 1654 | flag, 0 corrections |
| 2 | plain jump, A only (24, x 1548) | 1817 | flag, 0 corrections |
| 3 | pull back in mid-air (28, x 1668) | 2098 | flag, 0 corrections |
| 4 | pull back in mid-air (31, x 1941) | 3026, stuck at the tower | flag, 0 corrections |
| 5 | short running jump (41, x 2967) | **3193, the flag** | flag, 0 corrections |

The path was not this smooth. The logs show each of these:

- **Step 2:** the first frozen game after it reached 921, because the corrector also played the new jump at
  decision 9. That jump was tried once at decision 9 in step 1 and reached 921 there too. Two repairs followed
  (921 → 1646 → 1817). A repair is 6 more rows at the decision where the frozen game left the tested one: "keep"
  rows where a move should not play, or correction rows where a correction dropped out.
- **Step 4:** the first candidate (a jump at decision 32) undid the correction at decision 24 three times in a
  row. The correction in the table is the second candidate.
- **After step 4:** a failed round sent 107 more rows, and the next frozen game was back at 1817. The correction at
  decision 28 had dropped out. Six repair rows restored 3026. **Later rows can undo a correction.**
- **Step 5:** the first try at the tower tested decisions 51 to 53 and found nothing. By our reading Mario only
  bounces against the wall there. We then changed the rule that picks the decisions (by our account: start from the
  decision that reaches the wall. In the logs: 4 decisions, not 3). The next leg tested decisions 41 to 38 and found the short jump.
- **A limit we set before the run was missed.** Step 2 should pass within 60 repeats and passed at 64. We continued.
  The 64 is in the log. The limit of 60 is in our private plan.

## Checks

| check | result | where |
|---|---|---|
| 79 more rows (all "keep" rows from 1-1), so both correctors refit | 2-1 flag again before any repair, 1-1 flag | `refine12/confirm-frontier.jsonl.gz` |
| a fresh start with an empty client cache: the player answers the first 2-1 and 1-1 game live (156 of 162 decisions, and 6 repeat an exact context) | 2-1 flag, 1-1 flag, the same moves | `refine12/live-base-frontier.jsonl.gz` |
| replay of the logged moves from the level start, no key | all 46 positions match, flag | the command below |
| the player during the run | 0 rows sent to it, and the same row counts and training time at the start of all 9 logs | the `start` line of each log, and `base_feedback_calls: 0` in every `summary` (7 of the 9 logs have one) |

Replay the two frozen games yourself (free, a few seconds):

```bash
python scripts/refine_replay.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag final --out 2-1.gif
python scripts/refine_replay.py --log runs/refine-2-1/refine12/leg2-frontier.jsonl.gz --tag r01-1-1
```

The replay shows that the logged moves reach the flag on the emulator. That the server chose those moves rests on
the logs.

## The three retired tries

`refine12` is the fourth corrector pair of the night. Each earlier pair is here with its logs.

| pair | reward settings | how it ended (from its log) |
|---|---|---|
| `refine9/` | no cap, no cost | step 1 passed. Step 2 did not pass in 404 repeats. A later frozen game was worse (1368, down from 1654) |
| `refine10/` | no cap, no cost | step 1 passed. Step 2 had not passed at 74 repeats. By our account we stopped the run by hand there. The log just ends |
| `refine11/` | cap 0.002, no cost | step 1 passed. In step 2 the repeats of the best candidate stopped playing the tested game (6 of its 10 plays reached 1730, not 1828) and the run stopped itself |
| `refine12/` | cap 0.002, cost 5% | the run that works |

A repeat count in these notes is the log's count. It includes the "keep" rows that the script sends beside the
tested move (for example 64 repeats = 40 plays of the move + 24 keep rows).

## What this does not show

1. **It is one player on one level.** Frozen games are deterministic, so 5 seeds would be 5 copies of one game.
2. **The method was tuned on this level,** over the four pairs above and the rule change before step 5.
3. **The reward is our choice:** Mario's progress in the next 512 frames, capped at 0.002 per trial, and multiplied
   by 0.95 for a correction (`reward_max`, `override_cost` and `horizon` in a log's `start` line. A missing key
   means the setting was off). What the logs
   show for these settings, with one run each: without the cap a correction did not pass in 404 repeats
   (`refine9`). With the cap and no cost, the mid-air margin that keeps 1-1 free of corrections was 0.10 after
   step 1 and 0.06 in step 2 (`refine11`). With both it was 0.14 and 0.13 (`refine12`).
4. **The method fixes one move at a time.** It cannot find a fix that needs several new moves together.
5. **Why this run passed the tower is partly luck of the route.** This is our own analysis and its evidence is not
   in this repo. In October we tried the same method on the original zero-start pair. It stopped at the same tower,
   x = 3026. On that pair's route the game did not create the springboard that stands there (a known bug of the NTSC
   game), and the method found no move that passes. On this player's route one changed jump is enough (that part
   is in the logs).
6. **An offline diagnosis also found the last correction.** This is our account and it is not in the logs. While the trials at decisions 41 to 38 ran, a reviewer
   who replayed the game offline reported that a change at decision 41 clears the tower. The trials were launched
   before that report, and no route was given to the learner. But we chose the new rule of step 5 after we saw
   where Mario reaches the wall.
7. **The flag holds on two corrector versions that differ by 79 "keep" rows from 1-1.** We sent no 2-1 rows after
   the flag, and 2-1 rows have undone corrections before.
8. **On 1-1 the mid-air corrector is quiet mostly because the server abstains** (30 of 31 mid-air answers in the
   1-1 game). The correctors also hold "keep" rows from 1-1 (316 of their 673 keep rows).
9. **The player is not the October zero-start domain pair.** By our account the server no longer held that pair's
   model. We fed its 8,498 rows again into new domains ([`../base-rebuild/`](../base-rebuild/)), which did not
   clear 1-1 (2737). The pair then played 1,000 more rows online (`base2/`) and cleared 1-1 at three checkpoints
   in a row, each with more rows than the one before.
10. **The server build is recorded for the player's stage (`f5e1ab8`), not for the corrector runs.** The corrector
    script did not read it.
11. **If you rerun this, your player will differ from ours.** The server does not fit the same model twice from the
    same rows. Your player may take another route through 2-1 and need other corrections, or stop at a place that
    one move cannot pass. Reproducible here means the rows, the code and the evaluations.

## Cost

Adapt-1 billed these runs in credits (1 per call: queries, rows and status reads). The script counts them.

| part | credits |
|---|---|
| the correctors that work (`refine12/`): leg 1 4,365, leg 2 1,733, the two checks 582 and 577 | ≈ 7,300 |
| the player's 1,000 extra rows (2,420) and its frozen checks (416 queries, not counted by the script) | ≈ 2,800 |
| the earlier feed of 8,498 rows ([`../base-rebuild/`](../base-rebuild/)) | 8,512 |
| the three retired pairs: 4,862 counted for `refine9/` and `refine11/`, and `refine10/` has no final count | ≈ 6,900 |

The last number is what is left of the 16,978 credits that our account page showed for the two days. All four
parts together are about 25,500 credits. The correctors hold 649 and 360 rows at the end (673 keep, 336 correction).

## Files

| path | what |
|---|---|
| `refine12/leg1-*`, `leg2-*` | the run that works: every trial, every frozen game, the credit count |
| `refine12/confirm-*`, `live-base-*` | the two checks after the flag |
| `refine12/leg2-stopped-*` | a first start of leg 2 on another commit of the script. By our account we stopped it by hand after one frozen game. The log just ends |
| `refine9/`, `refine10/`, `refine11/` | the three retired corrector pairs |
| `base2/` | the player's 1,000 online rows, its frozen checkpoints (GIFs) and its frozen games on 1-1 and 2-1 |
| `*.provenance.json` | the exact command of each run, and the commit when the file was written. The `start` line of a log has the commit at the start of the run |
| `index.json` | sha256 of every run file (not of the two pages, the GIF or the index itself). Check with `python scripts/pub/verify_runs.py runs/refine-2-1` |

Words in the logs: a **leg** is one start of the script on the same correctors. A line of kind `frozen` is one
frozen game with every decision: the player's move, the corrector's answer and scores, and the move that ran. A
line of kind `trial` is one tested move with its reward before the cap and the cost. Step status `selected` is the server's choice, and
`keep_dominance` is the client rule above. In a provenance command, `artifacts/refine12/run-20261116T174550Z/` is
`leg1-*` here and `run-20261116T221301Z/` is `leg2-*`.
