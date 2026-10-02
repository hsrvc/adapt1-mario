# Zero start: 1-1 learned from nothing (2026-10-01 → 02)

Run it: [`REPRODUCE.md` §5](../../REPRODUCE.md#5-zero-start--8500-records--15-hours) (≈ 8,500 Records, ≈ 15 hours).

![the first frozen clear, stage 5](stage-5/ep000-seed777.gif)

Two new, empty domains with the same design as the warm start (takeoff + apex, 8 macros, `extra_trees`, v3 features),
trained only on their own play. Credit: each decision is rewarded with Mario's progress per frame over the next 128 frames. Exploration:
UCB. Learner row budget (`max_samples`) 16,384, never reached. A frozen evaluation every 25 episodes on seed 777.

| stage | Records | episodes | online clears | frozen checkpoints (reach) |
|---|---|---|---|---|
| 1 | 2,000 | 45 | 1 | 315 → 722 → 2473 |
| 2 | 2,000 | 27 | 11 | 2020 → 1238 → 2018 |
| 3 | 2,000 | 26 | 18 | 2018 → 2017 |
| 4 | 2,000 | 24 | 19 | 2017 → 2776 |
| 5 | 431 | 5 | 5 | **3161 (flag)** at the opening checkpoint, after 8,000 Records |

The run stopped after the frozen clear. Once the model had refit on stage 5's 431 rows, it was evaluated again on seeds
777, 1, 2, 3 and 4 (`stage-5/confirm-after-refit/`): **3161 on all five**, by a different route through the pit before
the staircase.

![learning curve](learning-curve.png)

## Each stage folder

| file | what |
|---|---|
| `run-*.jsonl` | every episode, checkpoint and summary the loop logged |
| `run-*.jsonl.provenance.json` | the command, the full config, package versions |
| `curve.json` | online reach and frozen checkpoints, as plotted |
| `ep*-seed777.gif` | the frozen checkpoints, played |
| `console.log` | the loop's console output |
| `index.json` | sha256 of every file |

Stage 5 also has `confirm-after-refit/` (the five-seed evaluation after the refit) and
`model-state-after-frozen-clear.json` (the learners' state right after the clear).

## Reading the numbers

- Frozen reads move by hundreds of pixels between checkpoints even when little has changed. The server refits on every
  new batch of rows. Judge the trend and the online clear rate, not one read.
- The first clear was narrow: in stage 4 the same moves died just short. The five-seed confirmation after the refit
  shows it holds.
