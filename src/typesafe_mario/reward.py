"""Reward shaping — the objective the learner actually optimises.

Adapt-1 has **no built-in goal**: whatever `values.reward` carries is exactly what the
sequential/bandit learner maximises, loopholes included (design-notes §5). The gym default
(`SuperMarioBros-v0`: instantaneous velocity + a clock penalty + a −15 death spike, each
step) is *dense and unbounded*, and finding #15 showed why that is a trap — reward ≈ Δx is
so trivially predictable that `validation_skill` hit 0.9993 while **play did not improve at
all** (both cold and warm stalled at the second pipe). A high prediction score on an easy
target says nothing about the policy.

This module builds the §5 **composite, normalised** reward from *state deltas* (not the
opaque gym scalar), so every term is explicit and tunable:

    r = w·(Δx / progress_scale)  −  time_penalty  (+ flag_bonus | − death_penalty)   clipped

- **progress** (dense) — Δx per decision, the every-step signal a bandit can already use.
- **terminal** (sparse) — a big **+flag** / **−death** spike; deliberately several × a
  normal step and clipped so a death lands firmly negative and a flag firmly positive.
  Sparsity here is the knob that makes the *sequential* learner (delayed credit) earn its
  keep — a bandit can chase Δx, but only backward credit assigns the death to the jump you
  skipped 20 frames earlier.
- **time** — a small per-decision drag so dawdling costs something (efficiency nudge).

Normalised to ~[−1, 1] for stability (design-notes §5; [0, 1] would be required only if we
later add CUP). Coins/kills are deliberately **absent** — rewarding them makes a *different*
agent (§5); those are separate studies, not the completion objective.

`shaped_reward` is pure and deterministic — unit-tested offline, no env, no API. Passing
``config=None`` at the call sites keeps the raw gym reward, preserving the finding-#15
baseline for a clean A/B.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the composite reward. Defaults give a normal decision ≈ +0.3, a death
    ≈ −1, a flag ≈ +1 (terminal events ~3× a step and clipped, so they dominate)."""

    progress_scale: float = 80.0  # px of Δx mapped to 1.0 before weighting (~40px step → 0.5)
    progress_weight: float = 1.0  # primary, dense
    time_penalty: float = 0.01  # per-decision efficiency drag
    flag_bonus: float = 1.0  # added on stage clear
    death_penalty: float = 1.0  # subtracted on death
    clip: float = 1.0  # final clamp to [-clip, clip]
    # --- obstacle-aware terms (findings #17: the x=723 lever) -------------------------
    # The pipe is a *delayed-consequence* decision: run-jump and run give ~equal immediate
    # Δx over the takeoff frames, so the dense term above carries almost no signal at the one
    # decision that matters. These two terms restore an *immediate* jump-vs-run separation at
    # a flagged obstacle, so "jump the pipe" is learnable without leaning on delayed credit.
    # ⚠️ This credits the correct *outcome* directly (leaving the ground at an obstacle) — the
    # same shortcut Jev takes by being told the rule. It is a lever to beat 723, not a claim
    # that Adapt-1 discovered the pipe-jump from reward alone. Keep that distinction in results.
    # Both default to 0.0 → shaped_reward is byte-identical to the pre-#17 baseline (tests pin
    # this), so the obstacle-aware arm is opt-in and A/B-clean against the shaped baseline.
    obstacle_jump_bonus: float = 0.0  # + when grounded→airborne at a flagged obstacle (committed)
    obstacle_stall_penalty: float = 0.0  # − when grounded & ~0 Δx at a flagged obstacle (jammed)
    stall_dx: float = 8.0  # px; |Δx| below this at a flagged obstacle counts as stalling
    # --- grounded-cadence terms (findings #21, macros.py) -------------------------------
    # A macro spans up to ~70 frames and ~190 px, so with the frame-cadence scale a death at
    # the end of a long jump would net POSITIVE (2.3 − 1 → clipped to +1). `terminal_overrides`
    # makes a terminal step's reward exactly ±the terminal term, ignoring progress; `frames`
    # (an argument to shaped_reward) scales the time penalty per 8 frames so a long macro
    # costs proportionally more idle time. Both default off → byte-identical to the baseline.
    terminal_overrides: bool = False
    # findings #22 (online curve, 05:29): with reward per DECISION and the server discounting per
    # decision, a 152-px full jump (51 frames) pays 6.6x a 23-px run step (8 frames) although both
    # move ~3 px/frame — so the learner converged to "jump big everywhere" and max-jumped off pipe 4
    # into the pit. `per_frame` pays progress per 8 frames instead (Δx × 8 / frames), removing the
    # macro-length bias; the episode sum stays bounded (≤ ~1 while speed ≤ 3 px/frame).
    per_frame: bool = False

    def __post_init__(self) -> None:
        if self.progress_scale <= 0:
            raise ValueError("progress_scale must be positive")
        if self.clip <= 0:
            raise ValueError("clip must be positive")
        if self.obstacle_jump_bonus < 0:
            raise ValueError("obstacle_jump_bonus must be non-negative")
        if self.obstacle_stall_penalty < 0:
            raise ValueError("obstacle_stall_penalty must be non-negative")
        if self.stall_dx <= 0:
            raise ValueError("stall_dx must be positive")


DEFAULT = RewardConfig()

# Preset for the #17 obstacle-aware arm. Symmetric ±0.5 so a pipe-jump and a pipe-stall land a
# full ~1.0 apart (vs ~0.2 from Δx alone) while both stay inside the [-1, 1] clip in practice.
OBSTACLE_AWARE = RewardConfig(obstacle_jump_bonus=0.5, obstacle_stall_penalty=0.5)

# Preset for the grounded cadence (macros.py). progress_scale 240 → a max run-jump (~190 px)
# reads ~0.8 and an 8-frame walk step ~0.1; terminal_overrides keeps death at exactly −1 and
# the flag at +1 regardless of the Δx that preceded them.
GROUNDED = RewardConfig(progress_scale=240.0, terminal_overrides=True)

# findings #22 (2026-09-21): the engine's sequential target is the n-step SUM of per-step
# utilities, and every decision score is clipped to [0, 1] ("keep rewards in [0,1]",
# sequential-learning.md §6). Both earlier designs saturated: [-1,1] rewards floored at 0
# (every macro tied at 0), and (r+1)/2 rewards summed past 1 (every macro tied at 1). Dense
# rewards can only stay inside the contract if they sum to <= 1 over a whole episode, so
# this preset pays progress as a FRACTION OF THE LEVEL (x=3161 at the 1-1 flag): a 23-px
# step is 0.007, a max jump 0.05, a full clear sums to exactly 1.0. No time penalty, no
# terminal bonuses: a death is punished by the future progress it forfeits (the value of
# the states before it drops), a flag by completing the sum. Backward motion clips to 0
# via the component's declared min. Declare the component with min 0, max 1.
LEVEL_LENGTH_PX = 3161.0
PROGRESS_FRACTION = RewardConfig(
    progress_scale=LEVEL_LENGTH_PX, time_penalty=0.0, flag_bonus=0.0, death_penalty=0.0
)
# Same, per 8 frames: a full jump (152 px / 51 f) and a run step (23 px / 8 f) both pay ≈ 0.007–0.008,
# so a jump is only worth choosing where it buys future progress (a pit, a pipe, an enemy).
PROGRESS_RATE = RewardConfig(
    progress_scale=LEVEL_LENGTH_PX,
    time_penalty=0.0,
    flag_bonus=0.0,
    death_penalty=0.0,
    per_frame=True,
)


def shaped_reward(
    *,
    dx: float,
    terminal: bool,
    cleared: bool,
    dead: bool,
    obstacle_ahead: bool = False,
    grounded: bool = False,
    left_ground: bool = False,
    frames: int = 8,
    config: RewardConfig = DEFAULT,
) -> float:
    """Composite, normalised reward for one decision.

    Args:
        dx: forward progress this decision, ``next_snapshot.x - snapshot.x`` (pixels).
        terminal: whether the episode ended on this decision (death or flag).
        cleared: the flag was reached (a terminal win).
        dead: Mario died (a terminal loss). Only counts when ``terminal`` is also true —
            a mid-episode ``dead`` flag without termination is ignored.
        obstacle_ahead: a terrain obstacle is flagged within reach at the *decision* (raw
            ``snapshot.navigation_features()['obstacle_ahead']`` — arm-independent, so the
            reward target is identical across the full/perception arms and can't confound them).
        grounded: Mario was on the ground at the decision. The obstacle-aware terms only apply
            grounded, so an airborne pipe-crossing is never mistaken for a stall.
        left_ground: Mario was grounded at the decision and airborne after it — i.e. the
            decision *committed a jump*. This is a state outcome, not the action label.
        config: the weights.

    Returns:
        A float in ``[-config.clip, config.clip]``.

    The obstacle-aware block (``config.obstacle_*`` default 0.0 → inert) fires only at a
    flagged obstacle while grounded and not on a terminal win/death, so it sharpens the
    pipe decision without muddying the flag/death spikes:
      * committed a jump (``left_ground``)      → ``+obstacle_jump_bonus``
      * jammed/ran (``|dx| < stall_dx``)         → ``-obstacle_stall_penalty``
    """
    time_cost = config.time_penalty * (max(1, frames) / 8.0)
    progress = dx * (8.0 / max(1, frames)) if config.per_frame else dx
    r = config.progress_weight * (progress / config.progress_scale) - time_cost
    if cleared:
        r = config.flag_bonus if config.terminal_overrides else r + config.flag_bonus
    elif dead and terminal:
        r = -config.death_penalty if config.terminal_overrides else r - config.death_penalty
    elif obstacle_ahead and grounded:
        if left_ground:
            r += config.obstacle_jump_bonus
        elif abs(dx) < config.stall_dx:
            r -= config.obstacle_stall_penalty
    return max(-config.clip, min(config.clip, r))
