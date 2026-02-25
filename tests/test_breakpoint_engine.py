"""Tests for viz/breakpoint_engine.py — covers all condition types,
breakpoint lifecycle, scanning, and serialization."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viz.breakpoint_engine import (
    BreakpointEngine, Breakpoint, BreakpointHit, ConditionType,
)


# ── Sample data ────────────────────────────────────────────────────────────

STEPS = [
    {
        "step": 0, "phase": "init", "dfa_state": "q0",
        "dfa_symbol": None, "dfa_transition": None,
        "activations": {
            "q0": {
                "winners": list(range(100)), "template_overlap": 1.0,
                "activation_margin": 10.0, "num_new_winners": 0,
                "winners_gained": list(range(100)), "winners_lost": [],
                "overlap_with_prev": 0.0, "template_name": "q0",
            },
        },
    },
    {
        "step": 1, "phase": "project", "dfa_state": "q1",
        "dfa_symbol": "a", "dfa_transition": "(q0, a) → q1",
        "activations": {
            "q0": {
                "winners": [], "template_overlap": 0.0,
                "activation_margin": 0, "num_new_winners": 0,
                "winners_gained": [], "winners_lost": list(range(100)),
                "overlap_with_prev": 0.0, "template_name": "q0",
            },
            "q1": {
                "winners": list(range(100)), "template_overlap": 1.0,
                "activation_margin": 5.0, "num_new_winners": 0,
                "winners_gained": list(range(100)), "winners_lost": [],
                "overlap_with_prev": 0.0, "template_name": "q1",
            },
            "Transitions": {
                "winners": list(range(100)), "template_overlap": 1.0,
                "activation_margin": 10.0, "num_new_winners": 0,
                "winners_gained": list(range(100)), "winners_lost": [],
                "overlap_with_prev": 0.0, "template_name": "trans_q0_a_q1",
            },
        },
    },
    {
        "step": 2, "phase": "project", "dfa_state": "q1",
        "dfa_symbol": "b", "dfa_transition": "(q1, b) → q1",
        "activations": {
            "q1": {
                "winners": list(range(80)) + list(range(200, 220)),
                "template_overlap": 0.72,
                "activation_margin": 1.5, "num_new_winners": 20,
                "winners_gained": list(range(200, 220)),
                "winners_lost": list(range(80, 100)),
                "overlap_with_prev": 0.8, "template_name": "q1",
            },
            "Transitions": {
                "winners": list(range(100, 200)),
                "template_overlap": 1.0,
                "activation_margin": 8.0, "num_new_winners": 0,
                "winners_gained": list(range(100, 200)),
                "winners_lost": list(range(100)),
                "overlap_with_prev": 0.0, "template_name": "trans_q1_b_q1",
            },
        },
    },
]


# ── Breakpoint lifecycle ───────────────────────────────────────────────────

class TestBreakpointLifecycle:
    def test_add_breakpoint(self):
        engine = BreakpointEngine()
        bp = engine.add_breakpoint(ConditionType.OVERLAP_BELOW, threshold=0.9)
        assert bp.id == "bp_0"
        assert bp.condition == ConditionType.OVERLAP_BELOW
        assert bp.threshold == 0.9
        assert bp.enabled is True

    def test_add_with_custom_id(self):
        engine = BreakpointEngine()
        bp = engine.add_breakpoint(
            ConditionType.MARGIN_BELOW, threshold=2.0, bp_id="my_bp"
        )
        assert bp.id == "my_bp"

    def test_auto_increment_id(self):
        engine = BreakpointEngine()
        bp1 = engine.add_breakpoint(ConditionType.OVERLAP_BELOW, 0.5)
        bp2 = engine.add_breakpoint(ConditionType.MARGIN_BELOW, 1.0)
        assert bp1.id == "bp_0"
        assert bp2.id == "bp_1"

    def test_remove_breakpoint(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, 0.5, bp_id="x")
        assert engine.remove_breakpoint("x") is True
        assert len(engine.breakpoints) == 0

    def test_remove_nonexistent(self):
        engine = BreakpointEngine()
        assert engine.remove_breakpoint("nope") is False

    def test_toggle_breakpoint(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, 0.5, bp_id="x")
        assert engine.toggle_breakpoint("x") is False  # was True → False
        assert engine.breakpoints["x"].enabled is False
        assert engine.toggle_breakpoint("x") is True   # back to True

    def test_clear_all(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, 0.5)
        engine.add_breakpoint(ConditionType.MARGIN_BELOW, 1.0)
        engine.clear_all()
        assert len(engine.breakpoints) == 0


# ── Condition evaluation ───────────────────────────────────────────────────

class TestOverlapBelow:
    def test_triggers_on_low_overlap(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, threshold=0.9)
        hits = engine.evaluate_step(STEPS[2])
        # q1 has overlap 0.72 < 0.9 → should trigger
        assert any(h.area == "q1" for h in hits)

    def test_no_trigger_on_high_overlap(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, threshold=0.5)
        hits = engine.evaluate_step(STEPS[2])
        # q1 has overlap 0.72 ≥ 0.5 → should not trigger
        q1_hits = [h for h in hits if h.area == "q1"]
        assert len(q1_hits) == 0

    def test_area_filter(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(
            ConditionType.OVERLAP_BELOW, threshold=0.9, area="Transitions"
        )
        hits = engine.evaluate_step(STEPS[2])
        # Transitions has overlap 1.0 ≥ 0.9 → no trigger
        assert len(hits) == 0


class TestMarginBelow:
    def test_triggers_on_low_margin(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.MARGIN_BELOW, threshold=3.0)
        hits = engine.evaluate_step(STEPS[2])
        # q1 has margin 1.5 < 3.0 → trigger
        assert any(h.area == "q1" for h in hits)

    def test_no_trigger_on_high_margin(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.MARGIN_BELOW, threshold=1.0)
        hits = engine.evaluate_step(STEPS[2])
        q1_hits = [h for h in hits if h.area == "q1"]
        assert len(q1_hits) == 0


class TestNewWinnersAbove:
    def test_triggers_on_many_new(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.NEW_WINNERS_ABOVE, threshold=10)
        hits = engine.evaluate_step(STEPS[2])
        # q1 has 20 new winners > 10 → trigger
        assert any(h.area == "q1" for h in hits)

    def test_no_trigger_on_few_new(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.NEW_WINNERS_ABOVE, threshold=50)
        hits = engine.evaluate_step(STEPS[2])
        q1_hits = [h for h in hits if h.area == "q1"]
        assert len(q1_hits) == 0


class TestOverlapPrevBelow:
    def test_triggers(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_PREV_BELOW, threshold=0.9)
        hits = engine.evaluate_step(STEPS[2])
        # q1 has overlap_with_prev 0.8 < 0.9 → trigger
        assert any(h.area == "q1" for h in hits)


class TestWinnersChanged:
    def test_triggers_on_any_change(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.WINNERS_CHANGED)
        hits = engine.evaluate_step(STEPS[2])
        # q1 has gained+lost → trigger
        assert any(h.area == "q1" for h in hits)

    def test_no_trigger_when_stable(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.WINNERS_CHANGED, area="q1")
        # Step 1 q1 has all gained (first activation) but step 0 had no q1
        # Use a custom step where winners didn't change
        stable_step = {
            "step": 99, "activations": {
                "q1": {"winners_gained": [], "winners_lost": []},
            }
        }
        hits = engine.evaluate_step(stable_step)
        assert len(hits) == 0


class TestStateChanged:
    def test_triggers_on_dfa_change(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.STATE_CHANGED)
        hits = engine.evaluate_step(STEPS[1], prev_step=STEPS[0])
        # q0 → q1
        assert len(hits) == 1
        assert "q0" in hits[0].message and "q1" in hits[0].message

    def test_no_trigger_same_state(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.STATE_CHANGED)
        hits = engine.evaluate_step(STEPS[2], prev_step=STEPS[1])
        # q1 → q1, no change
        assert len(hits) == 0


class TestDisabledBreakpoint:
    def test_disabled_not_evaluated(self):
        engine = BreakpointEngine()
        bp = engine.add_breakpoint(ConditionType.OVERLAP_BELOW, threshold=0.9)
        bp.enabled = False
        hits = engine.evaluate_step(STEPS[2])
        assert len(hits) == 0


# ── Scanning ───────────────────────────────────────────────────────────────

class TestScanning:
    def test_find_next_breakpoint(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, threshold=0.9)
        hit = engine.find_next_breakpoint(STEPS, from_step=0)
        # Step 1 has q0 dormant (overlap 0.0 < 0.9) → first hit
        assert hit is not None
        assert hit.step == 1
        assert hit.area == "q0"

    def test_find_next_from_middle(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, threshold=0.9)
        # Start from step 2, still triggers at step 2
        hit = engine.find_next_breakpoint(STEPS, from_step=2)
        assert hit is not None
        assert hit.step == 2

    def test_find_next_no_hit(self):
        engine = BreakpointEngine()
        # threshold 0.0 means overlap must be < 0.0, which cannot happen
        engine.add_breakpoint(ConditionType.MARGIN_BELOW, threshold=-1.0)
        hit = engine.find_next_breakpoint(STEPS)
        assert hit is None

    def test_find_all_breakpoints(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.STATE_CHANGED)
        hits = engine.find_all_breakpoints(STEPS)
        # Step 1: q0→q1
        assert len(hits) == 1
        assert hits[0].step == 1


# ── Serialization ──────────────────────────────────────────────────────────

class TestSerialization:
    def test_breakpoint_round_trip(self):
        bp = Breakpoint(
            id="test", condition=ConditionType.OVERLAP_BELOW,
            threshold=0.85, area="q1",
        )
        d = bp.to_dict()
        bp2 = Breakpoint.from_dict(d)
        assert bp2.id == "test"
        assert bp2.condition == ConditionType.OVERLAP_BELOW
        assert bp2.threshold == 0.85
        assert bp2.area == "q1"

    def test_summary(self):
        engine = BreakpointEngine()
        engine.add_breakpoint(ConditionType.OVERLAP_BELOW, 0.9)
        engine.add_breakpoint(ConditionType.MARGIN_BELOW, 2.0)
        summary = engine.get_breakpoint_summary()
        assert len(summary) == 2
        assert summary[0]["condition"] == "overlap_below"
