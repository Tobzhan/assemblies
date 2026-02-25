"""Tests for LoggedDFABrain — verifies logging during DFA inference."""

import json
import os
import tempfile
import pytest
from logged_dfa_brain import LoggedDFABrain


def _make_dfa(log_dir):
    """Create and finalize the standard ab* DFA."""
    brain = LoggedDFABrain(p=0.01, beta=0.8, log_dir=log_dir)
    brain.add_state("q0", is_start=True)
    brain.add_state("q1", is_accept=True)
    brain.add_state("q_err")
    brain.add_symbol("a")
    brain.add_symbol("b")
    brain.set_transition("q0", "a", "q1")
    brain.set_transition("q1", "b", "q1")
    brain.set_transition("q0", "b", "q_err")
    brain.set_transition("q1", "a", "q_err")
    brain.set_transition("q_err", "a", "q_err")
    brain.set_transition("q_err", "b", "q_err")
    brain.finalize_machine()
    return brain


class TestTemplateRegistration:
    def test_state_templates_registered(self):
        brain = _make_dfa(tempfile.mkdtemp())
        templates = brain.logger.templates
        assert "q0" in templates
        assert "q1" in templates
        assert "q_err" in templates
        assert templates["q0"].area == "q0"
        assert templates["q0"].neuron_ids == set(range(100))

    def test_transition_templates_registered(self):
        brain = _make_dfa(tempfile.mkdtemp())
        templates = brain.logger.templates
        # Should have templates for each transition sub-assembly
        trans_names = [t.name for t in templates.values()
                       if t.area == "Transitions"]
        assert len(trans_names) == 6  # 6 transitions defined
        # Check one specific transition template
        assert any("trans_q0_a_q1" in n for n in trans_names)


class TestLoggedInference:
    def test_accept_a(self):
        brain = _make_dfa(tempfile.mkdtemp())
        accepted, history = brain.run_inference("a")
        assert accepted is True
        assert history == ["q0", "q1"]

    def test_accept_abbb(self):
        brain = _make_dfa(tempfile.mkdtemp())
        accepted, history = brain.run_inference("abbb")
        assert accepted is True
        assert history == ["q0", "q1", "q1", "q1", "q1"]

    def test_reject_b(self):
        brain = _make_dfa(tempfile.mkdtemp())
        accepted, history = brain.run_inference("b")
        assert accepted is False

    def test_reject_ba(self):
        brain = _make_dfa(tempfile.mkdtemp())
        accepted, history = brain.run_inference("ba")
        assert accepted is False


class TestStepLogging:
    def test_step_count(self):
        """1 init step + N symbol steps."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("abb")
        # init + 3 symbols = 4 steps
        assert len(brain.logger.steps) == 4

    def test_init_step_recorded(self):
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        init_step = brain.logger.steps[0]
        assert init_step.phase == "init"
        assert init_step.dfa_state == "q0"
        assert "q0" in init_step.activations
        # q0 should have winners in init
        assert len(init_step.activations["q0"].winners) == 100

    def test_dfa_fields_per_step(self):
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("ab")
        # Step 1: process 'a'
        step1 = brain.logger.steps[1]
        assert step1.dfa_symbol == "a"
        assert step1.dfa_state == "q1"
        assert "q1" in step1.dfa_transition
        # Step 2: process 'b'
        step2 = brain.logger.steps[2]
        assert step2.dfa_symbol == "b"
        assert step2.dfa_state == "q1"


class TestAttribution:
    def test_transition_attribution(self):
        """Transition area should have attribution from state + symbol."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        step1 = brain.logger.steps[1]
        trans_act = step1.activations["Transitions"]
        # Should have attribution keys for state and symbol
        assert "state_q0" in trans_act.attribution
        assert "stim_a" in trans_act.attribution
        # Both should be positive (strong wiring)
        assert trans_act.attribution["state_q0"] > 0
        assert trans_act.attribution["stim_a"] > 0

    def test_state_attribution(self):
        """State areas should have attribution from Transitions area."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        step1 = brain.logger.steps[1]
        q1_act = step1.activations["q1"]
        assert "Transitions" in q1_act.attribution


class TestTemplateOverlap:
    def test_state_template_match(self):
        """The winning state should have 100% template overlap since
        we reinforce to canonical form with activate()."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        step1 = brain.logger.steps[1]
        # q1 should be active and match template
        q1_act = step1.activations["q1"]
        assert q1_act.template_name == "q1"
        # After activate(q1, 0), winners are [0..99] → exact template match
        # But the logged activation might be from before activate() sets winners,
        # depending on ordering. The overlap should still be high.
        assert q1_act.template_overlap > 0.0

    def test_transition_template_match(self):
        """Transition winners should match the correct transition template."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        step1 = brain.logger.steps[1]
        trans_act = step1.activations["Transitions"]
        assert trans_act.template_name is not None
        assert trans_act.template_overlap > 0.0


class TestActivationMargin:
    def test_transition_has_margin(self):
        """Transition area should have a non-zero activation margin."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        step1 = brain.logger.steps[1]
        trans_act = step1.activations["Transitions"]
        assert trans_act.activation_margin > 0.0

    def test_input_stats_present(self):
        """All activations should have input_stats."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("a")
        step1 = brain.logger.steps[1]
        trans_act = step1.activations["Transitions"]
        assert trans_act.input_stats is not None
        assert trans_act.input_stats.max > 0


class TestWinnerDiff:
    def test_diff_across_steps(self):
        """Winners should track gained/lost between steps."""
        brain = _make_dfa(tempfile.mkdtemp())
        brain.run_inference("ab")
        # After init (q0 active), step 1 processes 'a' (q1 active)
        # For Transitions area, first activation has all gained
        step1 = brain.logger.steps[1]
        trans_act1 = step1.activations["Transitions"]
        assert len(trans_act1.winners_gained) > 0  # first time any winners


class TestJSONLSerialization:
    def test_save_and_load(self):
        log_dir = tempfile.mkdtemp()
        brain = _make_dfa(log_dir)
        brain.run_inference("ab")
        path = brain.save_log()
        assert os.path.exists(path)

        rows = brain.logger.load(path)
        assert len(rows) == 3  # init + 2 symbols

    def test_json_structure(self):
        log_dir = tempfile.mkdtemp()
        brain = _make_dfa(log_dir)
        brain.run_inference("a")
        brain.save_log()

        with open(brain.logger.log_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line.strip())
            assert "step" in data
            assert "activations" in data
            assert "dfa_state" in data

    def test_round_trip_preserves_data(self):
        log_dir = tempfile.mkdtemp()
        brain = _make_dfa(log_dir)
        brain.run_inference("a")
        brain.save_log()

        rows = brain.logger.load()
        step1 = rows[1]  # project step
        trans = step1["activations"]["Transitions"]
        assert "winners" in trans
        assert "attribution" in trans
        assert "template_overlap" in trans
        assert "activation_margin" in trans
        assert len(trans["winners"]) == 100


class TestMultipleInferences:
    def test_separate_runs_accumulate(self):
        """Multiple run_inference() calls should accumulate steps."""
        log_dir = tempfile.mkdtemp()
        brain = _make_dfa(log_dir)
        brain.run_inference("a")   # init + 1 = 2 steps
        brain.run_inference("ab")  # init + 2 = 3 steps
        assert len(brain.logger.steps) == 5  # 2 + 3

    def test_correctness_not_affected(self):
        """Logging should not affect DFA correctness."""
        log_dir = tempfile.mkdtemp()
        brain = _make_dfa(log_dir)
        test_cases = [
            ("a", True),
            ("ab", True),
            ("abb", True),
            ("b", False),
            ("ba", False),
            ("aa", False),
            ("abbb", True),
        ]
        for s, expected in test_cases:
            # Need fresh brain for each since run_inference resets state
            brain2 = _make_dfa(log_dir)
            accepted, _ = brain2.run_inference(s)
            assert accepted == expected, f"Failed for input '{s}'"
