"""Tests for dfa_sim.py — (ab)* and (aba)*c*ba DFAs on Assembly Calculus."""

import pytest
import dfa_sim


# ══════════════════════════════════════════════════════════════════════
#  (ab)* DFA
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def dfa():
    """Build and train the (ab)* DFA once for all tests in this module."""
    d = dfa_sim.build_ab_star_dfa(k=100, beta=0.05, p=0.05, seed=42)
    d.train(rounds=30)
    return d


class TestAbStarAccept:
    @pytest.mark.parametrize("s", ["", "ab", "abab", "ababab", "abababab"])
    def test_accept(self, dfa, s):
        accepted, final, trace = dfa.run(s)
        assert accepted, f"Expected ACCEPT for '{s}', got trace {trace}"
        assert final == "q0"


class TestAbStarReject:
    @pytest.mark.parametrize("s", [
        "a", "b", "ba", "aa", "bb",
        "aba", "abba", "ababb", "bab", "aab",
    ])
    def test_reject(self, dfa, s):
        accepted, final, trace = dfa.run(s)
        assert not accepted, f"Expected REJECT for '{s}', got trace {trace}"


class TestAbStarTransitions:
    def test_q0_a_gives_q1(self, dfa):
        dfa.reset()
        new, conf = dfa.step("a")
        assert new == "q1"
        assert conf > 0.5

    def test_q0_b_gives_trash(self, dfa):
        dfa.reset()
        new, _ = dfa.step("b")
        assert new == "q_trash"

    def test_q1_b_gives_q0(self, dfa):
        dfa.reset()
        dfa.step("a")
        new, _ = dfa.step("b")
        assert new == "q0"

    def test_q1_a_gives_trash(self, dfa):
        dfa.reset()
        dfa.step("a")
        new, _ = dfa.step("a")
        assert new == "q_trash"

    def test_trash_stays_trash_on_a(self, dfa):
        dfa.reset()
        dfa.step("b")
        new, _ = dfa.step("a")
        assert new == "q_trash"

    def test_trash_stays_trash_on_b(self, dfa):
        dfa.reset()
        dfa.step("b")
        new, _ = dfa.step("b")
        assert new == "q_trash"


class TestAbStarTrace:
    def test_ab_trace(self, dfa):
        _, _, trace = dfa.run("ab")
        assert trace == ["q0", "q1", "q0"]

    def test_abab_trace(self, dfa):
        _, _, trace = dfa.run("abab")
        assert trace == ["q0", "q1", "q0", "q1", "q0"]

    def test_empty_trace(self, dfa):
        _, _, trace = dfa.run("")
        assert trace == ["q0"]


class TestAbStarConfidence:
    def test_confidence_above_threshold(self, dfa):
        dfa.reset()
        _, conf = dfa.step("a")
        assert conf >= 0.4, f"Confidence too low: {conf}"


class TestAbStarEdgeCases:
    def test_unknown_symbol_raises(self, dfa):
        dfa.reset()
        with pytest.raises(ValueError, match="Unknown symbol"):
            dfa.step("c")

    def test_run_unknown_symbol_raises(self, dfa):
        with pytest.raises(ValueError):
            dfa.run("abc")

    def test_untrained_step_raises(self):
        d = dfa_sim.build_ab_star_dfa()
        with pytest.raises(RuntimeError, match="train"):
            d.step("a")

    def test_reset_returns_to_start(self, dfa):
        dfa.step("a")
        dfa.reset()
        assert dfa.current_state == "q0"


# ══════════════════════════════════════════════════════════════════════
#  (aba)*c*ba DFA
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def dfa2():
    """Build and train the (aba)*c*ba DFA once for this module."""
    d = dfa_sim.build_abacba_dfa(k=100, beta=0.05, p=0.05, seed=42)
    d.train(rounds=30)
    return d


class TestAbacbaAccept:
    @pytest.mark.parametrize("s", [
        "ba",           # n=0 m=0
        "cba",          # n=0 m=1
        "ccba",         # n=0 m=2
        "cccba",        # n=0 m=3
        "ababa",        # n=1 m=0
        "abacba",       # n=1 m=1
        "abaccba",      # n=1 m=2
        "abaababa",     # n=2 m=0
        "abaabacba",    # n=2 m=1
    ])
    def test_accept(self, dfa2, s):
        accepted, final, trace = dfa2.run(s)
        assert accepted, f"Expected ACCEPT for '{s}', got trace {trace}"
        assert final == "q5"


class TestAbacbaReject:
    @pytest.mark.parametrize("s", [
        "",             # empty
        "a", "b", "c",  # single chars
        "ab", "aba",    # incomplete
        "abc", "bba",
        "abba", "abaca",
        "cca", "baab",
        "abababac",     # trailing junk
        "cab",          # wrong order
    ])
    def test_reject(self, dfa2, s):
        accepted, final, trace = dfa2.run(s)
        assert not accepted, f"Expected REJECT for '{s}', got trace {trace}"


class TestAbacbaTransitions:
    """Spot-check individual transitions."""

    def test_q0_a(self, dfa2):
        dfa2.reset()
        new, _ = dfa2.step("a")
        assert new == "q1"

    def test_q0_b(self, dfa2):
        dfa2.reset()
        new, _ = dfa2.step("b")
        assert new == "q4"

    def test_q0_c(self, dfa2):
        dfa2.reset()
        new, _ = dfa2.step("c")
        assert new == "q3"

    def test_q1_b(self, dfa2):
        dfa2.reset()
        dfa2.step("a")  # q0 -> q1
        new, _ = dfa2.step("b")
        assert new == "q2"

    def test_q2_a_returns_to_q0(self, dfa2):
        dfa2.reset()
        dfa2.step("a")  # q0 -> q1
        dfa2.step("b")  # q1 -> q2
        new, _ = dfa2.step("a")
        assert new == "q0"

    def test_q3_c_stays(self, dfa2):
        dfa2.reset()
        dfa2.step("c")  # q0 -> q3
        new, _ = dfa2.step("c")
        assert new == "q3"

    def test_q3_b_goes_q4(self, dfa2):
        dfa2.reset()
        dfa2.step("c")  # q0 -> q3
        new, _ = dfa2.step("b")
        assert new == "q4"

    def test_q4_a_goes_q5(self, dfa2):
        dfa2.reset()
        dfa2.step("b")  # q0 -> q4
        new, _ = dfa2.step("a")
        assert new == "q5"


class TestAbacbaTrace:
    def test_ba_trace(self, dfa2):
        _, _, trace = dfa2.run("ba")
        assert trace == ["q0", "q4", "q5"]

    def test_abacba_trace(self, dfa2):
        _, _, trace = dfa2.run("abacba")
        assert trace == ["q0", "q1", "q2", "q0", "q3", "q4", "q5"]

    def test_abaababa_trace(self, dfa2):
        _, _, trace = dfa2.run("abaababa")
        assert trace == ["q0", "q1", "q2", "q0", "q1", "q2", "q0", "q4", "q5"]

