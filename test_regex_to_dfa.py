"""Tests for regex_to_dfa.py — Brzozowski derivative regex→DFA conversion."""

import pytest
from regex_to_dfa import regex_to_dfa, parse, ParseError, EMPTY, EPSILON, Lit


# ── Parser tests ────────────────────────────────────────────────────────

class TestParser:
    def test_single_char(self):
        r = parse("a", {"a", "b"})
        assert isinstance(r, Lit) and r.char == "a"

    def test_concat(self):
        r = parse("ab", {"a", "b"})
        assert repr(r)  # just check it parses

    def test_alt(self):
        r = parse("a|b", {"a", "b"})
        assert r.nullable() is False

    def test_star(self):
        r = parse("a*", {"a"})
        assert r.nullable() is True

    def test_grouped_star(self):
        r = parse("(ab)*", {"a", "b"})
        assert r.nullable() is True

    def test_complex(self):
        r = parse("(a|b)*", {"a", "b"})
        assert r.nullable() is True

    def test_empty_pattern(self):
        r = parse("", {"a"})
        assert r.nullable() is True  # ε

    def test_char_not_in_alphabet(self):
        with pytest.raises(ParseError, match="not in alphabet"):
            parse("c", {"a", "b"})

    def test_unmatched_paren(self):
        with pytest.raises(ParseError):
            parse("(ab", {"a", "b"})

    def test_extra_paren(self):
        with pytest.raises(ParseError):
            parse("ab)", {"a", "b"})


# ── DFA construction tests ──────────────────────────────────────────────

class TestRegexToDFA:
    def test_ab_star(self):
        """(ab)* should produce a DFA equivalent to our hand-coded one."""
        states, alpha, start, accept, trans = regex_to_dfa("(ab)*", ["a", "b"])

        assert start in states
        assert start in accept  # ε is in (ab)*
        assert set(alpha) == {"a", "b"}

        # Verify by simulation
        def simulate(s):
            state = start
            for c in s:
                state = trans[(state, c)]
            return state in accept

        assert simulate("") is True
        assert simulate("ab") is True
        assert simulate("abab") is True
        assert simulate("a") is False
        assert simulate("b") is False
        assert simulate("ba") is False
        assert simulate("aba") is False

    def test_a_or_b(self):
        """a|b should accept 'a' and 'b', nothing else."""
        states, alpha, start, accept, trans = regex_to_dfa("a|b", ["a", "b"])

        def simulate(s):
            state = start
            for c in s:
                state = trans[(state, c)]
            return state in accept

        assert simulate("a") is True
        assert simulate("b") is True
        assert simulate("") is False
        assert simulate("ab") is False
        assert simulate("aa") is False

    def test_a_star(self):
        """a* should accept ε, a, aa, aaa, ..."""
        states, alpha, start, accept, trans = regex_to_dfa("a*", ["a", "b"])

        def simulate(s):
            state = start
            for c in s:
                state = trans[(state, c)]
            return state in accept

        assert simulate("") is True
        assert simulate("a") is True
        assert simulate("aaa") is True
        assert simulate("b") is False
        assert simulate("ab") is False

    def test_aba_star_c_star_ba(self):
        """(aba)*c*ba should work correctly."""
        states, alpha, start, accept, trans = regex_to_dfa(
            "(aba)*c*ba", ["a", "b", "c"])

        def simulate(s):
            state = start
            for c in s:
                state = trans[(state, c)]
            return state in accept

        # Accept
        assert simulate("ba") is True
        assert simulate("cba") is True
        assert simulate("ccba") is True
        assert simulate("ababa") is True
        assert simulate("abacba") is True
        assert simulate("abaababa") is True

        # Reject
        assert simulate("") is False
        assert simulate("a") is False
        assert simulate("aba") is False
        assert simulate("ab") is False
        assert simulate("abc") is False
        assert simulate("bba") is False

    def test_completeness(self):
        """Every (state, symbol) pair should have a transition."""
        states, alpha, start, accept, trans = regex_to_dfa("(ab)*", ["a", "b"])
        for s in states:
            for c in alpha:
                assert (s, c) in trans, f"Missing transition for ({s}, {c})"

    def test_trash_state_exists(self):
        """DFA for 'a' should have a trash state for 'b' transitions."""
        states, alpha, start, accept, trans = regex_to_dfa("a", ["a", "b"])
        assert "q_trash" in states

    def test_empty_alphabet_raises(self):
        with pytest.raises(ParseError, match="Alphabet must not be empty"):
            regex_to_dfa("a", [])

    def test_a_star_b_star(self):
        """a*b* — zero or more a's followed by zero or more b's."""
        states, alpha, start, accept, trans = regex_to_dfa("a*b*", ["a", "b"])

        def simulate(s):
            state = start
            for c in s:
                state = trans[(state, c)]
            return state in accept

        assert simulate("") is True
        assert simulate("a") is True
        assert simulate("b") is True
        assert simulate("aabb") is True
        assert simulate("aaab") is True
        assert simulate("ba") is False
        assert simulate("aba") is False

    def test_nested_parens(self):
        """((a|b))* — should work like (a|b)*."""
        states, alpha, start, accept, trans = regex_to_dfa("((a|b))*", ["a", "b"])

        def simulate(s):
            state = start
            for c in s:
                state = trans[(state, c)]
            return state in accept

        assert simulate("") is True
        assert simulate("a") is True
        assert simulate("b") is True
        assert simulate("abba") is True
        assert simulate("aabb") is True
