"""
Regex → DFA conversion using Brzozowski derivatives.

Supported syntax:
    a, b, c, ...   literal characters
    RS              concatenation (juxtaposition)
    R|S             alternation (union)
    R*              Kleene star (zero or more)
    (R)             grouping

The algorithm:
    1. Parse regex string → AST
    2. Starting from the initial regex, compute derivatives w.r.t.
       each alphabet symbol to discover new states
    3. Collect all reachable states → DFA
    4. Add a trash/sink state for completeness

Example:
    >>> states, alpha, start, accept, trans = regex_to_dfa("(ab)*", ['a','b'])
"""

from __future__ import annotations
from typing import FrozenSet, Tuple


# ══════════════════════════════════════════════════════════════════════
#  Regex AST Nodes (immutable, hashable for use as DFA states)
# ══════════════════════════════════════════════════════════════════════

class Regex:
    """Base class for regex AST nodes."""
    def nullable(self) -> bool:
        """Does this regex accept the empty string?"""
        raise NotImplementedError

    def derivative(self, c: str) -> "Regex":
        """Brzozowski derivative: what remains after consuming character c."""
        raise NotImplementedError


class Empty(Regex):
    """Matches nothing (∅)."""
    def nullable(self):
        return False

    def derivative(self, c):
        return EMPTY

    def __eq__(self, other):
        return isinstance(other, Empty)

    def __hash__(self):
        return hash("Empty")

    def __repr__(self):
        return "∅"


class Epsilon(Regex):
    """Matches only the empty string (ε)."""
    def nullable(self):
        return True

    def derivative(self, c):
        return EMPTY

    def __eq__(self, other):
        return isinstance(other, Epsilon)

    def __hash__(self):
        return hash("Epsilon")

    def __repr__(self):
        return "ε"


class Lit(Regex):
    """Matches a single literal character."""
    def __init__(self, char: str):
        self.char = char

    def nullable(self):
        return False

    def derivative(self, c):
        return EPSILON if c == self.char else EMPTY

    def __eq__(self, other):
        return isinstance(other, Lit) and self.char == other.char

    def __hash__(self):
        return hash(("Lit", self.char))

    def __repr__(self):
        return self.char


class Cat(Regex):
    """Concatenation: left followed by right."""
    def __init__(self, left: Regex, right: Regex):
        self.left = left
        self.right = right

    def nullable(self):
        return self.left.nullable() and self.right.nullable()

    def derivative(self, c):
        # Dc(LR) = Dc(L)·R  |  (if L nullable) Dc(R)
        d = make_cat(self.left.derivative(c), self.right)
        if self.left.nullable():
            d = make_alt(d, self.right.derivative(c))
        return d

    def __eq__(self, other):
        return (isinstance(other, Cat) and
                self.left == other.left and self.right == other.right)

    def __hash__(self):
        return hash(("Cat", self.left, self.right))

    def __repr__(self):
        return f"({self.left}{self.right})"


class Alt(Regex):
    """Alternation (union): matches left or right."""
    def __init__(self, parts: FrozenSet[Regex]):
        self.parts = parts  # frozen set for canonical ordering

    def nullable(self):
        return any(p.nullable() for p in self.parts)

    def derivative(self, c):
        return make_alt_set(frozenset(p.derivative(c) for p in self.parts))

    def __eq__(self, other):
        return isinstance(other, Alt) and self.parts == other.parts

    def __hash__(self):
        return hash(("Alt", self.parts))

    def __repr__(self):
        return "(" + "|".join(sorted(repr(p) for p in self.parts)) + ")"


class Star(Regex):
    """Kleene star: zero or more repetitions."""
    def __init__(self, inner: Regex):
        self.inner = inner

    def nullable(self):
        return True

    def derivative(self, c):
        # Dc(R*) = Dc(R) · R*
        return make_cat(self.inner.derivative(c), self)

    def __eq__(self, other):
        return isinstance(other, Star) and self.inner == other.inner

    def __hash__(self):
        return hash(("Star", self.inner))

    def __repr__(self):
        return f"({self.inner})*"


# Singleton constants
EMPTY = Empty()
EPSILON = Epsilon()


# ══════════════════════════════════════════════════════════════════════
#  Smart constructors (normalize on creation)
# ══════════════════════════════════════════════════════════════════════

def make_cat(left: Regex, right: Regex) -> Regex:
    """Create a normalized concatenation."""
    if left == EMPTY or right == EMPTY:
        return EMPTY
    if left == EPSILON:
        return right
    if right == EPSILON:
        return left
    return Cat(left, right)


def make_alt(left: Regex, right: Regex) -> Regex:
    """Create a normalized alternation from two regexes."""
    parts = set()
    _collect_alt_parts(left, parts)
    _collect_alt_parts(right, parts)
    return make_alt_set(frozenset(parts))


def make_alt_set(parts: FrozenSet[Regex]) -> Regex:
    """Create a normalized alternation from a set of regexes."""
    # Remove ∅
    parts = frozenset(p for p in parts if p != EMPTY)
    if not parts:
        return EMPTY
    if len(parts) == 1:
        return next(iter(parts))
    return Alt(parts)


def _collect_alt_parts(r: Regex, out: set):
    """Flatten nested Alt nodes into a set."""
    if isinstance(r, Alt):
        out.update(r.parts)
    elif r != EMPTY:
        out.add(r)


def make_star(inner: Regex) -> Regex:
    """Create a normalized Kleene star."""
    if isinstance(inner, Star):
        return inner  # (R*)* = R*
    if inner == EMPTY or inner == EPSILON:
        return EPSILON  # ∅* = ε, ε* = ε
    return Star(inner)


# ══════════════════════════════════════════════════════════════════════
#  Parser:  regex string → AST
# ══════════════════════════════════════════════════════════════════════

class ParseError(Exception):
    """Raised when the regex string is invalid."""
    pass


def parse(pattern: str, alphabet: set[str]) -> Regex:
    """Parse a regex string into an AST.

    Grammar:
        expr    → term ('|' term)*
        term    → factor factor*
        factor  → atom '*'*
        atom    → '(' expr ')' | literal
    """
    parser = _Parser(pattern, alphabet)
    result = parser.parse_expr()
    if parser.pos < len(parser.s):
        raise ParseError(
            f"Unexpected character '{parser.s[parser.pos]}' at position {parser.pos}")
    return result


class _Parser:
    def __init__(self, s: str, alphabet: set[str]):
        self.s = s
        self.pos = 0
        self.alphabet = alphabet

    def peek(self) -> str | None:
        if self.pos < len(self.s):
            return self.s[self.pos]
        return None

    def consume(self, expected: str = None) -> str:
        ch = self.peek()
        if ch is None:
            raise ParseError(f"Unexpected end of input")
        if expected and ch != expected:
            raise ParseError(
                f"Expected '{expected}' at position {self.pos}, got '{ch}'")
        self.pos += 1
        return ch

    def parse_expr(self) -> Regex:
        """expr → term ('|' term)*"""
        left = self.parse_term()
        while self.peek() == '|':
            self.consume('|')
            right = self.parse_term()
            left = make_alt(left, right)
        return left

    def parse_term(self) -> Regex:
        """term → factor factor*"""
        factors = []
        while self.peek() is not None and self.peek() not in ('|', ')'):
            factors.append(self.parse_factor())
        if not factors:
            return EPSILON  # empty term matches ε
        result = factors[0]
        for f in factors[1:]:
            result = make_cat(result, f)
        return result

    def parse_factor(self) -> Regex:
        """factor → atom '*'*"""
        atom = self.parse_atom()
        while self.peek() == '*':
            self.consume('*')
            atom = make_star(atom)
        return atom

    def parse_atom(self) -> Regex:
        """atom → '(' expr ')' | literal"""
        ch = self.peek()
        if ch == '(':
            self.consume('(')
            expr = self.parse_expr()
            self.consume(')')
            return expr
        elif ch is not None and ch not in ('|', ')', '*'):
            # Literal character
            self.consume()
            if ch not in self.alphabet:
                raise ParseError(
                    f"Character '{ch}' at position {self.pos - 1} "
                    f"is not in alphabet {sorted(self.alphabet)}")
            return Lit(ch)
        else:
            raise ParseError(
                f"Unexpected character '{ch}' at position {self.pos}")


# ══════════════════════════════════════════════════════════════════════
#  Brzozowski DFA construction
# ══════════════════════════════════════════════════════════════════════

def regex_to_dfa(
    pattern: str,
    alphabet: list[str],
) -> tuple[list[str], list[str], str, list[str], dict]:
    """Convert a regex pattern to a complete DFA specification.

    Args:
        pattern:  regex string, e.g. "(ab)*"
        alphabet: list of characters, e.g. ["a", "b"]

    Returns:
        (states, alphabet, start_state, accept_states, transitions)
        Ready to pass directly to DFABrain(...).

    Raises:
        ParseError: if the pattern is invalid or uses chars not in alphabet.
    """
    alpha_set = set(alphabet)
    if not alphabet:
        raise ParseError("Alphabet must not be empty")

    # Parse
    ast = parse(pattern, alpha_set)

    # BFS to discover all reachable derivative states
    state_map: dict[Regex, str] = {}  # regex → state name
    queue: list[Regex] = []
    transitions: dict[tuple[str, str], str] = {}
    accept_states: list[str] = []

    def get_or_create_state(r: Regex) -> str:
        if r in state_map:
            return state_map[r]
        name = f"q{len(state_map)}"
        state_map[r] = name
        queue.append(r)
        if r.nullable():
            accept_states.append(name)
        return name

    start_name = get_or_create_state(ast)

    while queue:
        current_regex = queue.pop(0)
        current_name = state_map[current_regex]
        for c in alphabet:
            derivative = current_regex.derivative(c)
            next_name = get_or_create_state(derivative)
            transitions[(current_name, c)] = next_name

    states = [state_map[r] for r in state_map]

    # Check if we need a trash state (any state that transitions to ∅)
    # The ∅ regex is already captured as a state if reachable.
    # Rename ∅ state to q_trash for clarity.
    empty_name = state_map.get(EMPTY)
    if empty_name is not None:
        # Rename to q_trash
        old_name = empty_name
        new_name = "q_trash"
        states = [new_name if s == old_name else s for s in states]
        accept_states = [new_name if s == old_name else s for s in accept_states]
        transitions = {
            (new_name if s == old_name else s, c):
            (new_name if t == old_name else t)
            for (s, c), t in transitions.items()
        }
        if start_name == old_name:
            start_name = new_name

    return states, list(alphabet), start_name, accept_states, transitions
