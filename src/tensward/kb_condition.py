"""The condition language of knowledge records: a small expression over facts.

Grammar (``not`` binds tighter than ``and``, ``and`` than ``or``)::

    expr    := or
    or      := and ("or" and)*
    and     := not ("and" not)*
    not     := "not" not | cmp
    cmp     := value (op value)?
    op      := "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not in" | "matches"
    value   := fact | literal | list | "known(" fact ")" | "(" expr ")"
    literal := number | "string" | true | false
    list    := "[" literal ("," literal)* "]"

Evaluation is three-valued (Kleene): a comparison that reads an unknown fact is unknown, and a
record fires only on true. ``matches`` tests a version fact against a PEP 440 specifier, and
``x in fact`` tests membership when the fact is a set.
"""

from __future__ import annotations

import functools
import json
import operator
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .facts import ARCH_BY_CC, UNKNOWN, VOCABULARY, Facts, FactSpec, canonical_json

Truth = bool | None  # None: unknown (Kleene)
Scalar = int | float | str | bool


@dataclass(frozen=True, slots=True)
class FactRef:
    name: str


@dataclass(frozen=True, slots=True)
class Lit:
    value: Scalar


@dataclass(frozen=True, slots=True)
class ListLit:
    values: tuple[Scalar, ...]


@dataclass(frozen=True, slots=True)
class Known:
    fact: str


@dataclass(frozen=True, slots=True)
class Compare:
    left: Node
    op: str
    right: Node


@dataclass(frozen=True, slots=True)
class Not:
    term: Node


@dataclass(frozen=True, slots=True)
class And:
    terms: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Or:
    terms: tuple[Node, ...]


Node = Or | And | Not | Compare | Known | FactRef | Lit | ListLit
OPS = ("==", "!=", "<", "<=", ">", ">=", "in", "not in", "matches")
_ORDER = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_ORDERED = tuple(_ORDER)
_NEGATED = {
    "==": "!=",
    "!=": "==",
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
    "in": "not in",
    "not in": "in",
    "matches": "not matches",
    "contains": "not contains",
    "not contains": "contains",
    "known": "unknown",
}


class ConditionError(ValueError):
    """A condition that does not parse; the message carries the column."""


@dataclass(frozen=True, slots=True)
class Term:
    """One comparison that decided a match, with the value the run had."""

    fact: str
    op: str
    value: Any
    observed: Any


@dataclass(frozen=True, slots=True)
class Explanation:
    fired_on: tuple[Term, ...]  # the terms that made the condition true
    unknown: tuple[str, ...]  # the facts whose absence left it unknown


_TOKEN = re.compile(
    r"\s*(?:(?P<number>-?\d+(?:\.\d+)?(?![\w.]))|(?P<string>\"[^\"]*\"|'[^']*')"
    r"|(?P<op>==|!=|<=|>=|<|>)|(?P<punct>[()\[\],])|(?P<word>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))"
)
_KEYWORDS = frozenset({"and", "or", "not", "in", "matches", "true", "false", "known"})


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # "number", "string", "op", "word", a keyword or a punctuation mark, or "end"
    text: str
    column: int


def _tokens(text: str) -> list[_Token]:
    found, at = [], 0
    while text[at:].strip():
        match = _TOKEN.match(text, at)
        if match is None or match.lastgroup is None:
            column = len(text) - len(text[at:].lstrip()) + 1
            raise ConditionError(f"column {column}: unexpected {text[column - 1]!r}")
        kind = match.lastgroup
        value = match[kind]
        if kind == "punct" or (kind == "word" and value in _KEYWORDS):
            kind = value
        found.append(_Token(kind, value, match.start(match.lastgroup) + 1))
        at = match.end()
    found.append(_Token("end", "", len(text) + 1))
    return found


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = _tokens(text)
        self.at = 0

    def peek(self, ahead: int = 0) -> _Token:
        return self.tokens[min(self.at + ahead, len(self.tokens) - 1)]

    def take(self, kind: str) -> _Token:
        token = self.peek()
        if token.kind != kind:
            raise ConditionError(
                f"column {token.column}: expected {kind}, found {token.text or 'the end'!r}"
            )
        self.at += 1
        return token

    def taking(self, kind: str) -> bool:
        if self.peek().kind == kind:
            self.at += 1
            return True
        return False

    def expression(self) -> Node:
        terms = [self.conjunction()]
        while self.taking("or"):
            terms.append(self.conjunction())
        return terms[0] if len(terms) == 1 else Or(tuple(terms))

    def conjunction(self) -> Node:
        terms = [self.negation()]
        while self.taking("and"):
            terms.append(self.negation())
        return terms[0] if len(terms) == 1 else And(tuple(terms))

    def negation(self) -> Node:
        if self.taking("not"):
            return Not(self.negation())
        return self.comparison()

    def comparison(self) -> Node:
        left = self.value()
        token = self.peek()
        if token.kind == "not" and self.peek(1).kind == "in":
            self.at += 2
            return Compare(left, "not in", self.value())
        if token.kind in ("op", "in", "matches"):
            self.at += 1
            return Compare(left, token.text, self.value())
        return left

    def value(self) -> Node:
        if self.taking("("):
            inner = self.expression()
            self.take(")")
            return inner
        if self.taking("known"):
            self.take("(")
            name = self.take("word").text
            self.take(")")
            return Known(name)
        if self.taking("["):
            values = [self.literal()]
            while self.taking(","):
                values.append(self.literal())
            self.take("]")
            return ListLit(tuple(values))
        if self.peek().kind == "word":
            return FactRef(self.take("word").text)
        return Lit(self.literal())

    def literal(self) -> Scalar:
        token = self.peek()
        if token.kind == "number":
            self.at += 1
            return float(token.text) if "." in token.text else int(token.text)
        if token.kind == "string":
            self.at += 1
            return token.text[1:-1]
        if token.kind in ("true", "false"):
            self.at += 1
            return token.kind == "true"
        found = token.text or "the end"
        raise ConditionError(f"column {token.column}: expected a value, found {found!r}")


def parse(text: str) -> Node:
    """The AST of ``text``; ConditionError names the column of the first problem."""
    parser = _Parser(text)
    node = parser.expression()
    end = parser.peek()
    if end.kind != "end":
        raise ConditionError(f"column {end.column}: unexpected {end.text!r}")
    return node


def check(node: Node, vocabulary: Mapping[str, FactSpec] = VOCABULARY) -> list[str]:
    """What is wrong with ``node`` as a condition: unknown facts, comparisons whose types or
    units do not match, enum literals outside their values. Empty when it is sound."""
    problems: list[str] = []
    _check_condition(node, vocabulary, problems)
    return problems


def _check_condition(node: Node, vocabulary: Mapping[str, FactSpec], out: list[str]) -> None:
    if isinstance(node, (And, Or)):
        for term in node.terms:
            _check_condition(term, vocabulary, out)
    elif isinstance(node, Not):
        _check_condition(node.term, vocabulary, out)
    elif isinstance(node, Known):
        _spec_of(node.fact, vocabulary, out)
    elif isinstance(node, Compare):
        _check_compare(node, vocabulary, out)
    elif isinstance(node, FactRef):
        spec = _spec_of(node.name, vocabulary, out)
        if spec is not None and spec.type != "bool":
            out.append(f"{node.name} is {spec.type}, not a condition")
    elif not (isinstance(node, Lit) and type(node.value) is bool):
        out.append(f"{_text(node)} is not a condition")


def _spec_of(name: str, vocabulary: Mapping[str, FactSpec], out: list[str]) -> FactSpec | None:
    spec = vocabulary.get(name)
    if spec is None:
        out.append(f"{name} is not in the fact vocabulary")
    return spec


def _check_compare(node: Compare, vocabulary: Mapping[str, FactSpec], out: list[str]) -> None:
    left, op, right = node.left, node.op, node.right
    for side in (left, right):
        if not isinstance(side, (FactRef, Lit, ListLit)):
            out.append(f"{_text(side)} is not a value to compare")
            return
    specs: list[FactSpec | None] = []
    for side in (left, right):
        spec = _spec_of(side.name, vocabulary, out) if isinstance(side, FactRef) else None
        if isinstance(side, FactRef) and spec is None:
            return
        specs.append(spec)
    lspec, rspec = specs
    if lspec is None and rspec is None:
        out.append(f"{_text(node)} compares no fact")
        return
    if isinstance(left, ListLit) or (isinstance(right, ListLit) and op not in ("in", "not in")):
        out.append(f"{_text(node)}: a list belongs only on the right of in or not in")
        return
    if op == "matches":
        _check_matches(node, lspec, out)
    elif op in ("in", "not in"):
        _check_member(node, lspec, rspec, out)
    else:
        _check_pair(node, lspec, rspec, out)


def _check_matches(node: Compare, spec: FactSpec | None, out: list[str]) -> None:
    if spec is None or spec.type != "version":
        out.append(f"{_text(node)}: matches needs a version fact on the left")
    elif not (isinstance(node.right, Lit) and isinstance(node.right.value, str)):
        out.append(f"{_text(node)}: matches needs a version specifier string")
    else:
        try:
            SpecifierSet(node.right.value)
        except InvalidSpecifier:
            out.append(f"{_text(node)}: {node.right.value!r} is not a version specifier")


def _check_member(
    node: Compare, lspec: FactSpec | None, rspec: FactSpec | None, out: list[str]
) -> None:
    left, right = node.left, node.right
    if isinstance(right, ListLit):
        if lspec is None or lspec.type in ("set", "version"):
            out.append(f"{_text(node)}: a list is tested against one scalar fact")
            return
        out.extend(f"{_text(node)}: {p}" for v in right.values if (p := _fits(lspec, v)))
    elif rspec is not None and rspec.type == "set":
        element = _element(rspec)
        if isinstance(left, Lit):
            if problem := _fits(element, left.value):
                out.append(f"{_text(node)}: {problem}")
        elif lspec is not None and (lspec.type, lspec.values) != (element.type, element.values):
            out.append(f"{_text(node)}: {lspec.name} is not an element of {rspec.name}")
    else:
        out.append(f"{_text(node)}: in needs a list or a set fact on the right")


def _check_pair(
    node: Compare, lspec: FactSpec | None, rspec: FactSpec | None, out: list[str]
) -> None:
    for spec in (lspec, rspec):
        if spec is not None and spec.type in ("set", "version"):
            out.append(f"{_text(node)}: {spec.name} is a {spec.type}; use in or matches")
            return
    if node.op in _ORDERED:
        for spec in (lspec, rspec):
            if spec is not None and spec.type not in ("int", "float"):
                out.append(f"{_text(node)}: {spec.name} is {spec.type}, not a number")
                return
    if lspec is not None and rspec is not None:
        numeric = {lspec.type, rspec.type} <= {"int", "float"}
        if not numeric and (lspec.type, lspec.values) != (rspec.type, rspec.values):
            out.append(f"{_text(node)}: {lspec.name} and {rspec.name} have different types")
        elif lspec.unit != rspec.unit:
            out.append(f"{_text(node)}: {lspec.unit} against {rspec.unit}")
        return
    spec = lspec or rspec
    literal = node.right if lspec is not None else node.left
    if spec is not None and isinstance(literal, Lit) and (problem := _fits(spec, literal.value)):
        out.append(f"{_text(node)}: {problem}")


def _element(spec: FactSpec) -> FactSpec:
    return FactSpec(
        name=spec.name,
        type="enum" if spec.values else "int",
        unit=spec.unit,
        values=spec.values,
        source=spec.source,
        description=spec.description,
    )


def _fits(spec: FactSpec, value: Scalar) -> str | None:
    """Why ``value`` cannot be a value of ``spec``; None when it can."""
    numeric = isinstance(value, (int, float)) and type(value) is not bool
    if spec.type == "int" and numeric and float(value).is_integer():
        return None
    if spec.type == "float" and numeric:
        return None
    if spec.type == "bool" and type(value) is bool:
        return None
    if spec.type == "str" and isinstance(value, str):
        return None
    if spec.type == "enum" and isinstance(value, str):
        if value in (spec.values or ()):
            return None
        return f"{value!r} is not one of {spec.name}'s values ({', '.join(spec.values or ())})"
    return f"{spec.name} is {spec.type}, not {value!r}"


def _text(node: Node) -> str:
    if isinstance(node, FactRef):
        return node.name
    if isinstance(node, Lit):
        return _literal(node.value)
    if isinstance(node, ListLit):
        return "[" + ", ".join(_literal(v) for v in node.values) + "]"
    if isinstance(node, Known):
        return f"known({node.fact})"
    if isinstance(node, Compare):
        return f"{_text(node.left)} {node.op} {_text(node.right)}"
    if isinstance(node, Not):
        return f"not {_text(node.term)}"
    joined = f" {'and' if isinstance(node, And) else 'or'} ".join(_text(t) for t in node.terms)
    return f"({joined})"


def _literal(value: Scalar) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    return f'"{value}"' if isinstance(value, str) else repr(value)


def _value(node: Node, facts: Facts) -> Any:
    if type(node) is FactRef:
        return facts[node.name]
    if type(node) is Lit:
        return node.value
    if type(node) is ListLit:
        return node.values
    return UNKNOWN


Memo = dict[Any, Any]  # one match's truths and terms of comparisons, by id(node)


def evaluate(node: Node, facts: Facts, memo: Memo | None = None) -> Truth:
    """True, False, or None when an unknown fact decides it. ``memo``, when given, keeps the
    truth of each comparison for later calls on the same ``facts``; its nodes must outlive it."""
    if type(node) is Compare:
        if memo is None:
            return _compare(_value(node.left, facts), node.op, _value(node.right, facts))
        key = id(node)
        if key in memo:
            seen: Truth = memo[key]
            return seen
        truth = memo[key] = _compare(_value(node.left, facts), node.op, _value(node.right, facts))
        return truth
    if type(node) is And or type(node) is Or:
        decides = type(node) is Or
        unknown = False
        for term in node.terms:
            truth = evaluate(term, facts, memo)
            if truth is decides:
                return decides
            unknown = unknown or truth is None
        return None if unknown else not decides
    if type(node) is Not:
        inner = evaluate(node.term, facts, memo)
        return None if inner is None else not inner
    if type(node) is Known:
        return facts[node.fact] is not UNKNOWN
    value = _value(node, facts)
    return value if type(value) is bool else None


@functools.lru_cache(maxsize=1024)
def _specifier(text: str) -> SpecifierSet:
    return SpecifierSet(text)


@functools.lru_cache(maxsize=1024)
def _version(text: str) -> Version:
    return Version(text)


def _compare(left: Any, op: str, right: Any) -> Truth:
    if left is UNKNOWN or right is UNKNOWN:
        return None
    try:
        if op == "==":
            return bool(left == right)
        if op == "in":
            return left in right
        if op == "matches":
            return _specifier(right).contains(_version(left), prereleases=True)
        if op == "!=":
            return not bool(left == right)
        if op == "not in":
            return left not in right
        return bool(_ORDER[op](left, right))
    except (InvalidSpecifier, InvalidVersion, TypeError):
        return None


def explain(
    node: Node, facts: Facts, truth: Truth | object = UNKNOWN, memo: Memo | None = None
) -> Explanation:
    """Why ``node`` is true on ``facts`` (the terms that decided it), or, when it is unknown,
    the facts that would decide it if Tensward knew them. ``truth``, when given, is
    ``evaluate(node, facts)``; ``memo`` is as for :func:`evaluate`."""
    if truth is UNKNOWN:
        truth = evaluate(node, facts, memo)
    if truth is True:
        return Explanation(tuple(_support(node, facts, True, memo)), ())
    if truth is None:
        return Explanation((), tuple(dict.fromkeys(_unknowns(node, facts, memo))))
    return Explanation((), ())


def _support(node: Node, facts: Facts, want: bool, memo: Memo | None) -> Iterator[Term]:
    if isinstance(node, (And, Or)):
        every = isinstance(node, And) == want
        for term in node.terms:
            if every:
                yield from _support(term, facts, want, memo)
            elif evaluate(term, facts, memo) is want:
                yield from _support(term, facts, want, memo)
                return
    elif isinstance(node, Not):
        yield from _support(node.term, facts, not want, memo)
    elif isinstance(node, Known):
        op = "known" if want else "unknown"
        yield Term(node.fact, op, None, _shown(facts[node.fact]))
    elif isinstance(node, FactRef):
        yield Term(node.name, "==", want, _shown(facts[node.name]))
    elif isinstance(node, Compare):
        if memo is None:
            yield _term(node, facts, want)
            return
        key = (id(node), want)
        found = memo.get(key)
        if found is None:
            found = memo[key] = _term(node, facts, want)
        yield found


def _term(node: Compare, facts: Facts, want: bool) -> Term:
    left, op, right = node.left, node.op, node.right
    if not isinstance(left, FactRef) and isinstance(right, FactRef):
        op = {"in": "contains", "not in": "not contains"}.get(op, op)
        left, right = right, left
    name = left.name if isinstance(left, FactRef) else _text(left)
    return Term(
        name,
        op if want else _NEGATED[op],
        _shown(_value(right, facts)),
        _shown(_value(left, facts)),
    )


def _shown(value: Any) -> Any:
    if value is UNKNOWN:
        return None
    if isinstance(value, frozenset):
        return sorted(value)
    return list(value) if isinstance(value, tuple) else value


def _unknowns(node: Node, facts: Facts, memo: Memo | None) -> Iterator[str]:
    if isinstance(node, (And, Or)):
        for term in node.terms:
            if evaluate(term, facts, memo) is None:
                yield from _unknowns(term, facts, memo)
    elif isinstance(node, Not):
        yield from _unknowns(node.term, facts, memo)
    elif isinstance(node, FactRef):
        yield node.name
    elif isinstance(node, Compare):
        names = [s.name for s in (node.left, node.right) if isinstance(s, FactRef)]
        yield from [n for n in names if facts[n] is UNKNOWN] or names


def to_json(node: Node) -> Any:
    """The AST as stored in the pack."""
    if isinstance(node, FactRef):
        return {"fact": node.name}
    if isinstance(node, Lit):
        return {"lit": node.value}
    if isinstance(node, ListLit):
        return {"list": list(node.values)}
    if isinstance(node, Known):
        return {"known": node.fact}
    if isinstance(node, Compare):
        return {"cmp": [to_json(node.left), node.op, to_json(node.right)]}
    if isinstance(node, Not):
        return {"not": to_json(node.term)}
    return {"and" if isinstance(node, And) else "or": [to_json(t) for t in node.terms]}


def from_json(data: Any) -> Node:
    if not isinstance(data, dict) or len(data) != 1:
        raise ConditionError(f"not a condition node: {data!r}")
    ((kind, body),) = data.items()
    if kind == "fact" and isinstance(body, str):
        return FactRef(body)
    if kind == "known" and isinstance(body, str):
        return Known(body)
    if kind == "lit" and _scalar(body):
        return Lit(body)
    if kind == "list" and isinstance(body, list) and body and all(map(_scalar, body)):
        return ListLit(tuple(body))
    if kind == "cmp" and isinstance(body, list) and len(body) == 3 and body[1] in OPS:
        return Compare(from_json(body[0]), body[1], from_json(body[2]))
    if kind == "not":
        return Not(from_json(body))
    if kind in ("and", "or") and isinstance(body, list) and len(body) > 1:
        terms = tuple(from_json(term) for term in body)
        return And(terms) if kind == "and" else Or(terms)
    raise ConditionError(f"not a condition node: {data!r}")


def to_refs(node: Node, terms: dict[str, int]) -> list[int | str]:
    """The AST in prefix order, each comparison, ``known`` or bare value by its number in
    ``terms`` (canonical JSON -> number, added to as new terms are met), ``and`` and ``or`` as
    "&n" and "|n" over their n terms, ``not`` as "!": a pack's form."""
    if isinstance(node, (And, Or)):
        found: list[int | str] = [f"{'&' if isinstance(node, And) else '|'}{len(node.terms)}"]
        for term in node.terms:
            found += to_refs(term, terms)
        return found
    if isinstance(node, Not):
        return ["!", *to_refs(node.term, terms)]
    return [terms.setdefault(canonical_json(to_json(node)).decode(), len(terms) + 1)]


def from_refs(form: Sequence[int | str], terms: Mapping[int, Node]) -> Node:
    """The AST of a :func:`to_refs` form, its terms looked up in ``terms``."""
    return _from_refs(form, 0, terms)[0]


def _from_refs(form: Sequence[int | str], at: int, terms: Mapping[int, Node]) -> tuple[Node, int]:
    token = form[at]
    if type(token) is int:
        return terms[token], at + 1
    if token == "!":
        inner, at = _from_refs(form, at + 1, terms)
        return Not(inner), at
    parts = []
    at += 1
    for _ in range(int(str(token)[1:])):
        part, at = _from_refs(form, at, terms)
        parts.append(part)
    return (And if str(token)[0] == "&" else Or)(tuple(parts)), at


def terms_from_json(rows: Iterable[tuple[int, str]]) -> dict[int, Node]:
    """Term number -> node, from a pack's ``terms`` rows."""
    return {number: from_json(json.loads(text)) for number, text in rows}


def _scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str)) and value == value


ANCHOR_KEYS = ("engine", "family", "arch", "diag", "goal")
_ANCHORED = {"model.family": "family", "gpu.arch": "arch", "diag.primary": "diag", "goal": "goal"}
ANY = ("*",)


def anchors(node: Node | None, *, engine: str, gpu_cc: str | None) -> dict[str, tuple[str, ...]]:
    """Per key, the values a top-level ``and`` term pins; a key with no such term maps to
    ("*",). Only top-level ``and`` terms count; anything under ``or``/``not`` is "*"."""
    pinned: dict[str, tuple[str, ...]] = {"engine": (engine,)}
    terms = node.terms if isinstance(node, And) else () if node is None else (node,)
    found = [pin for term in terms if (pin := _pin(term)) is not None]
    if gpu_cc is not None:
        found.append(("arch", _arches_in_range(gpu_cc)))
    for key, values in found:
        if key not in pinned:
            pinned[key] = values
        elif key != "diag":  # a run has one family, arch and goal, but many fired classes
            pinned[key] = tuple(v for v in pinned[key] if v in values)
    return {key: pinned.get(key, ANY) for key in ANCHOR_KEYS}


def _pin(term: Node) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(term, Compare):
        return None
    left, op, right = term.left, term.op, term.right
    if isinstance(left, Lit) and op == "in" and right == FactRef("diag.fired"):
        return "diag", (str(left.value),)
    if not isinstance(left, FactRef):
        return None
    if left.name == "gpu.cc":
        return "arch", _arches(op, right)
    key = _ANCHORED.get(left.name)
    if key is None:
        return None
    if op == "==" and isinstance(right, Lit):
        values: tuple[Scalar, ...] = (right.value,)
    elif op == "in" and isinstance(right, ListLit):
        values = right.values
    else:
        return None
    spec = VOCABULARY[left.name]
    order = spec.values or tuple(sorted(str(v) for v in values))
    return key, tuple(v for v in order if v in values)


def _arches(op: str, right: Node) -> tuple[str, ...]:
    """The architectures with some compute capability for which ``gpu.cc <op> right`` holds."""
    if op == "in" and isinstance(right, ListLit):
        bounds = [("==", float(v)) for v in right.values if not isinstance(v, (str, bool))]
        return tuple(a for _, _, a in ARCH_BY_CC if any(_reach(a, o, v) for o, v in bounds))
    if op in ("==", *_ORDERED) and isinstance(right, Lit) and isinstance(right.value, (int, float)):
        return tuple(a for _, _, a in ARCH_BY_CC if _reach(a, op, float(right.value)))
    return tuple(a for _, _, a in ARCH_BY_CC)


def _reach(arch: str, op: str, value: float) -> bool:
    low, high = next((lo, hi) for lo, hi, name in ARCH_BY_CC if name == arch)
    if op in (">", ">="):
        return high > value
    if op == "<":
        return low < value
    if op == "<=":
        return low <= value
    return low <= value < high


def _arches_in_range(gpu_cc: str) -> tuple[str, ...]:
    """The architectures a ``validity.gpu_cc`` specifier (">=7.5", ">=8.0,<9.0") reaches."""
    every = tuple(a for _, _, a in ARCH_BY_CC)
    try:
        specifiers = list(SpecifierSet(gpu_cc))
    except InvalidSpecifier:
        return every
    reached = set(every)
    for specifier in specifiers:
        try:
            value = float(specifier.version)
        except ValueError:
            continue
        if specifier.operator in ("==", *_ORDERED):
            reached &= {a for a in every if _reach(a, specifier.operator, value)}
    return tuple(a for a in every if a in reached)


def facts_in(node: Node) -> set[str]:
    """Every fact a condition reads."""
    if isinstance(node, (And, Or)):
        return set().union(*(facts_in(term) for term in node.terms))
    if isinstance(node, Not):
        return facts_in(node.term)
    if isinstance(node, Known):
        return {node.fact}
    if isinstance(node, Compare):
        return facts_in(node.left) | facts_in(node.right)
    return {node.name} if isinstance(node, FactRef) else set()
