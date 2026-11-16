"""Text that reads differently on the terminal and in report.md: report.md keeps a move's noise
band, the terminal says the move without it."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

BANDS: ContextVar[bool] = ContextVar("bands", default=True)


class Said(str):
    """The report.md text of a sentence, with its terminal wording in ``plain``."""

    plain: str

    def __new__(cls, text: str, plain: str) -> Said:
        found = super().__new__(cls, text)
        found.plain = plain
        return found

    def __getnewargs__(self) -> tuple[str, str]:  # type: ignore[override]
        return str(self), self.plain


@contextmanager
def without_bands() -> Iterator[None]:
    token = BANDS.set(False)
    try:
        yield
    finally:
        BANDS.reset(token)


def said(make: Callable[[], str]) -> str:
    """The text ``make`` builds for report.md, carrying its terminal wording when that differs.
    ``make`` reads the wording it is asked for with ``banded`` and ``worded``."""
    with without_bands():
        plain = make()
    full = make()
    return full if plain == full else Said(full, plain)


def plain(text: str) -> str:
    """``text`` as the terminal shows it."""
    return text.plain if isinstance(text, Said) else text


def worded(text: str) -> str:
    """``text`` in the wording being built: its terminal wording inside ``without_bands``."""
    return text if BANDS.get() else plain(text)


def band_part(band: str) -> str:
    """`` ±2%`` for report.md, nothing for the terminal."""
    return f" ±{band}" if BANDS.get() else ""
