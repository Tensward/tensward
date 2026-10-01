"""The tool-argument checker: required keys and basic property types, nothing more."""

from __future__ import annotations

import pytest

from tensward.toolcalls import matches_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "ratio": {"type": "number"},
        "units": {"type": ["string", "null"]},
        "flag": {"type": "boolean"},
    },
    "required": ["city"],
}


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ({"city": "Paris"}, True),
        ({"city": "Paris", "days": 3, "ratio": 2, "units": None, "flag": False}, True),
        ({}, False),  # a required key is missing
        ({"city": 5}, False),
        ({"city": "Paris", "days": 2.5}, False),  # not an integer
        ({"city": "Paris", "days": True}, False),  # a bool is not an integer
        ({"city": "Paris", "ratio": "1"}, False),
        ({"city": "Paris", "units": 3}, False),  # not in the type list
        (["Paris"], False),
    ],
)
def test_matches_schema(arguments: object, expected: bool) -> None:
    assert matches_schema(arguments, SCHEMA) is expected
