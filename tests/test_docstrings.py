"""Tests that keep the Google-style docstrings of data classes complete.

``ruff`` (pydocstyle, Google convention) and ``pydoclint`` check docstring format and
function signatures. Class attributes need one extra check, done here: every field of
a dataclass must be described in the ``Attributes:`` section of its docstring.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import pytest

import football_vision


def dataclasses_of_package() -> list[type]:
    """Collects every dataclass defined in the package."""
    found: list[type] = []
    for info in pkgutil.walk_packages(football_vision.__path__, "football_vision."):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if dataclasses.is_dataclass(obj) and obj.__module__ == module.__name__:
                found.append(obj)
    return found


@pytest.mark.parametrize("cls", dataclasses_of_package(), ids=lambda c: c.__qualname__)
def test_every_dataclass_field_is_documented(cls: type) -> None:
    """Checks that all public fields appear in the Attributes section."""
    doc = inspect.getdoc(cls) or ""
    assert "Attributes:" in doc, f"{cls.__qualname__} has no Attributes section"
    attributes = doc.split("Attributes:", 1)[1]
    for field in dataclasses.fields(cls):
        if not field.name.startswith("_"):
            assert f"{field.name}:" in attributes, (
                f"{cls.__qualname__}.{field.name} is undocumented"
            )


def test_every_public_module_has_a_docstring() -> None:
    """Checks that each module starts with a docstring (Google style rule)."""
    for info in pkgutil.walk_packages(football_vision.__path__, "football_vision."):
        module = importlib.import_module(info.name)
        assert inspect.getdoc(module), f"{info.name} has no module docstring"
