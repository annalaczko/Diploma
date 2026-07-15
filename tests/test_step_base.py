"""Tests for the abstract Step base class."""

import pytest

from pipeline.steps.base import Step


class ConcreteStep(Step):
    """Minimal concrete implementation for testing."""

    def run(self, data):
        return data


class DoubleStep(Step):
    def run(self, data):
        return data * 2


# --- instantiation ---

def test_default_name_uses_class_name():
    step = ConcreteStep()
    assert step.name == "ConcreteStep"


def test_custom_name_is_stored():
    step = ConcreteStep(name="MyStep")
    assert step.name == "MyStep"


def test_repr_contains_name():
    step = ConcreteStep(name="Foo")
    assert repr(step) == "<Foo>"


def test_repr_uses_class_name_when_no_name_given():
    step = ConcreteStep()
    assert repr(step) == "<ConcreteStep>"


# --- abstract enforcement ---

def test_cannot_instantiate_step_directly():
    with pytest.raises(TypeError):
        Step()  # type: ignore[abstract]


# --- __call__ delegates to run ---

def test_call_is_alias_for_run():
    step = DoubleStep()
    assert step(3) == step.run(3)


def test_run_passthrough():
    step = ConcreteStep()
    data = [1, 2, 3]
    assert step.run(data) is data
