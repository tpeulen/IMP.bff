"""Keep scalar GraphPort writes allocation-independent in behaviour."""

from IMP.bff import GraphPort


def test_scalar_write_preserves_float_and_integer_port_contracts():
    value = GraphPort(1.0)
    value.set_value(2.75)
    assert value.get_value() == 2.75

    integer = GraphPort(1)
    integer.set_value(2.75)
    assert integer.get_value_int() == 2
    assert integer.get_value() == 2.0


def test_scalar_link_write_updates_the_follower():
    source = GraphPort(1.0)
    follower = GraphPort(0.0)
    follower.set_link(source)
    source.set_value(4.5)
    assert follower.get_value() == 4.5
