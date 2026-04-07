from amuse.units import units
from amuse.units.quantities import new_quantity
import configparser


def find_unit(string):
    if str(string) == string:
        # for simple units:
        unit = getattr(units, string)
    else:
        unit = 1
        # for compound units:
        for component in string:
            if '**' in component:
                component, power = component.split('**')
            else:
                power = 1
            component_unit = getattr(units, component)**power
            unit *= component_unit
    return unit


def read_quantity(string):
    """
    convert a string to a quantity or vectorquantity

    the string must be formatted as '[1, 2, 3] unit' for a vectorquantity,
    or '1 unit' for a quantity.
    """

    if "]" in string:
        values = list(map(float, string[1:].split('] ')[0].split(',')))
        unit = find_unit(string.split('] ')[1].split(' '))
        quantity = new_quantity(values, unit)
    else:
        value = float(string.split(' ')[0])
        unit = find_unit(string.split(' ')[1:])
        quantity = new_quantity(value, unit)
    return quantity


def read_config(settings, filename, setup):
    config = configparser.ConfigParser()
    config.read(filename)
    for setting in config[setup]:
        if hasattr(settings, setting):
            setattr(settings, setting, config[setup][setting])
