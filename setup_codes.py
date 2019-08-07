"""
Setup / create codes
"""
from __future__ import print_function
from amuse.community.fastkick.interface import FastKick
from amuse.units import units
from bridge import CalculateFieldForCodes


def new_field_gravity_code(
        converter, epsilon=0. | units.m,
):
    "Create a new field code"
    print("Creating field code")
    result = FastKick(
        converter,
        redirection="none",
        # redirect_file=(
        #     p.dir_codelogs + "/field_gravity_code.log"
        #     ),
        mode="cpu",
        number_of_workers=8,
    )
    result.parameters.epsilon_squared = (epsilon)**2
    print(result.parameters)
    return result


def new_field_code(
        code,
        converter=None,
        epsilon=0. | units.m,
):
    " something"
    # result = CalculateFieldForCodesUsingReinitialize(
    result = CalculateFieldForCodes(
        new_field_gravity_code,
        [code],
        converter=converter,
        epsilon=epsilon,
        # required_attributes=[
        #     'mass', 'radius',
        #     'x', 'y', 'z',
        #     'vx', 'vy', 'vz',
        # ]
        # required_attributes=[
        #     'mass', 'u',
        #     'x', 'y', 'z',
        #     'vx', 'vy', 'vz',
        # ]
    )
    return result
