#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup / create codes
"""
from amuse.community.fastkick.interface import FastKick
from amuse.units import units
from ekster.bridge import CalculateFieldForCodes


def new_field_gravity_code(
        converter, epsilon_squared=0.1 | units.parsec**2,
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
    result.parameters.epsilon_squared = epsilon_squared
    print(result.parameters)
    return result


def new_field_code(
        code,
        # converter=None,
        # epsilon_squared=0. | units.m**2,
):
    " something"
    # result = CalculateFieldForCodesUsingReinitialize(
    result = CalculateFieldForCodes(
        new_field_gravity_code,
        [code],
        # converter=converter,
        # epsilon_squared=epsilon_squared,
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
