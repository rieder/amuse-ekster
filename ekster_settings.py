#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default settings for Ekster simulation run.

DO NOT change settings in this file, instead write them to settings.ini in your
local run dir!
"""
from amuse.units import units

from amuse.units.quantities import (
    new_quantity, ScalarQuantity, VectorQuantity
)
import configparser


def find_unit(string):
    """
    parse a list to determine which AMUSE unit(s) it represents
    """
    if str(string) == string:
        # list consists of just one string, so just find the unit
        unit = getattr(units, string)
    else:
        # list has more than one component, which need to be multiplied to find
        # the unit
        unit = 1
        for component in string:
            if component == "*":
                # we're always multiplying so this can be ignored
                continue
            elif component == "/":
                # TODO needs some thought
                raise NotImplementedError(
                    "Can't parse division yet - "
                    "please use a negative power instead"
                )
            elif '**' in component:
                component, power = component.split('**')
                power = float(power)
            else:
                power = 1
            if hasattr(units, component):
                component_unit = getattr(units, component)**power
                # if the unit isn't found it is either a number or something we
                # don't recognise
            elif float(component) == component:
                component_unit = float(component)**power
            else:
                raise Exception(
                    "Can't parse as unit: %s" % component
                )
            unit *= component_unit
    return unit


def read_quantity(string):
    """
    convert a string to a quantity or vectorquantity

    the string must be formatted as '[1, 2, 3] unit' for a vectorquantity,
    or '1 unit' for a quantity.
    """

    if "]" in string:
        # It's a list, so convert it to a VectorQuantity.
        # The unit part comes after the list.
        # The list itself must consist of floats only!
        values = list(
            map(
                float,
                string[1:].split('] ')[0].split(',')
            )
        )
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
    try:
        for setting in config[setup]:
            if hasattr(settings, setting):
                setting_type = type(getattr(settings, setting))
                if setting_type == bool:
                    setattr(
                        settings, setting, config[setup].getboolean(setting)
                    )
                elif setting_type == int:
                    setattr(
                        settings, setting, config[setup].getint(setting)
                    )
                elif setting_type == float:
                    setattr(
                        settings, setting, config[setup].getfloat(setting)
                    )
                elif setting_type == (ScalarQuantity or VectorQuantity):
                    setattr(
                        settings, setting, read_quantity(
                            config[setup][setting]
                        )
                    )
                else:
                    setattr(settings, setting, config[setup][setting])
    except KeyError:
        raise Exception(
            "Error: no such setup '%s' in config file '%s'!"
            % (setup, filename)
        )
    return settings


def write_config(settings, filename, setup):
    config = configparser.ConfigParser()
    config[setup] = {}
    for setting in dir(settings):
        if setting[0] != '_':
            config[setup][setting] = str(getattr(settings, setting))
    with open(filename, 'w') as configfile:
        config.write(configfile)


class Settings:
    def __init__(self):
        self.rundir = "./"
        self.filename_stars = None
        self.filename_gas = None
        self.filename_sinks = None
        self.filename_random = None
        self.step = 0
        self.plot_dpi = 200
        self.plot_width = 10 | units.pc
        self.plot_bins = 800
        self.plot_image_size_scale = 2
        self.plot_starscale = 1
        self.plot_colorbar = False

        # phantom_solarm = 1.9891e30 | units.kg
        # phantom_pc = 3.086e16 | units.m
        self.gas_rscale = 3.086e15 | units.m
        self.gas_mscale = 1.9891e30 | units.kg
        self.star_rscale = 0.1 | units.parsec
        self.star_mscale = 100 | units.MSun

        self.timestep = 0.01 | units.Myr
        self.timestep_bridge = 0.005 | units.Myr
        self.epsilon_gas = 0.1 | units.parsec
        self.epsilon_stars = 0.1 | units.parsec

        self.isothermal_gas_temperature = 30 | units.K

        self.density_threshold = 1e-18 | units.g * units.cm**-3

        # For combined-sinks method, use smallest length i.e. epsilon_stars
        # for single-sinks method, should be bigger probably
        # definitely not smaller than epsilon_stars
        self.minimum_sink_radius = 0.25 | units.pc
        # h_acc should be the same as the sink radius
        self.h_acc = self.minimum_sink_radius
        # minimum_sink_radius = 0.25 | units.pc
        self.desired_sink_mass = 200 | units.MSun

        self.alpha = 0.1
        self.beta = 4.0
        self.gamma = 5./3.

        # 1 = isothermal, 2 = adiabatic
        self.ieos = 1

        # 0 = disabled, 1 = h2cooling (if ichem=1) OR Gammie cooling (for
        # disks), 2 = SD93 cooling, 4 = molecular clouds cooling
        self.icooling = 0

        self.Tide = None
        # self.Tide = "TimeDependentSpiralArmsDiskModel"
        # self.tide_spiral_type = "normal"
        # self.tide_time_offset = (5.0802 * 1.4874E+15 | units.s)
        # self.tide_spiral_type = "strong"
        # self.tide_time_offset = (2.2 * 1.4874E+15 | units.s)

        # stop_after_each_step = True
        self.stop_after_each_step = False

        self.write_backups = True
        self.use_wind = False
        # star_formation_method = "grouped"  # or "single"
        self.star_formation_method = "single"

        self.star_code = "Petar"

        self.begin_time = 0 | units.Myr
        self.model_time = 0 | units.Myr
