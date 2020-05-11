#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default settings for Ekster simulation run
"""
import numpy
from amuse.units import units
from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)


dpi = 200
L = 600
N = 800
image_size_scale = 1
starscale = 1

phantom_solarm = 1.9891e33 | units.g
phantom_pc = 3.086e18 | units.cm
gas_rscale = 0.1 * phantom_pc
gas_mscale = 1.0 * phantom_solarm
star_rscale = 0.1 | units.parsec
star_mscale = 1 | units.MSun

timestep = 0.01 | units.Myr
epsilon_gas = 0.1 | units.parsec
epsilon_stars = 0.1 | units.parsec

density_threshold = (
    (50 | units.MSun)  # ~ smoothed number of gas particles
    / (4/3 * numpy.pi * (0.1 | units.pc)**3)  # ~sphere with radius smoothing/softening length
)
minimum_sink_radius = 0.25 | units.pc
desired_sink_mass = 200 | units.MSun

alpha = 0.1
gamma = 1.
ieos = 2  # 1 = isothermal, 2 = adiabatic
icooling = 1  # 0 = disabled, 1 = h2cooling (if ichem=1) OR Gammie cooling (for disks), 2 = SD93 cooling, 4 = molecular clouds cooling

# Tide = None
Tide = TimeDependentSpiralArmsDiskModel
tide_spiral_type = "normal"
tide_time_offset = (5.0802 * 1.4874E+15 | units.s)
# tide_type = "strong"
# tide_time_offset = (2.2 * 1.4874E+15 | units.s)

stop_after_each_step = True
