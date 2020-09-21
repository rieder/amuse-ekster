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
image_size_scale = 2
starscale = 1

phantom_solarm = 1.9891e33 | units.g
phantom_pc = 3.086e18 | units.cm
gas_rscale = 0.1 * phantom_pc
gas_mscale = 1.0 * phantom_solarm
star_rscale = 0.1 | units.parsec
star_mscale = 100 | units.MSun

timestep = 0.01 | units.Myr
timestep_bridge = 0.005 | units.Myr
epsilon_gas = 0.1 | units.parsec
epsilon_stars = 0.1 | units.parsec
h_acc = 0.25 | units.parsec

isothermal_gas_temperature = 20 | units.K

density_threshold = 1e-18 | units.g * units.cm**-3
# minimum_sink_radius = 0.25 | units.pc
minimum_sink_radius = 0.25 | units.pc
desired_sink_mass = 200 | units.MSun

alpha = 0.1
beta = 4.0
gamma = 5./3.
ieos = 1  # 1 = isothermal, 2 = adiabatic
icooling = 0  # 0 = disabled, 1 = h2cooling (if ichem=1) OR Gammie cooling (for disks), 2 = SD93 cooling, 4 = molecular clouds cooling

# Tide = None
Tide = TimeDependentSpiralArmsDiskModel
tide_spiral_type = "normal"
tide_time_offset = (5.0802 * 1.4874E+15 | units.s)
# tide_type = "strong"
# tide_time_offset = (2.2 * 1.4874E+15 | units.s)

# stop_after_each_step = True
stop_after_each_step = False

write_backups = True
use_wind = False
