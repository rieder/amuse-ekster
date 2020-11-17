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
from amuse.community.petar.interface import Petar as star_code

### Parameters on grouped star formation
star_formation_method = "grouped"
group_distance = 1 | units.pc
group_speed_mach = 5
group_age = 0.1 | units.Myr


### Parameters on image
dpi = 200               # dots per inch 
L = 20                  # image width
N = 800                 # number of bins
image_size_scale = 2    
starscale = 100  #1

### Parameters on evolving codes
phantom_solarm = 1.9891e33 | units.g
phantom_pc = 3.086e18 | units.cm
gas_rscale = 0.1 * phantom_pc
gas_mscale = 1.0 * phantom_solarm
star_rscale = 0.1 | units.parsec
star_mscale = 100 | units.MSun

timestep = 0.01 | units.Myr
timestep_bridge = 0.005 | units.Myr
epsilon_gas = 0.1 | units.parsec
epsilon_stars = 0.05 | units.parsec

### Parameters on sink particles
density_threshold = 1e-18 | units.g * units.cm**-3
# For combined-sinks method, use smallest length i.e. epsilon_stars 
minimum_sink_radius = epsilon_stars      # 0.25 | units.pc
# h_acc should be the same as the sink radius
h_acc = minimum_sink_radius
desired_sink_mass = 200 | units.MSun

### Parameters on thermodynamics
isothermal_gas_temperature = 20 | units.K
gamma = 5./3.
ieos = 1  # 1 = isothermal, 2 = adiabatic
icooling = 0  # 0 = disabled, 1 = h2cooling (if ichem=1) OR Gammie cooling (for disks), 2 = SD93 cooling, 4 = molecular clouds cooling

### Parameters on artificial viscoscity
alpha = 0.1     # minimum artificial viscoscity parameter
beta = 4.0      # beta viscoscity

### Other parameters
Tide = None
#Tide = TimeDependentSpiralArmsDiskModel
#tide_spiral_type = "normal"
#tide_time_offset = (5.0802 * 1.4874E+15 | units.s)
# tide_type = "strong"
# tide_time_offset = (2.2 * 1.4874E+15 | units.s)

# stop_after_each_step = True
stop_after_each_step = False

write_backups = True
use_wind = False



