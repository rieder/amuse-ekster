"""
Default settings for Ekster simulation run
"""
from amuse.units import units

from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)


dpi = 200
L = 500
N = 600
image_size_scale = 2
starscale = 0.5

gas_rscale = 10 | units.parsec
gas_mscale = 1000 | units.MSun
star_rscale = 1 | units.parsec
star_mscale = 150 | units.MSun

timestep = 0.01 | units.Myr
epsilon_gas = 0.1 | units.parsec
epsilon_stars = 0.1 | units.parsec

density_threshold = 5e5 | units.amu * units.cm**-3
alpha = 0.1
gamma = 1.0
ieos = 1  # 1 = isothermal, 2 = adiabatic

# Tide = None
Tide = TimeDependentSpiralArmsDiskModel

stop_after_each_step = False
