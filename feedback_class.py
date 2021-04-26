#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback class
"""

from amuse.units import units
from amuse.units.trigo import sin, cos, arccos, arctan


def test_stellar_feedback(gas_, stars):
    # Extremely simplified stellar feedback onto gas
    gas = gas_.copy()
    mass_cutoff = 5 | units.MSun
    rmax = 0.05 | units.pc
    massive_stars = stars.select(
        lambda m:
        m >= mass_cutoff,
        ["mass"]
    )
    for i, star in enumerate(massive_stars):
        gas.position -= star.position
        gas.dist = gas.position.lengths()
        gas = gas.sorted_by_attribute("dist")
        gas_near_star = gas.select(
            lambda r:
            r < rmax,
            ["dist"]
        )
        theta = arccos(gas_near_star.z/gas_near_star.dist)
        phi = arctan(gas_near_star.y / gas_near_star.x)
        # Something related to energy
        # (Slow) solar wind is about 400 km/s
        # (1 | units.AU) / self.feedback_timestep
        base_velocity = (20 | units.kms)
        gas_near_star.vx += sin(theta) * cos(phi) * base_velocity
        gas_near_star.vy += sin(theta) * sin(phi) * base_velocity
        gas_near_star.vz += cos(theta) * base_velocity

        gas.position += star.position

    return gas
