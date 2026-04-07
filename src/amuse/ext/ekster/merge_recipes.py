#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recipes for merging particles to form new or bigger stars
"""

from amuse.datamodel import Particle, Particles
from amuse.units import units


def merge_two_stars(primary, secondary):
    "Merge two colliding stars into one new one"
    colliders = Particles()
    colliders.add_particle(primary)
    colliders.add_particle(secondary)
    new_particle = Particle()
    setattr(new_particle, "mass", colliders.mass.sum())
    setattr(new_particle, "position", colliders.center_of_mass())
    setattr(new_particle, "velocity", colliders.center_of_mass_velocity())

    return new_particle


def form_new_star(
        densest_gas_particle,
        gas_particles,
        # density_threshold=
):
    """
    Forms a new star by merging all gas particles within a Jeans radius of the
    densest particle. Returns the new star and all particles used to make it.
    """
    new_star = Particle()
    gas_copy = gas_particles.copy()

    # densest_gas_particle = gas_copy.sorted_by_attribute("density")[0]
    jeans_radius = 0.025 | units.parsec  # 100 | units.AU
    gas_copy.position -= densest_gas_particle.position
    gas_copy.distance_to_core = gas_copy.position.lengths()
    dense_gas = gas_copy.select(
        lambda x:
        x < jeans_radius,
        ["distance_to_core"]
    )
    setattr(new_star, "mass", dense_gas.mass.sum())
    setattr(
        new_star, "position", (
            dense_gas.center_of_mass()
            + densest_gas_particle.position
        )
    )
    setattr(new_star, "velocity", dense_gas.center_of_mass_velocity())
    setattr(new_star, "radius", jeans_radius)
    absorbed_gas = dense_gas

    return new_star, absorbed_gas
