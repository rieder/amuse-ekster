#!/usr/bin/env python3
# enc: utf-8
"""
Routines related to stellar interactions/collisions
"""
from amuse.datamodel import Particles
from amuse.units import units, constants
from amuse.community.smalln.interface import SmallN
from amuse.community.kepler.interface import Kepler
from amuse.couple import multiples


SMALLN = None


def init_smalln(converter):
    global SMALLN
    SMALLN = SmallN(convert_nbody=converter)


def new_smalln():
    SMALLN.reset()
    return SMALLN


def stop_smalln():
    global SMALLN
    SMALLN.stop()


def merge_two_stars(bodies, particles_in_encounter):
    """
    Merge two stars into one
    """
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    star_0 = particles_in_encounter[0]
    star_1 = particles_in_encounter[1]

    new_particle = Particles(1)
    new_particle.birth_age = particles_in_encounter.birth_age.min()
    new_particle.mass = particles_in_encounter.total_mass()
    new_particle.age = min(particles_in_encounter.age) \
        * max(particles_in_encounter.mass)/new_particle.mass
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.name = "Star"
    new_particle.radius = particles_in_encounter.radius.max()
    print("# old radius:", particles_in_encounter.radius.in_(units.AU))
    print("# new radius:", new_particle.radius.in_(units.AU))
    bodies.add_particles(new_particle)
    print(
        "# Two stars (M=",
        particles_in_encounter.mass.in_(units.MSun),
        ") collided at d=",
        (star_0.position - star_1.position).length().in_(units.AU)
    )
    bodies.remove_particles(particles_in_encounter)


def new_multiples_code(gravity, converter):
    """
    Initialise a multiples instance with specified gravity code.
    The gravity code must support stopping conditions (collision detection).
    """
    gravity.parameters.epsilon_squared = (0.0 | units.parsec)**2
    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()

    init_smalln(converter)
    kep = Kepler(unit_converter=converter)
    kep.initialize_code()
    multiples_code = multiples.Multiples(
        gravity,
        new_smalln,
        kep,
        constants.G,
    )
    multiples_code.neighbor_perturbation_limit = 0.05
    multiples_code.global_debug = 0
    return multiples_code
