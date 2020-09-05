#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster module for star formation from sinks

This module omits multiple stars for simplicity
"""
import logging
import numpy

from amuse.units import units  # , constants, nbody_system
from amuse.datamodel import Particle, Particles
# from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.units.trigo import sin, cos
from amuse.ext.masc import new_star_cluster
from amuse.ext.sink import SinkParticles

from sinks_class import should_a_sink_form
import default_settings


def new_kroupa_mass_distribution(
        number_of_particles,
        mass_min=None,
        mass_max=None,
        random=True,
):
    """
    Returns Kroupa (2001) mass distribution
    Modified from amuse.ic.brokenimf version - discard masses below mass_min
    """
    masses = [] | units.MSun
    while len(masses) < number_of_particles:
        next_mass = MultiplePartIMF(
            mass_boundaries=[0.01, 0.08, 0.5, 1000.0] | units.MSun,
            mass_max=mass_max,
            alphas=[-0.3, -1.3, -2.3],
            random=random,
        ).next_mass(1)
        if mass_min is None:
            masses.append(next_mass)
        elif next_mass >= mass_min:
            masses.append(next_mass)
    return masses


def generate_next_mass(
        initial_mass_function="kroupa",
        upper_mass_limit=100 | units.MSun,
        binary_fraction=0,
        triple_fraction=0,
):
    "Generate list of masses of next star/stars to form"
    rnd = numpy.random.random()
    is_triple = False
    is_binary = False
    if rnd < triple_fraction:
        is_triple = True
    elif rnd < triple_fraction + binary_fraction:
        is_binary = True
    if not (is_binary or is_triple):
        number_of_stars = 1
    elif is_binary:
        number_of_stars = 2
    elif is_triple:
        number_of_stars = 3

    if initial_mass_function == "kroupa":
        return new_kroupa_mass_distribution(
            number_of_stars,
            mass_max=upper_mass_limit,
        )
    return False


def form_stars(
        sink,
        upper_mass_limit=100 | units.MSun,
        local_sound_speed=0.2 | units.kms,
        logger=None,
        randomseed=None,
        **keyword_arguments
):
    """
    Let a sink form stars.
    """
    logger = logger or logging.getLogger(__name__)
    if randomseed is not None:
        logger.info("setting random seed to %i", randomseed)
        numpy.random.seed(randomseed)

    # sink_initial_density = sink.mass / (4/3 * numpy.pi * sink.radius**3)

    initialised = sink.initialised or False
    if not initialised:
        logger.debug("Initialising sink %i for star formation", sink.key)
        next_mass = generate_next_mass()
        # sink.next_number_of_stars = len(next_mass)
        # sink.next_total_mass = next_mass.sum()
        sink.next_primary_mass = next_mass[0]
        # if sink.next_number_of_stars > 1:
        #     sink.next_secondary_mass = next_mass[1]
        # if sink.next_number_of_stars > 2:
        #     sink.next_tertiary_mass = next_mass[2]
        sink.initialised = True
    if sink.mass < sink.next_primary_mass:
        logger.debug(
            "Sink %i is not massive enough for the next star", sink.key
        )
        return Particles()

    # We now have the first star that will be formed.
    # Next, we generate a list of stellar masses, so that the last star in the
    # list is just one too many for the sink's mass.

    mass_left = sink.mass - sink.next_primary_mass
    masses = new_star_cluster(
        stellar_mass=mass_left,
        upper_mass_limit=upper_mass_limit,
    ).mass
    number_of_stars = len(masses)

    new_stars = Particles(number_of_stars)
    new_stars.age = 0 | units.Myr
    new_stars[0].mass = sink.next_primary_mass
    new_stars[1:].mass = masses[:-1]
    sink.next_primary_mass = masses[-1]
    # if sink.next_number_of_stars > 1:
    #     new_stars[1].mass = sink.next_secondary_mass
    # if sink.next_number_of_stars > 2:
    #     new_stars[2].mass = sink.next_tertiary_mass
    new_stars.position = sink.position
    new_stars.velocity = sink.velocity

    # Random position within the sink radius
    radius = sink.radius
    rho = numpy.random.random(number_of_stars) * radius
    theta = (
        numpy.random.random(number_of_stars)
        * (2 * numpy.pi | units.rad)
    )
    phi = (
        numpy.random.random(number_of_stars) * numpy.pi | units.rad
    )
    x = rho * sin(phi) * cos(theta)
    y = rho * sin(phi) * sin(theta)
    z = rho * cos(phi)
    new_stars.x += x
    new_stars.y += y
    new_stars.z += z
    # Random velocity, sample magnitude from gaussian with local sound speed
    # like Wall et al (2019)
    # temperature = 10 | units.K
    try:
        local_sound_speed = sink.u.sqrt()
    except AttributeError:
        local_sound_speed = local_sound_speed
    # or (gamma * local_pressure / density).sqrt()
    velocity_magnitude = numpy.random.normal(
        # loc=0.0,  # <- since we already added the velocity of the sink
        scale=local_sound_speed.value_in(units.kms),
        size=number_of_stars,
    ) | units.kms
    velocity_theta = (
        numpy.random.random(number_of_stars)
        * (2 * numpy.pi | units.rad)
    )
    velocity_phi = (
        numpy.random.random(number_of_stars)
        * (numpy.pi | units.rad)
    )
    vx = velocity_magnitude * sin(velocity_phi) * cos(velocity_theta)
    vy = velocity_magnitude * sin(velocity_phi) * sin(velocity_theta)
    vz = velocity_magnitude * cos(velocity_phi)
    new_stars.vx += vx
    new_stars.vy += vy
    new_stars.vz += vz

    new_stars.origin_cloud = sink.key
    # For Pentacle, this is the PP radius
    new_stars.radius = 0.05 | units.parsec
    sink.mass -= new_stars.total_mass()
    # TODO: fix sink's momentum etc

    # EDIT: Do not shrink the sinks at this point, but rather when finished
    # forming stars.
    # # Shrink the sink's (accretion) radius to prevent it from accreting
    # # relatively far away gas and moving a lot
    # sink.radius = (
    #     (sink.mass / sink_initial_density)
    #     / (4/3 * numpy.pi)
    # )**(1/3)

    # cleanup
    # sink.initialised = False
    return new_stars


class StarFormingRegion(
        Particle,
):
    """
    Creates a StarFormingRegion Particle superclass.
    """

    def __init__(
            self,
            key=None,
            particles_set=None,
            set_index=None,
            set_version=-1,
            mass=0 | units.MSun,
            radius=0 | units.AU,
            position=[0, 0, 0] | units.parsec,
            velocity=[0, 0, 0] | units.kms,
            initial_mass_function="kroupa",
            binary_fraction=0,
            triple_fraction=0,
            upper_mass_limit=100 | units.MSun,
            **keyword_arguments):
        if particles_set is None:
            if key is None:
                particles_set = Particles(1)
                key = particles_set.get_all_keys_in_store()[0]
            else:
                particles_set = Particles(1, keys=[key])

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "particles_set", particles_set)
        object.__setattr__(self, "_set_index", set_index)
        object.__setattr__(self, "_set_version", set_version)

        for attribute_name in keyword_arguments:
            attribute_value = keyword_arguments[attribute_name]
            setattr(self, attribute_name, attribute_value)
        self.triple_fraction = triple_fraction
        self.mass = mass
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.initial_mass_function = initial_mass_function
        self.binary_fraction = binary_fraction
        self.upper_mass_limit = upper_mass_limit
        self.generate_next_mass()

    def generate_next_mass(self):
        """
        Generate the next (set of) stellar mass(es) that will be formed in this
        region.
        Assumes that 'radius' will be the 'orbit' that a newly formed star will
        be on around the centre of mass of this region.
        """
        rnd = numpy.random.random()
        is_triple = False
        is_binary = False
        if rnd < self.triple_fraction:
            is_triple = True
        elif rnd < self.triple_fraction + self.binary_fraction:
            is_binary = True

        if not (is_binary or is_triple):
            number_of_stars = 1
        elif is_binary:
            number_of_stars = 2
        elif is_triple:
            number_of_stars = 3

        self.next_mass = new_kroupa_mass_distribution(
            number_of_stars,
            mass_max=self.upper_mass_limit,
        )

    def yield_next(self):
        """
        Determine if (a) new star(s) can now be formed, and if so, return these
        """
        if self.mass >= self.next_mass.sum():
            number_of_stars = len(self.next_mass)
            # converter = nbody_system.nbody_to_si(
            #     self.mass,
            #     self.radius,
            # )
            # new_stars = new_plummer_distribution(
            #     converter, len(self.next_mass))
            new_stars = Particles(number_of_stars)
            new_stars.mass = self.next_mass
            new_stars.position = self.position
            new_stars.velocity = self.velocity

            # Random position within the sink radius
            radius = self.radius
            rho = numpy.random.random(number_of_stars) * radius
            theta = (
                numpy.random.random(number_of_stars) * 2 * numpy.pi | units.rad
            )
            phi = numpy.random.random(number_of_stars) * numpy.pi | units.rad
            x = rho * sin(phi) * cos(theta)
            y = rho * sin(phi) * sin(theta)
            z = rho * cos(phi)
            new_stars.x += x
            new_stars.y += y
            new_stars.z += z

            # Random velocity, sample magnitude from gaussian with local sound
            # speed like Wall et al (2019)
            local_sound_speed = 2 | units.kms
            # TODO: do this properly - see e.g. formula 5.17 in AMUSE book
            velocity_magnitude = numpy.random.normal(
                # loc=0.0,  # <- since we already added the velocity of the
                # sink
                scale=local_sound_speed.value_in(units.kms),
                size=number_of_stars,
            ) | units.kms
            velocity_theta = (
                numpy.random.random(number_of_stars) * 2 * numpy.pi | units.rad
            )
            velocity_phi = (
                numpy.random.random(number_of_stars) * numpy.pi | units.rad
            )
            vx = velocity_magnitude * sin(velocity_phi) * cos(velocity_theta)
            vy = velocity_magnitude * sin(velocity_phi) * sin(velocity_theta)
            vz = velocity_magnitude * cos(velocity_phi)
            new_stars.vx += vx
            new_stars.vy += vy
            new_stars.vz += vz

            new_stars.origin_cloud = self.key

            # Make sure quantities are (mostly) conserved
            # - mass
            self.mass -= new_stars.total_mass()
            # - momentum

            # Determine which star(s) should form next
            self.generate_next_mass()
            return new_stars
        return Particles()


def form_sinks(
        gas, sinks, critical_density,
        logger=None,
        minimum_sink_radius=default_settings.minimum_sink_radius,
        desired_sink_mass=default_settings.desired_sink_mass,
):
    """
    Determines where sinks should form from gas.
    Non-destructive function that creates new sinks and adds them to sinks.
    Accretes gas onto sinks.
    Returns new sinks and accreted particles.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    sink_cores = Particles()

    high_density_gas = gas.select_array(
        lambda density:
        density > critical_density,
        ["density"],
    ).sorted_by_attribute("density").reversed()
    print(
        "Number of gas particles above critical density (%s): %i" % (
            critical_density.in_(units.g * units.cm**-3),
            len(high_density_gas),
        )
    )

    high_density_gas.form_sink = False
    high_density_gas[
        high_density_gas.density/critical_density > 10
        ].form_sink = True
    high_density_gas[high_density_gas.form_sink is False].form_sink = \
        should_a_sink_form(
            high_density_gas[high_density_gas.form_sink is False], gas
        )
    sink_cores = high_density_gas[high_density_gas.form_sink]
    del sink_cores.form_sink
    desired_sink_radius = (desired_sink_mass / sink_cores.density)**(1/3)
    desired_sink_radius[
        desired_sink_radius < minimum_sink_radius
    ] = minimum_sink_radius

    sink_cores.radius = desired_sink_radius
    sink_cores.initial_density = sink_cores.density
    sink_cores.u = 0 | units.kms**2
    sinks = SinkParticles(sink_cores)
    print("AAP: ", sink_cores in gas)
    accreted_gas = sink_cores.copy()
    accreted_gas.add_particles(sinks.accrete(gas))

    logger.info(
        "Number of new sinks: %i",
        len(sinks)
    )
    return sinks, accreted_gas


def main():
    star_forming_region = StarFormingRegion(
        key=None,
        particles_set=None,
        set_index=None,
        set_version=-1,
        mass=150 | units.MSun,
        radius=1000 | units.AU,
        position=[0, 0, 0] | units.parsec,
        velocity=[0, 0, 0] | units.kms,
        initial_mass_function="kroupa",
        binary_fraction=0,
        triple_fraction=0,
        upper_mass_limit=100 | units.MSun,
    )
    print(star_forming_region)
    p = []
    q = Particles()
    p.append(star_forming_region)
    q.add_particle(star_forming_region)
    print(p[0])
    print(q[0])
    print(p[0] == star_forming_region)
    print(q[0] == star_forming_region)
    print(p[0] == q[0])
    new_stars = p[0].yield_next()
    # active_sfr = q[0]
    # q_new_stars = active_sfr.yield_next()
    i = 0
    while not new_stars.is_empty():
        print(i, new_stars.total_mass(), star_forming_region.mass)
        new_stars = p[0].yield_next()
        i += 1


if __name__ == "__main__":
    main()
