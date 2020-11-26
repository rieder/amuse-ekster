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
        return [sink, Particles()]

    # We now have the first star that will be formed.
    # Next, we generate a list of stellar masses, so that the last star in the
    # list is just one too many for the sink's mass.

    mass_left = sink.mass - sink.next_primary_mass
    masses = new_star_cluster(
        stellar_mass=mass_left,
        initial_mass_function="kroupa",
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
    new_stars.birth_mass = new_stars.mass
    return [sink, new_stars]


def assign_sink_group(
        sink,
        sink_particles,
        group_radius=1|units.pc,
        group_age=0.1|units.Myr,
        group_speed=0.2|units.kms,
        logger=None
):
    """
    Assign group index to sink particle. All initialised sinks must
    have a group index.
    """
    logger = logger or logging.getLogger(__name__)

    if not hasattr(sink, "in_group"):
        sink.in_group = 0

    number_of_groups = sink_particles.in_group.max()

    logger.info(
        'Grouping parameters: radius %s, age %s, speed %',
        group_radius, group_age, group_speed
    )

    initialised = sink.initialised or False
    if not initialised:
        logger.info(
            "Initialising sink %i for group assignment",
            sink.key
        )

        # Check if this sink belongs to any existing groups. Must
        # pass all checks.
        smallest_Etot = numpy.inf | units.J
        fail1 = fail2 = fail3 = fail4 = 0
        for i in range(number_of_groups):
            i += 1   # Change to one-based index
            group_i = sink_particles[sink_particles.in_group == i]

            # Check 1: see if this sink is within the sampling radius
            # from the center of mass of the i-th group.
            distance_from_group_com = (
                sink.position - group_i.center_of_mass()
            ).length()
            if distance_from_group_com > group_radius:
                #logger.info(
                #    'This sink is beyond group #%i (%s from COM)',
                #    i, distance_from_group_com.in_(units.pc)
                #)
                fail1 += 1
                continue

            # Check 2: see if this sink is within the sampling
            # velocity from the center-of-mass velocity of the group
            speed_from_group_com = (
                sink.velocity - group_i.center_of_mass_velocity()
            ).length()
            if speed_from_group_com > group_speed:
                #logger.info(
                #    'Speed is %s away for COM speed of group #%i',
                #    speed_from_group_com.in_(units.kms), i
                #)
                fail2 += 1
                continue

            # Check 3: see if 'the sink' is similar in age with the group
            age_difference = sink.birth_time - group_i.birth_time.min()
            if age_difference > group_age:
                #logger.info(
                #    'Age of this sink is not similar to group #%i '
                #    '(different by %s)',
                #    i, age_difference.in_(units.Myr)
                #)
                fail3 += 1
                continue

            group_and_sink = group_i.copy()
            group_and_sink.add_particle(sink.copy())
            Etot = (
                group_and_sink.kinetic_energy()
                + group_and_sink.potential_energy()
            )
            # # Check: see if the total energy of the group plus this
            # # sink is less than 0.
            # if Etot >= 0.0 | units.J:
            #     logger.info(
            #         'This sink is unbound to group #%i (Etot = %s)',
            #         i, Etot.in_(units.erg)
            #     )
            #     continue

            # Check 4: see if this sink is the most bound to this
            # group
            if Etot > smallest_Etot:
                #logger.info(
                #    'This sink is not the most bound to group #%i',
                #    i
                #)
                fail4 += 1
                continue

            # At this point, this sink passes all checks
            logger.info(
                "Sink %i passes all checks for group #%i",
                sink.key, i
            )
            smallest_Etot = Etot
            sink.in_group = i

        # If this sink is still unassigned to any of the groups,
        # create its own group
        if sink.in_group == 0:
            sink.in_group = number_of_groups + 1
            logger.info(
                'Failed to assign to any groups (%i %i %i %i), creating group #%i',
                fail1, fail2, fail3, fail4, sink.in_group
            )

        sink.initialised = True

    else:
        logger.info('This sink is already in group #%i', sink.in_group)

    number_of_groups = sink_particles.in_group.max()
    logger.info("There are %i groups right now", number_of_groups)

    return sink


def form_stars_from_group(
    group_index,
    sink_particles,
    upper_mass_limit=100 | units.MSun,
    local_sound_speed=0.2 | units.kms,
    minimum_sink_mass=0.01 | units.MSun,
    logger=None,
    randomseed=None,
    shrink_sinks=True,
    **keyword_arguments
):
    """
    Version 2 (Latest version)

    Form stars from specific group of sinks.
    """
    logger = logger or logging.getLogger(__name__)
    #logger.info(
    #    "Using form_stars_from_group on group %i",
    #    group_index
    #)
    if randomseed is not None:
        logger.info("Setting random seed to %i", randomseed)
        numpy.random.seed(randomseed)

    # Sanity check: each sink particle must be in a group.
    ungrouped_sinks = sink_particles.select_array(
        lambda x: x <= 0, ['in_group']
    )
    if not ungrouped_sinks.is_empty():
        logger.info(
            "WARNING: There exist ungrouped sinks. Something is wrong!"
        )
        return None

    # Consider only group with input group index from here onwards.
    group = sink_particles[sink_particles.in_group == group_index]

    # Sanity check: group must have at least a sink
    if group.is_empty():
        logger.info(
            "WARNING: There is no sink in the group: Something is wrong!"
        )
        return None

    number_of_sinks = len(group)
    group_mass = group.total_mass()
    logger.info(
        "%i sinks found in group #%i with total mass %s",
        number_of_sinks, group_index, group_mass.in_(units.MSun)
    )

    next_mass = generate_next_mass()[0][0]
    try:
        # Within a group, group_next_primary_mass values are either
        # a mass, or 0 MSun. If all values are 0 MSun, this is a
        # new group. Else, only interested on the non-zero value. The
        # non-zero values are the same.

        #logger.info(
        #    'SANITY CHECK: group_next_primary_mass %s',
        #    group.group_next_primary_mass
        #)
        if group.group_next_primary_mass.max() == 0 | units.MSun:
            logger.info('Initiate group #%i for star formation', group_index)
            group.group_next_primary_mass = next_mass
        else:
            next_mass = group.group_next_primary_mass.max()
    # This happens for the first ever assignment of this attribute
    except AttributeError:
        logger.info(
            'AttributeError exception: Initiate group #%i for star formation',
            group_index
        )
        group.group_next_primary_mass = next_mass

    #logger.info("Next mass is %s", next_mass)

    if group_mass < next_mass:
        logger.info(
            "Group #%i is not massive enough for the next star %s",
            group_index, next_mass.in_(units.MSun)
        )
        return None

    # Form stars from the leftover group sink mass
    mass_left = group_mass - next_mass
    masses = new_star_cluster(
        stellar_mass=mass_left,
        upper_mass_limit=upper_mass_limit,
        lower_mass_limit=0.01|units.MSun,
        initial_mass_function='kroupa'
    ).mass
    number_of_stars = len(masses)

    #logger.info(
    #    "%i stars created in group #%i with %i sinks",
    #    number_of_stars, group_index, number_of_sinks
    #)

    new_stars = Particles(number_of_stars)
    new_stars.age = 0 | units.Myr
    new_stars[0].mass = next_mass
    new_stars[1:].mass = masses[:-1]
    group.group_next_primary_mass = masses[-1]
    new_stars = new_stars.sorted_by_attribute("mass").reversed()

    # Create placeholders for attributes of new_stars
    new_stars.position = [0, 0, 0] | units.pc
    new_stars.velocity = [0, 0, 0] | units.kms
    new_stars.origin_cloud = group[0].key
    new_stars.star_forming_radius = 0 | units.pc
    new_stars.star_forming_u = local_sound_speed**2

    #logger.info(
    #    "Group's next primary mass is %s",
    #    group.group_next_primary_mass[0]
    #)

    # Don't mess with the actual group sink particle set.
    star_forming_regions = group.copy()
    star_forming_regions.sorted_by_attribute("mass").reversed()

    # Generate a probability list of star forming region indices the
    # stars should associate to
    probabilities = (
        star_forming_regions.mass/star_forming_regions.mass.sum()
    )
    probabilities /= probabilities.sum()    # Ensure sum is exactly 1
    logger.info(
        "Max & min probabilities: %s, %s",
        probabilities.max(), probabilities.min()
    )

    #logger.info("All probabilities: %s", probabilities)

    # Create index list of star forming regions from probability list
    sample = numpy.random.choice(
        len(star_forming_regions), number_of_stars, p=probabilities
    )

    # Assign the stars to the sampled star forming regions
    star_forming_regions_sampled = star_forming_regions[sample]
    new_stars.position = star_forming_regions_sampled.position
    new_stars.velocity = star_forming_regions_sampled.velocity
    new_stars.origin_cloud = star_forming_regions_sampled.key
    new_stars.star_forming_radius = star_forming_regions_sampled.radius
    try:
        new_stars.star_forming_u = star_forming_regions_sampled.u
    except AttributeError:
        new_stars.star_forming_u = local_sound_speed**2

    # Random position of stars within the sink radius they assigned to
    rho = (
        numpy.random.random(number_of_stars) * new_stars.star_forming_radius
    )
    theta = (
        numpy.random.random(number_of_stars)
        * (2 * numpy.pi | units.rad)
    )
    phi = (
        numpy.random.random(number_of_stars) * numpy.pi | units.rad
    )
    x = (rho * sin(phi) * cos(theta)).value_in(units.pc)
    y = (rho * sin(phi) * sin(theta)).value_in(units.pc)
    z = (rho * cos(phi)).value_in(units.pc)

    dX = list(zip(*[x, y, z])) | units.pc

    # Random velocity, sample magnitude from gaussian with local sound speed
    # like Wall et al (2019)
    # temperature = 10 | units.K

    # or (gamma * local_pressure / density).sqrt()
    velocity_magnitude = numpy.random.normal(
        # loc=0.0,  # <- since we already added the velocity of the sink
        scale=new_stars.star_forming_u.sqrt().value_in(units.kms),
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
    vx = (
        velocity_magnitude * sin(velocity_phi) * cos(velocity_theta)
    ).value_in(units.kms)
    vy = (
        velocity_magnitude * sin(velocity_phi) * sin(velocity_theta)
    ).value_in(units.kms)
    vz = (
        velocity_magnitude * cos(velocity_phi)
    ).value_in(units.kms)

    dV = list(zip(*[vx, vy, vz])) | units.kms

    #logger.info("Updating new stars...")
    new_stars.position += dX
    new_stars.velocity += dV

    # For Pentacle, this is the PP radius
    new_stars.radius = 0.05 | units.parsec

    # Remove sink mass according to the position of stars
    excess_star_mass = 0 | units.MSun
    for s in group:
        #logger.info('Sink mass before reduction: %s', s.mass.in_(units.MSun))
        total_star_mass_nearby = (
            new_stars[new_stars.origin_cloud == s.key]
        ).total_mass()

        # To prevent sink mass becomes negative
        if s.mass > minimum_sink_mass:
            if (s.mass - total_star_mass_nearby) <= minimum_sink_mass:
                excess_star_mass += (
                    total_star_mass_nearby - s.mass + minimum_sink_mass
                )
                #logger.info(
                #    'Sink mass goes below %s; excess mass is now %s',
                #    minimum_sink_mass.in_(units.MSun),
                #    excess_star_mass.in_(units.MSun)
                #)
                s.mass = minimum_sink_mass
            else:
                s.mass -= total_star_mass_nearby
        else:
            excess_star_mass += total_star_mass_nearby
            #logger.info(
            #    'Sink mass is already <= minimum mass allowed; '
            #    'excess mass is now %s',
            #    excess_star_mass.in_(units.MSun)
            #)

        #logger.info('Sink mass after reduction: %s', s.mass.in_(units.MSun))

    # Reduce all sinks in group equally with the excess star mass
    #logger.info('Reducing all sink mass equally with excess star mass...')
    mass_ratio = 1 - excess_star_mass/group.total_mass()
    group.mass *= mass_ratio

    logger.info(
        "Total sink mass in group after sink mass reduction: %s",
        group.total_mass().in_(units.MSun)
    )

    if shrink_sinks:
        group.radius = (
            (group.mass / group.initial_density)
            / (4/3 * numpy.pi)
        )**(1/3)
        #logger.info(
        #    "New radii: %s",
        #    group.radius.in_(units.pc)
        #)

    return new_stars

def form_stars_from_group_older_version(
    group_index,
    sink_particles,
    newly_removed_gas,
    upper_mass_limit=100 | units.MSun,
    local_sound_speed=0.2 | units.kms,
    minimum_sink_mass=0.01 | units.MSun,
    logger=None,
    randomseed=None,
    shrink_sinks=True,
    **keyword_arguments
):
    """
    Form stars from specific group of sinks.

    NOTE: This is the older version where removed gas is
    considered as star-forming region. This is now being
    updated to the above latest version.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(
        "Using form_stars_from_group on group %i",
        group_index
    )
    if randomseed is not None:
        logger.info("Setting random seed to %i", randomseed)
        numpy.random.seed(randomseed)

    # Sanity check: each sink particle must be in a group.
    ungrouped_sinks = sink_particles.select_array(
        lambda x: x <= 0, ['in_group']
    )
    if not ungrouped_sinks.is_empty():
        logger.info(
            "WARNING: There exist ungrouped sinks. Something is wrong!"
        )
        return None

    # Consider only group with input group index from here onwards.
    group = sink_particles[sink_particles.in_group == group_index]

    # Sanity check: group must have at least a sink
    if group.is_empty():
        logger.info(
            "WARNING: There is no sink in the group: Something is wrong!"
        )
        return None

    number_of_sinks = len(group)
    logger.info(
        "%i sinks found in group #%i: %s",
        number_of_sinks, group_index, group.key
    )
    group_mass = group.total_mass()
    logger.info(
        "Group mass: %s", group_mass.in_(units.MSun)
    )

    next_mass = generate_next_mass()[0][0]
    try:
        # Within a group, group_next_primary_mass values are either
        # a mass, or 0 MSun. If all values are 0 MSun, this is a
        # new group. Else, only interested on the non-zero value. The
        # non-zero values are the same.
        logger.info(
            'SANITY CHECK: group_next_primary_mass %s',
            group.group_next_primary_mass
        )
        if group.group_next_primary_mass.max() == 0 | units.MSun:
            logger.info('Initiate group #%i for star formation', group_index)
            group.group_next_primary_mass = next_mass
        else:
            next_mass = group.group_next_primary_mass.max()
    # This happens for the first ever assignment of this attribute
    except AttributeError:
        logger.info(
            'AttributeError exception: Initiate group #%i for star formation',
            group_index
        )
        group.group_next_primary_mass = next_mass

    logger.info("Next mass is %s", next_mass)

    if group_mass < next_mass:
        logger.info(
            "Group #%i is not massive enough for the next star",
            group_index
        )
        return None

    # Form stars from the leftover group sink mass
    mass_left = group_mass - next_mass
    masses = new_star_cluster(
        stellar_mass=mass_left,
        upper_mass_limit=upper_mass_limit,
    ).mass
    number_of_stars = len(masses)

    logger.info(
        "%i stars created in group #%i with %i sinks",
        number_of_stars, group_index, number_of_sinks
    )

    new_stars = Particles(number_of_stars)
    new_stars.age = 0 | units.Myr
    new_stars[0].mass = next_mass
    new_stars[1:].mass = masses[:-1]
    group.group_next_primary_mass = masses[-1]
    new_stars = new_stars.sorted_by_attribute("mass").reversed()

    logger.info(
        "Group's next primary mass is %s",
        group.group_next_primary_mass[0]
    )

    # Create placeholders for attributes of new_stars
    new_stars.position = [0, 0, 0] | units.pc
    new_stars.velocity = [0, 0, 0] | units.kms
    new_stars.origin_cloud = group[0].key
    new_stars.star_forming_radius = 0 | units.pc
    new_stars.star_forming_u = local_sound_speed**2

    # Find the newly removed gas in the group
    removed_gas = Particles()
    if not newly_removed_gas.is_empty():
        for s in group:
            removed_gas_by_this_sink = (
                newly_removed_gas[newly_removed_gas.accreted_by_sink == s.key]
            )
            removed_gas.add_particles(removed_gas_by_this_sink)

    logger.info(
        "%i removed gas found in this group",
        len(removed_gas)
    )

    # Star forming regions that contain the removed gas and the group
    # of sinks
    if not removed_gas.is_empty():
        removed_gas.radius = removed_gas.h_smooth
    star_forming_regions = group.copy()
    star_forming_regions.density = (
        star_forming_regions.initial_density / 1000
    )       # /1000 to reduce likelihood of forming stars in sinks
    star_forming_regions.accreted_by_sink = star_forming_regions.key
    try:
        star_forming_regions.u = star_forming_regions.u
    except AttributeError:
        star_forming_regions.u = local_sound_speed**2
    star_forming_regions.add_particles(removed_gas.copy())
    star_forming_regions.sorted_by_attribute("density").reversed()

    # Generate a probability list of star forming region indices the
    # stars should associate to
    probabilities = (
        star_forming_regions.density/star_forming_regions.density.sum()
    )
    probabilities /= probabilities.sum()    # Ensure sum is exactly 1
    logger.info(
        "Max & min probabilities: %s, %s",
        probabilities.max(), probabilities.min()
    )

    logger.info(
        "%i star forming regions",
        len(star_forming_regions)
    )

    def delta_positions_and_velocities(
            new_stars,
            star_forming_regions,
            probabilities
    ):
        """
        Assign positions and velocities of stars in the star forming regions
        according to the probability distribution
        """
        number_of_stars = len(new_stars)

        # Create an index list of removed gas from probability list
        sample = numpy.random.choice(
            len(star_forming_regions), number_of_stars, p=probabilities
        )

        # Assign the stars to the removed gas according to the sample
        star_forming_regions_sampled = star_forming_regions[sample]
        new_stars.position = star_forming_regions_sampled.position
        new_stars.velocity = star_forming_regions_sampled.velocity
        new_stars.origin_cloud = star_forming_regions_sampled.accreted_by_sink
        new_stars.star_forming_radius = star_forming_regions_sampled.radius
        try:
            new_stars.star_forming_u = star_forming_regions_sampled.u
        except AttributeError:
            new_stars.star_forming_u = local_sound_speed**2

        # Random position of stars within the sink radius they assigned to
        rho = (
            numpy.random.random(number_of_stars)
            * new_stars.star_forming_radius
        )
        theta = (
            numpy.random.random(number_of_stars)
            * (2 * numpy.pi | units.rad)
        )
        phi = (
            numpy.random.random(number_of_stars) * numpy.pi | units.rad
        )
        x = (rho * sin(phi) * cos(theta)).value_in(units.pc)
        y = (rho * sin(phi) * sin(theta)).value_in(units.pc)
        z = (rho * cos(phi)).value_in(units.pc)

        X = list(zip(*[x, y, z])) | units.pc

        # Random velocity, sample magnitude from gaussian with local sound
        # speed like Wall et al (2019)
        # temperature = 10 | units.K

        # or (gamma * local_pressure / density).sqrt()
        velocity_magnitude = numpy.random.normal(
            # loc=0.0,  # <- since we already added the velocity of the sink
            scale=new_stars.star_forming_u.sqrt().value_in(units.kms),
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
        vx = (
            velocity_magnitude * sin(velocity_phi) * cos(velocity_theta)
        ).value_in(units.kms)
        vy = (
            velocity_magnitude * sin(velocity_phi) * sin(velocity_theta)
        ).value_in(units.kms)
        vz = (
            velocity_magnitude * cos(velocity_phi)
        ).value_in(units.kms)

        V = list(zip(*[vx, vy, vz])) | units.kms

        return X, V

    dX, dV = delta_positions_and_velocities(
        new_stars, star_forming_regions, probabilities
    )
    logger.info("Updating new stars...")
    new_stars.position += dX
    new_stars.velocity += dV

    # For Pentacle, this is the PP radius
    new_stars.radius = 0.05 | units.parsec

    # mass_ratio = 1 - new_stars.total_mass()/group.total_mass()
    # group.mass *= mass_ratio

    excess_star_mass = 0 | units.MSun
    for s in group:
        logger.info('Sink mass before reduction: %s', s.mass.in_(units.MSun))
        total_star_mass_nearby = (
            new_stars[new_stars.origin_cloud == s.key]
        ).total_mass()

        # To prevent sink mass becomes negative
        if s.mass > minimum_sink_mass:
            if (s.mass - total_star_mass_nearby) <= minimum_sink_mass:
                excess_star_mass += (
                    total_star_mass_nearby - s.mass + minimum_sink_mass
                )
                logger.info(
                    'Sink mass goes below %s; excess mass is now %s',
                    minimum_sink_mass.in_(units.MSun),
                    excess_star_mass.in_(units.MSun)
                )
                s.mass = minimum_sink_mass
            else:
                s.mass -= total_star_mass_nearby
        else:
            excess_star_mass += total_star_mass_nearby
            logger.info(
                'Sink mass is already <= minimum mass allowed; '
                'excess mass is now %s',
                excess_star_mass.in_(units.MSun)
            )

        logger.info('Sink mass after reduction: %s', s.mass.in_(units.MSun))

    # Reduce all sinks in group equally with the excess star mass
    logger.info('Reducing all sink mass equally with excess star mass...')
    mass_ratio = 1 - excess_star_mass/group.total_mass()
    group.mass *= mass_ratio

    logger.info(
        "Total sink mass in group: %s",
        group.total_mass().in_(units.MSun)
    )

    if shrink_sinks:
        group.radius = (
            (group.mass / group.initial_density)
            / (4/3 * numpy.pi)
        )**(1/3)
        logger.info(
            "New radii: %s",
            group.radius.in_(units.pc)
        )

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
