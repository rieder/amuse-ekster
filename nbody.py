#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Convert gas to stars based on local star formation efficiency
"""
from __future__ import print_function, division

import sys

import numpy
from amuse.io import (read_set_from_file, write_set_to_file)
from amuse.datamodel import Particles
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.units import units, nbody_system
# from amuse.community.ph4.interface import ph4
# from amuse.community.bhtree.interface import BHTree
# from amuse.community.hermite0.interface import Hermite
# from amuse.community.huayno.interface import Huayno
from amuse.community.mi6.interface import MI6
from fujiipz import merge_two_stars

if __name__ == "__main__":
    numpy.random.seed(11)
    alpha_sfe = 0.02
    particles = read_set_from_file(sys.argv[1], 'amuse')
    total_gas_mass = particles.mass.sum()
    e_loc = (
        alpha_sfe * (
            particles.rho
            / (100 | units.MSun * units.parsec**-3)
            )**0.5
    )

    particles.e_loc = e_loc
    particles[numpy.where(particles.e_loc > 1)].e_loc = 1
    dense_region_particles = particles.select_array(
            lambda x: x > 1000 | units.MSun * units.parsec**-3,
            ["rho"])
    print(
        "Mean sfe: ", particles.e_loc.mean(),
        "Mean sfe (dense): ", dense_region_particles.e_loc.mean(),
    )

    selection_chance = numpy.random.random(len(particles))
    selected_particles = particles.select_array(
        lambda x: x > selection_chance, ["e_loc"]
    )
    stellar_masses = new_salpeter_mass_distribution(
        len(selected_particles),
        mass_min=0.3 | units.MSun,
        mass_max=100. | units.MSun
    )

    total_star_forming_mass = stellar_masses.sum()
    # total_star_forming_mass = (particles.mass * particles.e_loc).sum()
    print(
        "Total gas mass: %s, total star forming mass: %s" % (
            total_gas_mass.in_(units.MSun),
            total_star_forming_mass.in_(units.MSun),
        )
    )

    # stellar_masses = new_salpeter_mass_distribution(1)
    # while stellar_masses.sum() < total_star_forming_mass:
    #     stellar_masses.append(new_salpeter_mass_distribution(1))
    number_of_stars = len(stellar_masses)
    print("# Nr of stars: ", number_of_stars)
    stars = Particles(number_of_stars)
    stars.position = selected_particles.position
    stars.velocity = selected_particles.velocity
    stars.mass = stellar_masses
    rvir = stars.virial_radius().in_(units.parsec)
    mtot = stars.mass.sum()

    print("Virial radius = %s" % rvir)
    print("Stellar mass = %s" % mtot)
    converter = nbody_system.nbody_to_si(rvir, mtot)
    write_set_to_file(stars, "stars.hdf5", "amuse", append_to_file=False)
    # exit()
    # gravity = ph4(converter, number_of_workers=1, redirection="none")
    # gravity = BHTree(converter)
    # gravity = Hermite(converter)
    # gravity = Huayno(converter, mode="SHARED6_COLLISIONS")
    # gravity = Huayno(converter, mode="PASS_KDK")
    gravity = MI6(converter, redirection="none")
    gravity.parameters.maximum_timestep = 0.0005 | units.Myr
    gravity.parameters.calculate_postnewtonian = False
    gravity.parameters.epsilon_squared = (100 | units.AU)**2
    print(gravity.parameters)
    # exit()
    gravity.particles.add_particles(stars)
    energy_error_cumulative = 0.0
    tzero_total_energy = (gravity.kinetic_energy - gravity.potential_energy)
    last_total_energy = tzero_total_energy
    print(
        "Virial ratio |Ek|/|Ep| = %s" % (
            abs(gravity.kinetic_energy)
            / abs(gravity.potential_energy)
        )
    )
    time = 0 | units.Myr
    t_end = 10 | units.Myr
    t_diag = 1 | units.Myr
    dt = 0.02 | units.Myr
    while time < t_end:
        time += dt
        gravity.evolve_model(time)
        if time < gravity.model_time:
            time = gravity.model_time
        total_energy = gravity.kinetic_energy - gravity.potential_energy
        energy_error = (total_energy - last_total_energy) / tzero_total_energy
        energy_error_cumulative += abs(energy_error)
        last_total_energy = total_energy
        print(
            "time: ", gravity.model_time.in_(units.Myr),
            "cumulative energy error: ", energy_error_cumulative,
        )
