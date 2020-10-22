#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic routines
"""
import logging
import argparse
import numpy

from amuse.datamodel import ParticlesSuperset, Particles
from amuse.units import units, nbody_system
from amuse.io import read_set_from_file, write_set_to_file
from amuse.io.base import IoException
from amuse.community.hop.interface import Hop
# from amuse.ext.LagrangianRadii import LagrangianRadii


def identify_subgroups(
        unit_converter,
        particles,
        saddle_density_threshold=None,
        outer_density_threshold=None,
        peak_density_threshold="auto",
        logger=None,
):
    "Identify groups of particles by particle densities"
    # print(peak_density_threshold)
    # exit()
    logger = logger or logging.getLogger(__name__)
    hop = Hop(unit_converter)
    hop.particles.add_particles(particles)
    logger.info("particles added to Hop")
    hop.calculate_densities()
    logger.info("densities calculated")

    try:
        mean_density = hop.particles.density.mean()
    except:
        print("error")
    # if peak_density_threshold == "auto":
    #     peak_density_threshold = mean_density

    hop.parameters.peak_density_threshold = peak_density_threshold
    logger.info(
        "peak density threshold set to %s", peak_density_threshold,
    )
    print(peak_density_threshold/mean_density)
    saddle_density_threshold = (
        0.9*peak_density_threshold
        if saddle_density_threshold is None
        else saddle_density_threshold
    )
    hop.parameters.saddle_density_threshold = saddle_density_threshold
    logger.info(
        "saddle density threshold set to %s", saddle_density_threshold,
    )
    outer_density_threshold = (
        0.01*peak_density_threshold
        if outer_density_threshold is None
        else outer_density_threshold
    )
    hop.parameters.outer_density_threshold = outer_density_threshold
    logger.info(
        "outer density threshold set to %s", saddle_density_threshold,
    )
    hop.do_hop()
    logger.info("doing hop")
    result = [x.get_intersecting_subset_in(particles) for x in hop.groups()]
    hop.stop()
    print("hop done")
    logger.info("stopping hop")
    return result


def run_diagnostics(
        model,
        logger=None,
        length_unit=units.pc,
        mass_unit=units.MSun,
        time_unit=units.Myr,
):
    """
    Run diagnostics on model
    """
    logger = logger or logging.getLogger(__name__)
    stars = model.star_particles
    sinks = model.sink_particles
    gas = model.gas_particles
    converter = model.star_converter

    if not sinks.is_empty():
        non_collisional_bodies = Particles()
        non_collisional_bodies.add_particles(stars)
        non_collisional_bodies.add_particles(sinks)
    else:
        non_collisional_bodies = stars
    groups = identify_subgroups(
        converter,
        non_collisional_bodies,
        peak_density_threshold=1e-16 | units.g * units.cm**-3,
    )
    n_groups = len(groups)
    if hasattr(stars, 'group_id'):
        group_id_offset = 1 + max(stars.group_id)
    else:
        group_id_offset = 1  # a group id of 0 would mean "no group found"
    logger.info("Found %i groups", n_groups)
    for i, group in enumerate(groups):
        group_id = i + group_id_offset
        if hasattr(group, 'group_id'):
            group.previous_group_id = group.group_id
        else:
            group.previous_group_id = 0
        group.group_id = group_id
        stars_in_group = len(group)
        if (stars_in_group > 100):
            mass_in_group = group.total_mass().in_(mass_unit)
            mass_fraction = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            radii, new_mass_fraction = group.LagrangianRadii(
                unit_converter=converter, mf=mass_fraction, cm=group.center_of_mass(),
            )
            assert(new_mass_fraction == mass_fraction)
            radii = radii.value_in(length_unit)
            
            x, y, z = group.center_of_mass().value_in(length_unit)
            median_previous_group_id = numpy.median(group.previous_group_id)
            logger.info(
                "step %i group %i nstars %i mass %s xyz %f %f %f %s origin %i "
                "LR %f %f %f %f %f %f %f %f %f %s",
                model.step, group_id,
                stars_in_group, mass_in_group,
                x, y, z, length_unit,
                median_previous_group_id,
                radii[0], radii[1],
                radii[2], radii[3],
                radii[4], radii[5],
                radii[6], radii[7],
                radii[8],
                length_unit,
            )
    groups = ParticlesSuperset(groups)
    group_identifiers = Particles(keys=groups.key)
    group_identifiers.group_id = groups.group_id
    group_identifiers.previous_group_id = groups.previous_group_id
    return group_identifiers


class BasicEksterModel:
    def __init__(self):
        self.gas_particles = Particles()
        self.star_particles = Particles()
        self.sink_particles = Particles()
        self.star_converter = nbody_system.nbody_to_si(
            0.25 | units.pc,
            0.01 | units.Myr,
        )
        self.step = 0

    def run_diagnostics(self):
        self.groups = run_diagnostics(self)


def new_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', dest="step", type=int, default=None,
        help="snapshot number",
    )
    # parser.add_argument(
    #     '-g', dest="gas", type=str, default=None,
    #     help="gas file",
    # )
    # parser.add_argument(
    #     '-s', dest="stars", type=str, default=None,
    #     help="stars file",
    # )
    # parser.add_argument(
    #     '-i', dest="sinks", type=str, default=None,
    #     help="sinks file",
    # )
    return parser.parse_args()


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename="diagnostics.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
    args = new_argument_parser()
    model = BasicEksterModel()
    model.step = args.step

    try:
        gasfile = "gas-%04i.hdf5" % model.step
        gas = read_set_from_file(gasfile, "amuse")
        model.gas_particles.add_particles(gas)
    except IoException:
        gas = Particles()
    try:
        starsfile = "stars-%04i.hdf5" % model.step
        stars = read_set_from_file(starsfile, "amuse")
        if not hasattr(stars, "group_id"):
            try:
                groupsfile = "groups-%04i.hdf5" % (model.step-1)
                groups = read_set_from_file(groupsfile, "amuse")
                groups_to_stars = groups.new_channel_to(stars)
                groups_to_stars.copy_attributes(["group_id"])
            except IoException:
                stars.group_id = 0
        model.star_particles.add_particles(stars)
    except IoException:
        stars = Particles()
    try:
        sinksfile = "sinks-%04i.hdf5" % model.step
        sinks = read_set_from_file(sinksfile, "amuse")
        model.sink_particles.add_particles(sinks)
    except IoException:
        sinks = Particles()

    model.run_diagnostics()
    write_set_to_file(
        model.groups, "groups-%04i.hdf5" % model.step, "amuse"
    )


if __name__ == "__main__":
    main()
