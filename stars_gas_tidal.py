"""
Simulate a system of stars and gas, with an external tidal field
"""
from __future__ import division

import os
import sys
# import time
import logging

import numpy

from amuse.units import units
from amuse.io import write_set_to_file, read_set_from_file

# from plot_models import plot_cluster
from plotting_class import plot_hydro_and_stars
from embedded_star_cluster_class import ClusterInPotential

test = True
limited_radius = 100 | units.parsec
logger = logging.getLogger(__name__)

# DEBUG for more, WARNING, ERROR or CRITICAL for less.
logging_level = logging.INFO


def main():
    "Load stars and gas, and evolve them"
    model_name = (
        "test-model" if test else
        "galaxy-model_%03i" % (6)
    )
    save_dir = "Runs/%s" % model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(
        filename="%s/info.log" % (save_dir),
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
    if test:
        logger.info("Creating test initial conditions")
        from amuse.units import nbody_system
        from amuse.ic.plummer import new_plummer_model
        from amuse.ic.gasplummer import new_plummer_gas_model

        star_converter = nbody_system.nbody_to_si(
            3000 | units.MSun,
            3 | units.parsec,
        )
        gas_converter = nbody_system.nbody_to_si(
            10000 | units.MSun,
            3 | units.parsec,
        )
        stars = new_plummer_model(3000, star_converter)
        gas = new_plummer_gas_model(10000, gas_converter)
    else:
        logger.info("Reading initial conditions")
        if len(sys.argv) > 1:
            starfile = sys.argv[1]
            logger.info("Reading stars from file %s", starfile)
            stars = read_set_from_file(starfile, "amuse")
            if limited_radius:
                logger.info(
                    "Limiting stars to %s from center of stellar mass",
                    limited_radius.in_(units.parsec)
                )
                com = stars.center_of_mass()
                stars.from_com = (stars.position - com).lengths()
                selected_stars = stars.select(
                    lambda x: x <= limited_radius,
                    ["from_com"]
                )
                del selected_stars.from_com
                stars = selected_stars
                del selected_stars

            if len(sys.argv) > 2:
                gasfile = sys.argv[2]
                logger.info("Reading gas from file %s", gasfile)
                gas = read_set_from_file(gasfile, "amuse")
                if limited_radius:
                    logger.info(
                        "Limiting gas to %s from center of stellar mass",
                        limited_radius.in_(units.parsec)
                    )
                    gas.from_com = (gas.position - com).lengths()
                    selected_gas = gas.select(
                        lambda x: x <= limited_radius,
                        ["from_com"]
                    )
                    del selected_gas.from_com
                    gas = selected_gas
                    del selected_gas
            else:
                gas = None
        else:
            return
    logger.info("We have %i star particles", len(stars))
    logger.info(
        "Stellar center-of-mass is %s",
        stars.center_of_mass().in_(units.parsec)
    )
    logger.info("We have %i gas particles", len(gas))
    logger.info(
        "Gas center-of-mass is %s",
        gas.center_of_mass().in_(units.parsec)
    )
    logger.info("Creating cluster in potential model")
    model = ClusterInPotential(
        stars=stars,
        gas=gas,
    )

    # length = units.parsec
    timestep = 0.05 | units.Myr
    model_time = 0 | units.Myr
    time_unit = units.Myr

    x, y = model.star_particles.center_of_mass()[0:2]
    plotname = "%s/plot-%04i.png" % (save_dir, 0)
    logger.info("Creating plot")
    plot_hydro_and_stars(
        model.model_time,
        model.gas_code,
        model.star_particles,
        L=(
            600 if not test
            else star_converter.to_si(
                4 | nbody_system.length
            ).value_in(units.parsec)
        ),
        filename=plotname,
        offset_x=x,
        offset_y=y,
        title="time = %06.1f %s" % (
            model_time.value_in(time_unit),
            time_unit,
        ),
    )
    logger.info("Starting evolve loop")
    for i in range(1, 201):
        model_time += timestep
        logger.info("Evolving to %s", model_time.in_(units.Myr))
        model.evolve_model(model_time)
        logger.info("Model time: %s", (model.model_time))
        stars_backup_file = "%s/stars-%04i.hdf5" % (save_dir, i)
        logger.info(
            "Writing backup file of stars to %s",
            stars_backup_file,
        )
        write_set_to_file(
            model.star_particles,
            stars_backup_file, "amuse")
        gas_backup_file = "%s/gas-%04i.hdf5" % (save_dir, i)
        logger.info(
            "Writing backup file of gas to %s",
            gas_backup_file,
        )
        write_set_to_file(
            model.gas_particles,
            gas_backup_file, "amuse")
        x, y = model.star_particles.center_of_mass()[0:2]
        plot_file = "%s/plot-%04i.png" % (save_dir, i)
        logger.info(
            "Making plot, saving to file %s",
            plot_file,
        )
        plot_hydro_and_stars(
            model.model_time,
            model.gas_code,
            model.star_particles,
            L=(
                600 if not test
                else star_converter.to_si(
                    4 | nbody_system.length
                ).value_in(units.parsec)
            ),
            filename=plot_file,
            offset_x=x,
            offset_y=y,
            title="time = %06.1f %s" % (
                model_time.value_in(time_unit),
                time_unit,
            ),
        )


if __name__ == "__main__":
    numpy.random.seed(1)
    main()
