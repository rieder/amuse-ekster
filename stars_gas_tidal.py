"""
Simulate a system of stars and gas, with an external tidal field
"""
from __future__ import print_function, division

import os
import sys

import numpy

from amuse.units import units
from amuse.io import write_set_to_file, read_set_from_file

from plot_models import plot_cluster
from plotting_class import plot_hydro_and_stars
from embedded_star_cluster_class import ClusterInPotential

test = False
limited_radius = 100 | units.parsec


def main():
    "Load stars and gas, and evolve them"
    if test:
        from amuse.units import nbody_system
        from amuse.ic.plummer import new_plummer_model
        from amuse.ic.gasplummer import new_plummer_gas_model

        star_converter = nbody_system.nbody_to_si(
            1000 | units.MSun,
            2 | units.parsec,
        )
        gas_converter = nbody_system.nbody_to_si(
            1000 | units.MSun,
            2 | units.parsec,
        )
        stars = new_plummer_model(1000, star_converter)
        gas = new_plummer_gas_model(10000, gas_converter)
    else:
        if len(sys.argv) > 1:
            stars = read_set_from_file(sys.argv[1], "amuse")
            if limited_radius:
                com = stars.center_of_mass()
                stars.from_com = (stars.position - com).lengths()
                selected_stars = stars.select(
                    lambda x: x <= limited_radius,
                    ["from_com"]
                )
                del(selected_stars.from_com)
                stars = selected_stars
                del(selected_stars)

            if len(sys.argv) > 2:
                gas = read_set_from_file(sys.argv[2], "amuse")
                if limited_radius:
                    gas.from_com = (gas.position - com).lengths()
                    selected_gas = gas.select(
                        lambda x: x <= limited_radius,
                        ["from_com"]
                    )
                    del(selected_gas.from_com)
                    gas = selected_gas
                    del(selected_gas)
            else:
                gas = None
            # print(gas.dynamical_timescale())
            # exit()
        else:
            return
    print(len(stars))
    print(stars.center_of_mass().in_(units.parsec))
    print(len(gas))
    print(gas.center_of_mass().in_(units.parsec))
    model = ClusterInPotential(
        stars=stars,
        gas=gas,
    )

    length = units.parsec
    timestep = 0.05 | units.Myr
    time = 0 | units.Myr

    model_name = "galaxy-model_%03i" % (2)
    save_dir = "Runs/%s" % model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x, y = model.star_particles.center_of_mass()[0:2]
    plotname = "%s/plot-%04i.png" % (save_dir, 0)
    plot_hydro_and_stars(
        model.model_time, model.gas_code, model.star_particles, L=800,
        filename=plotname,
        offset_x=x,
        offset_y=y,
    )
    for i in range(1, 201):
        time += timestep
        print("Evolving to %s" % time.in_(units.Myr))
        model.evolve_model(time)
        print("Model time: %s" % (model.model_time))
        write_set_to_file(
            model.star_particles,
            "%s/stars-%04i.hdf5" % (save_dir, i), "amuse")
        write_set_to_file(
            model.gas_particles,
            "%s/gas-%04i.hdf5" % (save_dir, i), "amuse")
        x, y = model.star_particles.center_of_mass()[0:2]
        plotname = "%s/plot-%03i.png" % (save_dir, i)
        plot_hydro_and_stars(
            model.model_time, model.gas_code, model.star_particles, L=600,
            filename=plotname,
            offset_x=x,
            offset_y=y,
        )


if __name__ == "__main__":
    numpy.random.seed(1)
    main()
