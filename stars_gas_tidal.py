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

from embedded_star_cluster_class import ClusterInPotential

test = False


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
            if len(sys.argv) > 2:
                gas = read_set_from_file(sys.argv[2], "amuse")
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

    model_name = "DR21-model_%03i" % (1)
    save_dir = "Runs/%s" % model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(200):
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
        plot_cluster(
            model.star_particles,
            xmin=(-4 + x.value_in(length)),
            xmax=(4 + x.value_in(length)),
            ymin=(-4 + y.value_in(length)),
            ymax=(4 + y.value_in(length)),
            i=i,
            name="DR21-model",
        )


if __name__ == "__main__":
    numpy.random.seed(1)
    main()
