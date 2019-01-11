"""
Simulate a system of stars and gas, with an external tidal field
"""
from __future__ import print_function, division

import sys

import numpy

from amuse.units import units
from amuse.io import write_set_to_file, read_set_from_file

from plot_models import plot_cluster

from embedded_star_cluster_class import ClusterInPotential

test = True

def main():
    "Load stars and gas, and evolve them"
    if test:
        from amuse.units import nbody_system
        from amuse.ic.plummer import new_plummer_model
        from amuse.ic.gasplummer import new_gas_plummer_model

        star_converter = nbody_system.nbody_to_si(
            100 | units.MSun,
            5 | units.parsec,
        )
        gas_converter = nbody_system.nbody_to_si(
            1000 | units.MSun,
            5 | units.parsec,
        )
        stars = new_plummer_model(1000, star_converter)
        gas = new_gas_plummer_model(10000, gas_converter)
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
    model = ClusterInPotential(
        stars=stars,
        gas=gas,
    )

    length = units.parsec
    timestep = 0.01 | units.Myr
    time = 0 | units.Myr

    model_name = "Testmodel_%03i" % (1)
    save_dir = "Runs/%s" % model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(100):
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
            xmin=(-400 + x.value_in(length)),
            xmax=(400 + x.value_in(length)),
            ymin=(-400 + y.value_in(length)),
            ymax=(400 + y.value_in(length)),
            i=i,
            name="model-ph4",
        )


if __name__ == "__main__":
    numpy.random.seed(1)
    main()
