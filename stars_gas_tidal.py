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


def main():
    "Load stars and gas, and evolve them"
    if len(sys.argv) > 1:
        stars = read_set_from_file(sys.argv[1], "amuse")
        if len(sys.argv) > 2:
            gas = read_set_from_file(sys.argv[2], "amuse")
        else:
            gas = None
        # print(gas.dynamical_timescale())
        # exit()
        model = ClusterInPotential(
            stars=stars,
            gas=gas,
        )
    else:
        return

    length = units.parsec
    timestep = 0.1 | units.Myr
    time = 0 | units.Myr
    for i in range(100):
        time += timestep
        print("Evolving to %s" % time.in_(units.Myr))
        model.evolve_model(time)
        print("Model time: %s" % (model.model_time))
        write_set_to_file(
            model.star_particles,
            "model-stars-%04i.hdf5" % i, "amuse")
        write_set_to_file(
            model.gas_particles,
            "model-gas-%04i.hdf5" % i, "amuse")
        x, y = model.particles.center_of_mass()[0:2]
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
