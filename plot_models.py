from __future__ import print_function, division
import sys

import matplotlib.pyplot as plt

from amuse.units import units
from amuse.io import read_set_from_file
from amuse.io.base import IoException


def plot_cluster(
        particles,
        xmin=-100,
        xmax=100,
        ymin=-100,
        ymax=100,
        length_unit=units.parsec,
        dpi=200,
        i=0,
        name="figure2"
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        particles.x.value_in(length_unit),
        particles.y.value_in(length_unit),
        s=0.5,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig("%s-%04i.png" % (name, i), dpi=dpi)
    plt.close(fig)
    return


def main():
    # for i in range(24):
    try:
        i = int(sys.argv[1])
    except:
        i = 0
    while True:
        try:
            particles = read_set_from_file("model2-%03i.hdf5" % i, "amuse")
            x, y, z = particles.center_of_mass()
            length = units.parsec
            plot_cluster(
                particles,
                xmin=(-250 + x.value_in(length)),
                xmax=(250 + x.value_in(length)),
                ymin=(-250 + y.value_in(length)),
                ymax=(250 + y.value_in(length)),
                i=i,
            )
        except IoException:
            break
        i += 1
    print(i)

if __name__ == "__main__":
    main()
