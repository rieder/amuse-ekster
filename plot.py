#!/usr/bin/python
# enc: utf-8
"""
Routines related to plotting
"""
import numpy
import matplotlib.pyplot as pyplot
from prepare_figure import single_frame
from amuse.units import units, nbody_system


class DefaultUnits(object):
    """
    Set units to be used
    """
    def __init__(self):
        self.mass = units.MSun
        self.length = units.parsec
        self.time = units.Myr
        self.speed = units.kms

    def use_physical(self):
        """Use physical units"""
        self.mass = units.MSun
        self.length = units.parsec
        self.time = units.Myr
        self.speed = units.kms

    def use_nbody(self):
        """Use n-body (H'enon) units"""
        self.mass = nbody_system.mass
        self.length = nbody_system.length
        self.time = nbody_system.time
        self.speed = nbody_system.speed


def make_map(sph, number_of_cells=100, length_scaling=1):
    """
    create a density map
    """
    x, y = numpy.indices((number_of_cells+1, number_of_cells+1))
    x = length_scaling*(x.flatten()-number_of_cells/2.)/number_of_cells
    y = length_scaling*(y.flatten()-number_of_cells/2.)/number_of_cells
    z = x*0.
    vx = 0.*x
    vy = 0.*x
    vz = 0.*x

    x = units.parsec(x)
    y = units.parsec(y)
    z = units.parsec(z)
    vx = units.kms(vx)
    vy = units.kms(vy)
    vz = units.kms(vz)

    rho, rhovx, rhovy, rhovz, rhoe = sph.get_hydro_state_at_point(
        x, y, z, vx, vy, vz,
    )
    del(rhovx, rhovy, rhovz, rhoe)
    rho = rho.reshape((number_of_cells+1, number_of_cells+1))
    return rho


def plot_hydro(
        time, i, sph, length_scaling=10,):
    """
    make a plot of the current gas densities in a hydro code
    """
    # pylint: disable=too-many-locals
    unitsystem = DefaultUnits()
    x_label = "x [%s]" % unitsystem.length
    y_label = "y [%s]" % unitsystem.length
    fig = single_frame(
        x_label, y_label,
        logx=False, logy=False,
        xsize=15, ysize=15)

    # gas = sph.gas_particles
    dm_particles = sph.dm_particles
    rho = make_map(sph, number_of_cells=400, length_scaling=length_scaling)
    cax = pyplot.imshow(
        numpy.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
        extent=[
            -length_scaling/2, length_scaling/2,
            -length_scaling/2, length_scaling/2],
        vmin=1, vmax=5, origin="lower")

    cbar = fig.colorbar(cax, orientation='vertical', fraction=0.045)
    cbar.set_label('projected density [$amu/cm^3$]', rotation=270)

    title_label = "time: %08.2f %s" % (
        time.value_in(unitsystem.time), unitsystem.time
    )

    fig.suptitle(title_label)

    # color_map = pyplot.cm.get_cmap('RdBu')
    color_map = pyplot.cm.get_cmap('viridis')
    if not dm_particles.is_empty():
        # m = 10.0*dm_particles.mass/dm_particles.mass.max()
        mass_scaling = 30*numpy.log10(
            dm_particles.mass/dm_particles.mass.min()
        )
        color_scaling = numpy.sqrt(dm_particles.mass/dm_particles.mass.max())
        pyplot.scatter(
            dm_particles.y.value_in(units.parsec),
            dm_particles.x.value_in(units.parsec),
            c=color_scaling, s=mass_scaling, lw=0, cmap=color_map)

    pyplot.savefig('gas-fig-%04i.png' % i)
    pyplot.close(fig)


def plot_stars(
        time, i, gravity, length_scaling=10,):
    """
    make a plot of the current gas densities in a hydro code
    """
    # pylint: disable=too-many-locals
    unitsystem = DefaultUnits()
    x_label = "x [%s]" % unitsystem.length
    y_label = "y [%s]" % unitsystem.length
    fig = single_frame(
        x_label, y_label,
        logx=False, logy=False,
        xsize=15, ysize=15)

    axes = fig.add_subplot(111)
    xmin = (-length_scaling/2)
    xmax = (length_scaling/2)
    ymin = (-length_scaling/2)
    ymax = (length_scaling/2)
    axes.set_xlim((xmin, xmax))
    axes.set_ylim((ymin, ymax))

    stars = gravity.particles

    title_label = "time: %08.2f %s" % (
        time.value_in(unitsystem.time), unitsystem.time
    )

    fig.suptitle(title_label)

    color_map = pyplot.cm.get_cmap('RdBu')

    if not stars.is_empty():
        # m = 10.0*dm_particles.mass/dm_particles.mass.max()
        mass_scaling = 30*numpy.log10(
            stars.mass/stars.mass.min()
        )
        color_scaling = numpy.sqrt(stars.mass/stars.mass.max())
        pyplot.scatter(
            stars.y.value_in(units.parsec),
            stars.x.value_in(units.parsec),
            c=color_scaling, s=mass_scaling, lw=0, cmap=color_map)

    pyplot.savefig('stars-fig-%04i.png' % i)
    pyplot.close(fig)
