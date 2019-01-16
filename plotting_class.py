"Class for plotting stuff"
from __future__ import print_function, division
import numpy

from matplotlib import pyplot
# import matplotlib.cm as cm

# from amuse.datamodel import Particles
# from amuse.io import read_set_from_file
from amuse.units import units

from prepare_figure import single_frame
# from prepare_figure import figure_frame, set_tickmarks
# from distinct_colours import get_distinct


def make_map(sph, N=100, L=1):
    "Create a density map from an SPH code"
    x, y = numpy.indices((N+1, N+1))
    x = L*(x.flatten()-N/2.)/N
    y = L*(y.flatten()-N/2.)/N
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
        x, y, z, vx, vy, vz)
    rho = rho.reshape((N+1, N+1))
    return rho


def plot_hydro_and_stars(time, sph, stars, L=10, filename=None):
    fig = pyplot.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1,)
    rho = make_map(sph, N=200, L=L).transpose()
    pyplot.imshow(
        numpy.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
        extent=[-L/2, L/2, -L/2, L/2],
        vmin=1, vmax=5,
        origin="lower"
    )
    # subplot.set_title("GMC at zero age")
    # cbar = fig.colorbar(
    #     ax, ticks=[4, 7.5, 11],
    #     orientation='vertical', fraction=0.045,
    # )
    # cbar.set_label('projected density [$amu/cm^3$]', rotation=270)

    if len(stars):
        # m = 100.0*stars.mass/max(stars.mass)
        m = 3.0*stars.mass/stars.mass.mean()
        c = stars.mass/stars.mass.mean()
        x = -stars.x.value_in(units.parsec)
        y = stars.y.value_in(units.parsec)
        pyplot.scatter(-x, y, s=m, c="white", lw=0)
    pyplot.xlim(-L/2., L/2.)
    pyplot.ylim(-L/2., L/2.)
    # pyplot.title("Molecular cloud at time="+time.as_string_in(units.Myr))
    pyplot.xlabel("x [pc]")
    pyplot.ylabel("y [pc]")
    # pyplot.title("GMC at time="+time.as_string_in(units.Myr))
    if filename is None:
        filename = "test.png"
    pyplot.savefig(filename, dpi=300)
    # pyplot.show()
    pyplot.close(fig)


def plot_hydro(time, sph, L=10):
    x_label = "x [pc]"
    y_label = "y [pc]"
    fig = single_frame(
        x_label, y_label, logx=False,
        logy=False, xsize=12, ysize=12,
    )
    ax = fig.add_subplot(1, 1, 1,)

    # gas = sph.code.gas_particles
    # dmp = sph.code.dm_particles
    rho = make_map(sph, N=200, L=L)
    ax.imshow(
        numpy.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
        extent=[-L/2, L/2, -L/2, L/2], vmin=1, vmax=5, origin="lower",
    )

    # cbar = fig.colorbar(cax, orientation='vertical', fraction=0.045)
    # cbar.set_label('projected density [$amu/cm^3$]', rotation=270)

    # cm = pyplot.cm.get_cmap('RdBu')
    # cm = pyplot.cm.jet #gist_ncar
    # if len(dmp):
    #     # m = 10.0*dmp.mass/dmp.mass.max()
    #     m = 30*numpy.log10(dmp.mass/dmp.mass.min())
    #     c = numpy.sqrt(dmp.mass/dmp.mass.max())
    #     pyplot.scatter(dmp.y.value_in(units.parsec), dmp.x.value_in(
    #         units.parsec), c=c, s=m, lw=0, cmap=cm)

    pyplot.show()


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option(
        "-f", dest="filename", default="GMC_R2pcN20k_SE_T45Myr.amuse",
        help="input filename [%default]",
    )
    return result


if __name__ in ('__main__', '__plot__'):
    o, arguments = new_option_parser().parse_args()
    # plot_molecular_cloud(o.filename)
