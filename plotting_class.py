"Class for plotting stuff"
from __future__ import print_function, division
import logging
import numpy

from matplotlib import pyplot
# import matplotlib.cm as cm

# from amuse.datamodel import Particles
# from amuse.io import read_set_from_file
from amuse.units import units, constants

from prepare_figure import single_frame
# from prepare_figure import figure_frame, set_tickmarks
# from distinct_colours import get_distinct
from mpl_toolkits.axes_grid1 import make_axes_locatable


logger = logging.getLogger(__name__)


def temperature_to_u(
        temperature,
        gas_mean_molecular_weight=(2.33 / 6.02214179e+23) | units.g,
):
    internal_energy = (
        3.0 * constants.kB * temperature
        / (2.0 * gas_mean_molecular_weight)
    )
    return internal_energy


def u_to_temperature(
        internal_energy,
        gas_mean_molecular_weight=(2.33 / 6.02214179e+23) | units.g,
):
    temperature = (
        internal_energy * (2.0 * gas_mean_molecular_weight)
        / (3.0 * constants.kB)
    )
    return temperature


def _make_density_map(
        sph, N=100, L=1, offset_x=None, offset_y=None,
):
    "Create a density map from an SPH code"
    logger.info("Creating density map for gas")
    x, y = numpy.indices((N+1, N+1))
    x = L*(x.flatten()-N/2.)/N
    y = L*(y.flatten()-N/2.)/N
    z = x*0.
    vx = 0.*x
    vy = 0.*x
    vz = 0.*x

    x = units.parsec(x)
    if offset_x is not None:
        x += offset_x
    y = units.parsec(y)
    if offset_y is not None:
        y += offset_y
    z = units.parsec(z)
    vx = units.kms(vx)
    vy = units.kms(vy)
    vz = units.kms(vz)

    rho, rhovx, rhovy, rhovz, rhoe = sph.get_hydro_state_at_point(
        x, y, z, vx, vy, vz)
    rho = rho.reshape((N+1, N+1))
    return rho


def make_density_map(
        sph, N=70, L=1, offset_x=None, offset_y=None,
):
    "Create a density map from an SPH code"
    logger.info("Creating density map for gas")
    length = units.parsec

    xmin = -0.5 * L
    ymin = -0.5 * L
    xmax = 0.5 * L
    ymax = 0.5 * L
    if offset_x is not None:
        xmin += offset_x
        xmax += offset_x
    if offset_y is not None:
        ymin += offset_y
        ymax += offset_y

    gas = sph.gas_particles

    n, x_edges, y_edges = numpy.histogram2d(
        gas.x.value_in(length),
        gas.y.value_in(length),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
    )

    gas_rho, xedges, yedges = numpy.histogram2d(
        gas.x.value_in(length),
        gas.y.value_in(length),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
        weights=gas.rho.value_in(units.amu * units.cm**-3),
    )
    gas_rho = (gas_rho/n) | units.amu * units.cm**-3

    # Convolve with SPH kernel?
    return gas_rho


def make_temperature_map(
        sph, N=70, L=1, offset_x=None, offset_y=None,
):
    "Create a temperature map from an SPH code"
    logger.info("Creating temperature map for gas")
    length = units.parsec
    internal_energy = units.m**2 * units.s**-2

    xmin = -0.5 * L
    ymin = -0.5 * L
    xmax = 0.5 * L
    ymax = 0.5 * L
    if offset_x is not None:
        xmin += offset_x
        xmax += offset_x
    if offset_y is not None:
        ymin += offset_y
        ymax += offset_y

    gas = sph.gas_particles

    n, x_edges, y_edges = numpy.histogram2d(
        gas.x.value_in(length),
        gas.y.value_in(length),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
    )
 
    gas_u, xedges, yedges = numpy.histogram2d(
        gas.x.value_in(length),
        gas.y.value_in(length),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
        weights=gas.u.value_in(internal_energy),
        # weights=gas.temperature.value_in(temperature),
    )
    gas_u = (gas_u/n) | internal_energy

    gas_temperature = u_to_temperature(gas_u)
    # Convolve with SPH kernel?

    return gas_temperature


def plot_hydro_and_stars(
        time,
        sph,
        stars,
        L=10,
        N=200,
        filename=None,
        offset_x=None,
        offset_y=None,
        title="",
        gasproperties="density",
        colorbar=False,
        alpha_sfe=0.02,
):
    "Plot gas and stars"
    logger.info("Plotting gas and stars")

    number_of_subplots = max(
        1,
        len(gasproperties),
    )
    fig = pyplot.figure(figsize=(7*number_of_subplots, 5))
    for i in range(number_of_subplots):
        ax = fig.add_subplot(1, number_of_subplots, i+1)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
        if gasproperties:
            gasproperty = gasproperties[i]
            print("plotting %s" % gasproperty)
            ax.set_title(gasproperty)
            if gasproperty == "density":

                rho = make_density_map(
                    sph, N=N, L=L, offset_x=offset_x, offset_y=offset_y,
                ).transpose()
                xmin = -L/2
                xmax = L/2
                ymin = -L/2
                ymax = L/2
                if offset_x is not None:
                    xmin += offset_x.value_in(units.parsec)
                    xmax += offset_x.value_in(units.parsec)
                if offset_y is not None:
                    ymin += offset_y.value_in(units.parsec)
                    ymax += offset_y.value_in(units.parsec)

                # content = numpy.log10(
                #     1.e-5+rho.value_in(units.amu/units.cm**3)
                # )
                # from gas_class import sfe_to_density

                img = ax.imshow(
                    numpy.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
                    extent=[xmin, xmax, ymin, ymax],
                    # vmin=content.min(), vmax=content.max(),
                    vmin=0,
                    vmax=1+numpy.log10(
                        sph.parameters.stopping_condition_maximum_density.value_in(
                            units.amu * units.cm**-3
                        ),
                        # sfe_to_density(
                        #     1,
                        #     alpha=alpha_sfe,
                        # ).value_in(units.amu/units.cm**3),
                    ),
                    origin="lower"
                )
                img.cmap.set_under('k')
                img.cmap.set_bad('k', alpha=1.0)
                if colorbar:
                    cbar = pyplot.colorbar(
                        img, cax=cax, orientation='vertical',
                        pad=0.15,
                        extend='min'
                        # fraction=0.045,
                    )
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.set_label('log projected density [$amu/cm^3$]', rotation=270)

            if gasproperty == "temperature":
                temp = make_temperature_map(
                    sph, N=N, L=L, offset_x=offset_x, offset_y=offset_y,
                ).transpose()
                xmin = -L/2
                xmax = L/2
                ymin = -L/2
                ymax = L/2
                img = ax.imshow(
                    numpy.log10(1.e-5+temp.value_in(units.K)),
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=1,
                    vmax=4,
                    cmap="inferno",
                    origin="lower",
                )
                img.cmap.set_under('k')
                img.cmap.set_bad('k', alpha=1.0)
                if colorbar:
                    cbar = pyplot.colorbar(
                        img, cax=cax, orientation='vertical',
                        pad=0.15,
                        extend='min'
                        # fraction=0.045,
                    )
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.set_label('log projected temperature [$K$]', rotation=270)

        if not stars.is_empty():
            # m = 100.0*stars.mass/max(stars.mass)
            m = 2.0*stars.mass/stars.mass.mean()
            # c = stars.mass/stars.mass.mean()
            x = -stars.x.value_in(units.parsec)
            y = stars.y.value_in(units.parsec)
            ax.scatter(-x, y, s=m, c="white", lw=0)
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x [pc]")
        ax.set_ylabel("y [pc]")
    # pyplot.title(title)
    fig.suptitle(title)
    if filename is None:
        filename = "test.png"
    pyplot.savefig(filename, dpi=300)
    # pyplot.show()
    pyplot.close(fig)


def plot_hydro(time, sph, L=10):
    "Plot gas"
    x_label = "x [pc]"
    y_label = "y [pc]"
    fig = single_frame(
        x_label, y_label, logx=False,
        logy=False, xsize=12, ysize=12,
    )
    logger.info("Plotting gas")
    ax = fig.add_subplot(1, 1, 1,)

    # gas = sph.code.gas_particles
    # dmp = sph.code.dm_particles
    rho = make_density_map(sph, N=200, L=L)
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
    "Parse command line arguments"
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
