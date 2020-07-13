#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for plotting stuff
"""
import logging
import numpy

from matplotlib import pyplot
# import matplotlib.cm as cm

from amuse.datamodel import Particles
from amuse.io import read_set_from_file
from amuse.units import units, constants

# from prepare_figure import single_frame
# from prepare_figure import figure_frame, set_tickmarks
# from distinct_colours import get_distinct
from mpl_toolkits.axes_grid1 import make_axes_locatable

import default_settings


logger = logging.getLogger(__name__)


def temperature_to_u(
        temperature,
        # gas_mean_molecular_weight=(2.33 / 6.02214179e+23) | units.g,
        gas_mean_molecular_weight=None,
):
    if gas_mean_molecular_weight is None:
        gas_mean_molecular_weight = (((1.0)+0.4) / (0.1+(1.)) / 6.02214179e+23) | units.g
        # print("GMMW = %s" % gas_mean_molecular_weight.value_in(units.g))
    internal_energy = (
        3.0 * constants.kB * temperature
        / (2.0 * gas_mean_molecular_weight)
    )
    # Rg = (constants.kB * 6.02214076e23).value_in(units.erg * units.K**-1)
    # gmwvar = 1.2727272727
    # uergg = 6.6720409999999996E-8

    # internal_energy = (
    #     3./2. * temperature.value_in(units.K) * (Rg/gmwvar/uergg)
    # ) | units.kms**2
    return internal_energy


def u_to_temperature(
        internal_energy,
        # gas_mean_molecular_weight=(2.33 / 6.02214179e+23) | units.g,
        gas_mean_molecular_weight=None,
):
    if gas_mean_molecular_weight is None:
        gas_mean_molecular_weight = (((1.0)+0.4) / (0.1+(1.)) / 6.02214179e+23) | units.g
        # print("GMMW = %s" % gas_mean_molecular_weight.value_in(units.g))
    # temperature = (
    #     internal_energy * (2.0 * gas_mean_molecular_weight)
    #     / (3.0 * constants.kB)
    # )
    temperature = (
        2/3*internal_energy/(constants.kB/gas_mean_molecular_weight)
    )
    # Rg = (constants.kB * 6.02214076e23).value_in(units.erg * units.K**-1)
    # gmwvar = 1.2727272727
    # uergg = 6.6720409999999996E-8

    # temperature = (
    #     2.0/3.0*internal_energy.value_in(units.kms**2) / (Rg/gmwvar/uergg)
    # ) | units.K
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


def make_column_density_map(
        sph, N=default_settings.N, L=default_settings.L, offset_x=None, offset_y=None,
        length_unit=units.parsec, thickness=None,
):
    "Create a density map from an SPH code"
    logger.info("Creating density map for gas")

    xmin = -0.5 * L
    ymin = -0.5 * L
    xmax = 0.5 * L
    ymax = 0.5 * L
    # if offset_x is not None:
    #     xmin += offset_x
    #     xmax += offset_x
    # if offset_y is not None:
    #     ymin += offset_y
    #     ymax += offset_y

    gas = sph.gas_particles.copy()
    gas.x -= offset_x | length_unit
    gas.y -= offset_y | length_unit
    if thickness is not None:
        gas = gas.select(lambda x: x < 0.5 * thickness and x > -0.5 * thickness, ["z"])

    n, x_edges, y_edges = numpy.histogram2d(
        gas.x.value_in(length_unit),
        gas.y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
    )

    square = ((xmax-xmin) * (ymax-ymin) / L**2) | length_unit**2
    gas_coldens, xedges, yedges = numpy.histogram2d(
        gas.x.value_in(length_unit),
        gas.y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
        weights=gas.mass.value_in(units.MSun) / square.value_in(length_unit**2),
    )
    gas_coldens = (gas_coldens) | units.MSun / length_unit**2

    # Convolve with SPH kernel?
    return (gas_coldens, xedges, yedges)


def make_mean_density_map(
        sph, N=default_settings.N, L=default_settings.L,
        length_unit=units.parsec, thickness=None,
        offset_x=None, offset_y=None, offset_z=None,
        x_axis="x", y_axis="y", z_axis="z",
):
    "Create a mean density map from an SPH code"
    logger.info("Creating mean density map for gas")

    gas = sph.gas_particles.copy()
    if offset_x is not None:
        gas.x -= offset_x | length_unit
    if offset_y is not None:
        gas.y -= offset_y | length_unit
    if offset_z is not None:
        gas.z -= offset_z | length_unit
    if thickness is not None:
        gas = gas.select(
            lambda x: x < 0.5 * thickness and x > -0.5 * thickness, [z_axis]
        )

    if x_axis == "x":
        X = gas.x
    elif x_axis == "y":
        X = gas.y
    elif x_axis == "z":
        X = gas.z
    else:
        return -1
    if y_axis == "x":
        Y = gas.x
    elif y_axis == "y":
        Y = gas.y
    elif y_axis == "z":
        Y = gas.z
    else:
        return -1
    if z_axis == "x":
        Z = gas.x
    elif z_axis == "y":
        Z = gas.y
    elif z_axis == "z":
        Z = gas.z
    else:
        return -1
    n, x_edges, y_edges = numpy.histogram2d(
        X.value_in(length_unit),
        Y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
    )

    gas_mdens, xedges, yedges = numpy.histogram2d(
        X.value_in(length_unit),
        Y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
        weights=gas.density.value_in(units.g * units.cm**-3),
    )
    n[n==0] = 1
    gas_mdens = (gas_mdens/n) | units.g * units.cm**-3

    # Convolve with SPH kernel?
    return (gas_mdens, xedges, yedges)


def make_density_map(
        sph, N=default_settings.N, L=default_settings.L, offset_x=None, offset_y=None,
        length_unit=units.kpc, thickness=None,
):
    "Create a density map from an SPH code"
    logger.info("Creating density map for gas")

    xmin = -0.5 * L
    ymin = -0.5 * L
    xmax = 0.5 * L
    ymax = 0.5 * L
    # if offset_x is not None:
    #     xmin += offset_x
    #     xmax += offset_x
    # if offset_y is not None:
    #     ymin += offset_y
    #     ymax += offset_y

    gas = sph.gas_particles.copy()
    gas.x -= offset_x | length_unit
    gas.y -= offset_y | length_unit
    if thickness is not None:
        gas = gas.select(lambda x: x < 0.5 * thickness and x > -0.5 * thickness, ["z"])

    n, x_edges, y_edges = numpy.histogram2d(
        gas.x.value_in(length_unit),
        gas.y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
    )

    gas_rho, xedges, yedges = numpy.histogram2d(
        gas.x.value_in(length_unit),
        gas.y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
        weights=gas.rho.value_in(units.g * units.cm**-3),
    )
    gas_rho = (gas_rho/n) | units.g * units.cm**-3

    # Convolve with SPH kernel?
    return (gas_rho, xedges, yedges)


def make_temperature_map(
        sph, N=default_settings.N, L=default_settings.L,
        offset_x=None, offset_y=None, offset_z=None,
        length_unit=units.parsec, thickness=None,
        x_axis="x", y_axis="y", z_axis="z",
):
    "Create a temperature map from an SPH code"
    logger.info("Creating temperature map for gas")
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

    gas = sph.gas_particles.copy()
    if offset_x is not None:
        gas.x -= offset_x | length_unit
    if offset_y is not None:
        gas.y -= offset_y | length_unit
    if offset_z is not None:
        gas.z -= offset_z | length_unit
    if thickness is not None:
        gas = gas.select(lambda x: x < 0.5 * thickness and x > -0.5 * thickness, [z_axis])

    if x_axis == "x":
        X = gas.x
    elif x_axis == "y":
        X = gas.y
    elif x_axis == "z":
        X = gas.z
    else:
        return -1
    if y_axis == "x":
        Y = gas.x
    elif y_axis == "y":
        Y = gas.y
    elif y_axis == "z":
        Y = gas.z
    else:
        return -1
    if z_axis == "x":
        Z = gas.x
    elif z_axis == "y":
        Z = gas.y
    elif z_axis == "z":
        Z = gas.z
    else:
        return -1

    n, x_edges, y_edges = numpy.histogram2d(
        X.value_in(length_unit),
        Y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
    )

    gas_u, xedges, yedges = numpy.histogram2d(
        X.value_in(length_unit),
        Y.value_in(length_unit),
        bins=N,
        range=[
            [-0.5*L, 0.5*L],
            [-0.5*L, 0.5*L],
        ],
        weights=gas.u.value_in(internal_energy),
        # weights=gas.temperature.value_in(temperature),
    )
    n[n==0] = 1
    gas_u = (gas_u/n) | internal_energy

    gas_temperature = u_to_temperature(gas_u)
    # Convolve with SPH kernel?

    return gas_temperature


def plot_hydro_and_stars(
        time,
        sph,
        stars=None,
        sinks=None,
        L=default_settings.L,
        N=default_settings.N,
        image_size_scale=default_settings.image_size_scale,
        filename=None,
        offset_x=None,
        offset_y=None,
        offset_z=None,
        title="",
        gasproperties=["density",],
        colorbar=False,
        alpha_sfe=0.02,
        stars_are_sinks=False,
        starscale=default_settings.starscale,
        length_unit=units.parsec,
        dpi=default_settings.dpi,
        return_figure=False,
        thickness=None,
        x_axis="x",
        y_axis="y",
        z_axis="z",
):
    "Plot gas and stars"
    logger.info("Plotting gas and stars")
    xmin = -L/2
    xmax = L/2
    ymin = -L/2
    ymax = L/2
    if x_axis == "x":
        if offset_x is not None:
            xmin += offset_x
            xmax += offset_x
    elif x_axis == "y":
        if offset_y is not None:
            xmin += offset_y
            xmax += offset_y
    elif x_axis == "z":
        if offset_z is not None:
            xmin += offset_z
            xmax += offset_z
    if y_axis == "x":
        if offset_x is not None:
            ymin += offset_x
            ymax += offset_x
    elif y_axis == "y":
        if offset_y is not None:
            ymin += offset_y
            ymax += offset_y
    elif y_axis == "z":
        if offset_z is not None:
            ymin += offset_z
            ymax += offset_z

    number_of_subplots = max(
        1,
        len(gasproperties),
    )
    left = 0.2
    bottom = 0.1
    right = 1
    top = 0.9
    # fig = pyplot.figure(figsize=(6, 5))
    image_size = [image_size_scale*N, image_size_scale*N]
    naxes = len(gasproperties)
    figwidth = image_size[0] / dpi / (right - left)
    figheight = image_size[1] / dpi / (top - bottom)
    # figsize = (figwidth + (naxes-1)*0.5*figwidth, figheight)
    figsize = (figwidth, figheight)
    fig = pyplot.figure(figsize=figsize, dpi=dpi)
    # fig, ax = pyplot.subplots(nrows=1, ncols=naxes, figsize=figsize, dpi=dpi)
    # left = 0.1
    # bottom = 0.1
    # right = 0.9
    # top = 0.9
    wspace = 0.5
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    for i in range(number_of_subplots):
        ax = fig.add_subplot(1, naxes, i+1)
        # if colorbar:
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes('right', size='5%', pad=0.1)
        if gasproperties:
            gasproperty = gasproperties[i]
            # print("plotting %s" % gasproperty)
            ax.set_title(gasproperty)
            if gasproperty == "density":

                rho, xedges, yedges = make_mean_density_map(
                    sph, N=N, L=L,
                    length_unit=length_unit, thickness=thickness,
                    offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                    x_axis=x_axis, y_axis=y_axis,
                )
                rho = rho.transpose()
                # print(xedges.min(), xedges.max())
                # print(yedges.min(), yedges.max())

                # content = numpy.log10(
                #     1.e-5+rho.value_in(units.amu/units.cm**3)
                # )
                # from gas_class import sfe_to_density

                vmin = -25  # min value should be 1 particle / surface?
                rho[rho == 0 | units.g/units.cm**3] = 10**vmin | units.g/units.cm**3
                plot_data = numpy.log10(
                    # 1.e-5 + rho.value_in(units.MSun/length_unit**2)
                    rho.value_in(units.g/units.cm**3)
                )
                extent = [xmin, xmax, ymin, ymax]
                vmax = numpy.log10(
                    (
                        sph.parameters.stopping_condition_maximum_density.value_in(
                            units.g/units.cm**3
                        )
                    )
                )
                origin = "lower"
                numpy.nan_to_num(plot_data, nan=10**vmin, neginf=10**vmin, posinf=10**vmax)
                img = ax.imshow(
                    plot_data,
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                    origin=origin,
                    cmap='viridis'
                )
                # img = ax.pcolormesh(
                #     plot_data,
                #     vmin=vmin,
                #     vmax=vmax,
                # )
                # img.cmap.set_under('k')
                img.cmap.set_bad('k', alpha=1.0)
                if colorbar:
                    cbar = pyplot.colorbar(
                        img,
                        # cax=cax,
                        orientation='vertical',
                        pad=0.15,
                        extend='min'
                        # fraction=0.045,
                    )
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.set_label('log mean density [$g/cm^3$]', rotation=270)

            if gasproperty == "temperature":
                temp = make_temperature_map(
                    sph, N=N, L=L,
                    offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                    x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
                    thickness=thickness,
                ).transpose()
                vmin = 0
                vmax = 4
                # No gas -> probably should be very hot
                temp[temp < (10**vmin) | units.K] = 10**vmax | units.K
                img = ax.imshow(
                    numpy.log10(
                        temp.value_in(units.K)
                    ),
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=vmin,
                    vmax=vmax,
                    # cmap="inferno",
                    cmap="cividis",
                    # cmap="coolwarm",
                    origin="lower",
                )
                # img.cmap.set_under('k')
                img.cmap.set_bad('k', alpha=1.0)
                if colorbar:
                    cbar = pyplot.colorbar(
                        img,
                        # cax=cax,
                        orientation='vertical',
                        pad=0.15,
                        extend='min'
                        # fraction=0.045,
                    )
                    cbar.ax.get_yaxis().labelpad = 15
                    cbar.set_label('log mean projected temperature [$K$]', rotation=270)
                ax.set_title("temperature")

        if sinks is not None:
            if not sinks.is_empty():
                # s = 2*(
                #         (
                #         sinks.mass
                #         / sph.parameters.stopping_condition_maximum_density
                #     )**(1/3)
                # ).value_in(units.parsec)
                s = 0.1
                if x_axis == "x":
                    x = sinks.x.value_in(length_unit)
                elif x_axis == "y":
                    x = sinks.y.value_in(length_unit)
                elif x_axis == "z":
                    x = sinks.z.value_in(length_unit)
                if y_axis == "x":
                    y = sinks.x.value_in(length_unit)
                elif y_axis == "y":
                    y = sinks.y.value_in(length_unit)
                elif y_axis == "z":
                    y = sinks.z.value_in(length_unit)
                c = "black" if gasproperty == "temperature" else "white"
                ax.scatter(x, y, s=s, c=c, lw=0)
        if stars is not None:
            if not stars.is_empty():
            #if not stars_are_sinks:
                # m = 100.0*stars.mass/max(stars.mass)
                # directly scale with mass
                # s = starscale * stars.mass / (5 | units.MSun)  # stars.mass.mean()
                # more physical, scale surface ~ with luminosity
                # s = 0.1 * ((stars.mass / (7 | units.MSun))**(3.5 / 2))
                s = 0.1  # 0.1 * ((stars.mass / (7 | units.MSun))**(3.5 / 2))
                # c = stars.mass/stars.mass.mean()
                if x_axis == "x":
                    x = stars.x.value_in(length_unit)
                elif x_axis == "y":
                    x = stars.y.value_in(length_unit)
                elif x_axis == "z":
                    x = stars.z.value_in(length_unit)
                if y_axis == "x":
                    y = stars.x.value_in(length_unit)
                elif y_axis == "y":
                    y = stars.y.value_in(length_unit)
                elif y_axis == "z":
                    y = stars.z.value_in(length_unit)
                c = "black" if gasproperty == "temperature" else "white"
                ax.scatter(x, y, s=s, c=c, lw=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("%s [%s]" % (x_axis, length_unit))
        ax.set_ylabel("%s [%s]" % (y_axis, length_unit))
    # pyplot.title(title)
    fig.suptitle(title)
    # fig.tight_layout()
    if filename is None:
        filename = "test.png"

    if return_figure:
        return fig, ax
    else:
        pyplot.savefig(filename, dpi=dpi)
        # pyplot.show()
        pyplot.close(fig)


# def plot_hydro(time, sph, L=10):
#     "Plot gas"
#     x_label = "x [pc]"
#     y_label = "y [pc]"
#     fig = single_frame(
#         x_label, y_label, logx=False,
#         logy=False, xsize=12, ysize=12,
#     )
#     logger.info("Plotting gas")
#     ax = fig.add_subplot(1, 1, 1,)
# 
#     # gas = sph.code.gas_particles
#     # dmp = sph.code.dm_particles
#     rho, xedges, yedges = make_density_map(sph, N=200, L=L)
#     ax.imshow(
#         numpy.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
#         extent=[-L/2, L/2, -L/2, L/2], vmin=1, vmax=5, origin="lower",
#     )
# 
#     # cbar = fig.colorbar(cax, orientation='vertical', fraction=0.045)
#     # cbar.set_label('projected density [$amu/cm^3$]', rotation=270)
# 
#     # cm = pyplot.cm.get_cmap('RdBu')
#     # cm = pyplot.cm.jet #gist_ncar
#     # if len(dmp):
#     #     # m = 10.0*dmp.mass/dmp.mass.max()
#     #     m = 30*numpy.log10(dmp.mass/dmp.mass.min())
#     #     c = numpy.sqrt(dmp.mass/dmp.mass.max())
#     #     pyplot.scatter(dmp.y.value_in(units.parsec), dmp.x.value_in(
#     #         units.parsec), c=c, s=m, lw=0, cmap=cm)
# 
#     #pyplot.show()


def plot_stars(
        time,
        sph=None,
        stars=None,
        sinks=None,
        L=default_settings.L,
        N=None,
        filename=None,
        offset_x=None,
        offset_y=None,
        title="",
        gasproperties="density",
        colorbar=False,
        alpha_sfe=0.02,
        stars_are_sinks=False,
        starscale=default_settings.starscale,
        fig=None,
):
    "Plot stars, but still accept sph keyword for compatibility reasons"
    logger.info("Plotting stars")
    if sph is None:
        max_density = 100 | units.MSun * units.parsec**-3
    else:
        max_density = sph.parameters.stopping_condition_maximum_density
    xmin = -L/2
    xmax = L/2
    ymin = -L/2
    ymax = L/2
    if offset_x is not None:
        xmin += offset_x
        xmax += offset_x
    if offset_y is not None:
        ymin += offset_y
        ymax += offset_y

    if fig is None:
        # Create new figure
        fig = pyplot.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
        close_fig_when_done = True
    else:
        # Use existing figure
        close_fig_when_done = False
        if ax is None:
            # But new axes
            ax = fig.add_subplot(1, 1, 1)

    if sinks is not None:
        if not sinks.is_empty():
            s = 2*(
                    (
                    sinks.mass
                    / max_density
                )**(1/3)
            ).value_in(units.parsec)
            x = -sinks.x.value_in(units.parsec)
            y = sinks.y.value_in(units.parsec)
            ax.scatter(-x, y, s=s, c="red", lw=0)
    if stars is not None:
        if not stars.is_empty():
            s = starscale * stars.mass / (5 | units.MSun)  # stars.mass.mean()
            # more physical, scale surface ~ with luminosity
            # s = 0.1 * ((stars.mass / (1 | units.MSun))**(3.5 / 2))
            # c = stars.mass/stars.mass.mean()
            x = -stars.x.value_in(units.parsec)
            y = stars.y.value_in(units.parsec)
            ax.scatter(-x, y, s=s, c="white", lw=0)

    ax.set_xlim(xmax, xmin)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")
    ax.set_aspect(1)
    ax.set_facecolor('black')
    fig.suptitle(title)
    if filename is None:
        filename = "test.png"
    pyplot.savefig(filename, dpi=default_settings.dpi)

    if close_fig_when_done:
        pyplot.close(fig)
    # else:
        # just clear up


def new_argument_parser():
    "Parse command line arguments"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        dest='starsfilename',
        default='',
        help='file containing stars (optional) []',
    )
    parser.add_argument(
        '-i',
        dest='sinksfilename',
        default='',
        help='file containing sinks (optional) []',
    )
    parser.add_argument(
        '-g',
        dest='gasfilename',
        default='',
        help='file containing gas (optional) []',
    )
    parser.add_argument(
        '-o',
        dest='imagefilename',
        default='test',
        help='write image to this file [test]',
    )
    parser.add_argument(
        '-n',
        dest='n',
        default=default_settings.N,
        type=int,
        help='number of bins (None)',
    )
    parser.add_argument(
        '-x',
        dest='x',
        default=None,
        type=float,
        help='Central X coordinate (None)',
    )
    parser.add_argument(
        '-y',
        dest='y',
        default=None,
        type=float,
        help='Central Y coordinate (None)',
    )
    parser.add_argument(
        '-z',
        dest='z',
        default=None,
        type=float,
        help='Central Z coordinate (None)',
    )
    parser.add_argument(
        '-w',
        dest='w',
        default=default_settings.L,
        type=float,
        help='Width (None)',
    )
    return parser.parse_args()


def main():
    from amuse.community.phantom.interface import Phantom
    from amuse.units import nbody_system
    o = new_argument_parser()
    gasfilename = o.gasfilename
    starsfilename = o.starsfilename
    sinksfilename = o.sinksfilename
    imagefilename = o.imagefilename
    n = o.n
    x = o.x
    y = o.y
    z = o.z
    w = o.w
    image_size_scale = (
        default_settings.image_size_scale * (default_settings.N / n)
    ) or default_settings.image_size_scale
    stars = read_set_from_file(
        starsfilename,
        "amuse",
    ) if starsfilename != "" else Particles()
    sinks = read_set_from_file(
        sinksfilename,
        "amuse",
    ) if sinksfilename != "" else Particles()
    if gasfilename:
        gas = read_set_from_file(
            gasfilename,
            "amuse",
        )
    else:
        gas = Particles()
    # try:
    #     del gas.u
    # except:
    #     print("can't delete u")
    # try:
    #     u = gas.u.mean()
    # except:
    #     u = 0 | units.kms**2
    #     gas.u = u

    mtot = gas.total_mass()
    com = mtot * gas.center_of_mass()
    if not sinks.is_empty():
        mtot += sinks.total_mass()
        com += sinks.total_mass() * sinks.center_of_mass()
    if not stars.is_empty():
        mtot += stars.total_mass()
        com += stars.total_mass() * stars.center_of_mass()
    com = com / mtot

    print(com.value_in(units.parsec))
    try:
        time = gas.get_timestamp()
    except:
        time = 0.0 | units.Myr
    if time is None:
        time = 0.0 | units.Myr
    converter = nbody_system.nbody_to_si(
        default_settings.gas_rscale,
        default_settings.gas_mscale,
    )
    sph = Phantom(converter)
    sph.parameters.stopping_condition_maximum_density = (
        default_settings.density_threshold
    )
    sph.gas_particles.add_particles(gas)

    gasproperties = ["density", "temperature"]
    for gasproperty in gasproperties:
        L = o.w or default_settings.L
        N = o.n or default_settings.N
        offset_x = x or com[0].value_in(units.parsec)
        offset_y = y or com[1].value_in(units.parsec)
        figure, ax = plot_hydro_and_stars(
            time,
            sph,
            stars=stars,
            sinks=sinks,
            L=L,
            N=N,
            image_size_scale=image_size_scale,
            filename=imagefilename+".pdf",
            offset_x=offset_x,
            offset_y=offset_y,
            title="time = %06.2f %s" % (
                time.value_in(units.Myr),
                units.Myr,
            ),
            gasproperties=[gasproperty],
            # colorbar=True,  # causes weird interpolation
            # alpha_sfe=0.02,
            # stars_are_sinks=False,
            thickness=10 | units.pc,
            starscale=default_settings.starscale,
            length_unit=units.parsec,
            return_figure=True
        )
        plot_cluster_locations = False
        if plot_cluster_locations:
            from find_clusters import find_clusters
            clusters = find_clusters(stars, convert_nbody=converter,)
            for cluster in clusters:
                cluster_com = cluster.center_of_mass()
                cluster_x = cluster_com[0].value_in(units.parsec)
                cluster_y = cluster_com[1].value_in(units.parsec)
                lagrangian = cluster.LagrangianRadii(converter)
                lr90 = lagrangian[0][-2]
                s = lr90.value_in(units.parsec)
                # print("Circle with x, y, z: ", x, y, s)
                circle = pyplot.Circle((cluster_x, cluster_y), s, color='r', fill=False)
                ax.add_artist(circle)
        pyplot.savefig(imagefilename+"-%s.png" % gasproperty, dpi=default_settings.dpi)


if __name__ == "__main__":
    main()
