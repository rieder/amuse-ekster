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
from amuse.units import units, constants, nbody_system
from amuse.community.fi.interface import FiMap

# from prepare_figure import single_frame
# from prepare_figure import figure_frame, set_tickmarks
# from distinct_colours import get_distinct
# from mpl_toolkits.axes_grid1 import make_axes_locatable

import default_settings


logger = logging.getLogger(__name__)

gas_mean_molecular_weight = (
            ((1.0)+0.4) / (0.1+(1.)) / 6.02214179e+23
        ) | units.g


def temperature_to_u(
        temperature,
        # gas_mean_molecular_weight=(2.33 / 6.02214179e+23) | units.g,
        gas_mean_molecular_weight=gas_mean_molecular_weight,
):
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
        gas_mean_molecular_weight=gas_mean_molecular_weight,
):
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


def make_column_density_map(
        mapper,
        gas,
        N=default_settings.N,
        width=default_settings.L | units.pc,
        offset_x=0 | units.pc,
        offset_y=0 | units.pc,
        offset_z=0 | units.pc,
        weight_unit=units.MSun * units.pc**-2,  #  2 * units.amu * units.cm**-3,  # * units.pc**-2,
        x_axis="x",
        y_axis="y",
        z_axis="z",
):
    mapper.parameters.target_x = offset_x
    mapper.parameters.target_y = offset_y
    mapper.parameters.image_width = width
    mapper.parameters.image_size = [N, N]
    # positive z = top layer
    mapper.parameters.projection_direction = [0, 0, -1]
    # positive y = up
    mapper.parameters.upvector = [0, 1, 0]  # y

    pixel_size = (width / N)**2
    weight = (gas.mass / pixel_size).value_in(weight_unit)
    # weight = gas.mass.value_in(units.MSun)  # density.value_in(weight_unit)
    mapper.particles.weight = weight
    column_density_map = mapper.image.pixel_value.transpose()
    return column_density_map


def make_temperature_map(
        mapper,
        gas,
        N=default_settings.N,
        width=default_settings.L | units.pc,
        offset_x=0 | units.pc,
        offset_y=0 | units.pc,
        offset_z=0 | units.pc,
        weight_unit=units.K,
        x_axis="x",
        y_axis="y",
        z_axis="z",
):
    "Create a temperature map"
    logger.info("Creating temperature map for gas")

    mapper.parameters.target_x = offset_x
    mapper.parameters.target_y = offset_y
    mapper.parameters.image_width = width
    mapper.parameters.image_size = [N, N]
    # positive z = top layer
    mapper.parameters.projection_direction = [0, 0, -1]
    # positive y = up
    mapper.parameters.upvector = [0, 1, 0]  # y

    temperature = u_to_temperature(gas.u)
    mapper.particles.weight = temperature.value_in(weight_unit)
    temperature_map = mapper.image.pixel_value.transpose()
    mapper.particles.weight = 1
    count_map = mapper.image.pixel_value.transpose()
    mean_temperature_map = temperature_map / count_map

    return mean_temperature_map


def plot_hydro_and_stars(
        time,
        mapper=None,
        stars=None,
        sinks=None,
        gas=None,
        width=default_settings.L | units.pc,
        L=default_settings.L,
        N=default_settings.N,
        image_size_scale=default_settings.image_size_scale,
        vmin=None,
        vmax=None,
        filename=None,
        offset_x=0 | units.pc,
        offset_y=0 | units.pc,
        offset_z=0 | units.pc,
        title="",
        gasproperties=["density", ],
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
        use_fresco=False,
):
    "Plot gas and stars"
    logger.info("Plotting gas and stars")
    xmin = (-width/2).value_in(length_unit)
    xmax = (width/2).value_in(length_unit)
    ymin = (-width/2).value_in(length_unit)
    ymax = (width/2).value_in(length_unit)
    if x_axis == "x":
        xmin += offset_x.value_in(length_unit)
        xmax += offset_x.value_in(length_unit)
    elif x_axis == "y":
        xmin += offset_y.value_in(length_unit)
        xmax += offset_y.value_in(length_unit)
    elif x_axis == "z":
        xmin += offset_z.value_in(length_unit)
        xmax += offset_z.value_in(length_unit)
    if y_axis == "x":
        ymin += offset_x.value_in(length_unit)
        ymax += offset_x.value_in(length_unit)
    elif y_axis == "y":
        ymin += offset_y.value_in(length_unit)
        ymax += offset_y.value_in(length_unit)
    elif y_axis == "z":
        ymin += offset_z.value_in(length_unit)
        ymax += offset_z.value_in(length_unit)

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
    wspace = 0.5
    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=wspace)

    converter = nbody_system.nbody_to_si(
        1 | units.MSun,
        1 | units.pc,
    )
    stop_mapper = False
    if mapper is None:
        mapper = FiMap(converter, mode="openmp")
        if not hasattr(gas, "radius"):
            print("setting radius")
            gas.radius = gas.h_smooth
        mapper.particles.add_particles(gas)
        stop_mapper = True

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
                # gmmwu = gas_mean_molecular_weight.as_unit()
                # weight_unit = gmmwu * units.cm**-3
                weight_unit = units.MSun * units.pc**-2

                image = make_column_density_map(
                    mapper,
                    gas,
                    N=N,
                    width=width,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    offset_z=offset_z,
                    weight_unit=weight_unit,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    z_axis=z_axis,
                )

                logscale_image = numpy.log10(image)
                # numpy.nan_to_num(logscale_image, nan=10**vmin, neginf=10**vmin, posinf=10**vmax)
                extent = [xmin, xmax, ymin, ymax]
                origin = "lower"
                vmin = -1
                vmax = 3.5
                img = ax.imshow(
                    logscale_image,
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
                    cbar.set_label('log mean density [cm$^-3$]', rotation=270)

            if gasproperty == "temperature":
                temperature_map = make_temperature_map(
                    mapper, gas, N=N, width=L | units.pc,
                    offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                    x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
                    weight_unit=units.K,
                )
                logscale_temperature_map = numpy.log10(temperature_map)
                vmin = 0
                vmax = 4
                # No gas -> probably should be very hot
                # temperature_map[
                #     temperature_map < (10**vmin) | units.K
                # ] = 10**vmax | units.K

                img = ax.imshow(
                    logscale_temperature_map,
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
                    cbar.set_label(
                        'log mean projected temperature [$K$]', rotation=270
                    )
                ax.set_title("temperature")

        if sinks is not None:
            if not sinks.is_empty():
                # Scale sinks the same way as stars
                s = starscale * 0.1 * ((sinks.mass / (7 | units.MSun))**(3.5 / 2))
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
        if stars is not None:  # and not use_fresco:
            # if not stars_are_sinks:
            if not stars.is_empty():
                s = starscale * 0.1 * ((stars.mass / (7 | units.MSun))**(3.5 / 2))
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
                if not use_fresco:
                    use_fresco = 0
                ax.scatter(x, y, s=s, c=c, lw=0, alpha=1-use_fresco)
                print(
                    "    Most massive star is %s "
                    % stars.mass.max().in_(units.MSun)
                )

        if stars is not None and use_fresco:
            from amuse.ext.fresco.fresco import make_image
            converter = nbody_system.nbody_to_si(
                stars.total_mass(),
                width,
            )
            stars.x -= offset_x
            stars.y -= offset_y
            # stars.z -= offset_z | units.pc
            # gas = sph.gas_particles.copy()
            if gas is not None:
                gas.x -= offset_x
                gas.y -= offset_y

            fresco_image, vmax = make_image(
                stars=stars,
                gas=gas,
                converter=converter,
                image_width=width,
                image_size=[N, N],
                percentile=0.9995,
                calc_temperature=True,
                age=0 | units.Myr,
                vmax=None,
                sourcebands='ubvri',
                zoom_factor=N/2048,
                psf_type='hubble',
                psf_sigma=1.0,
                return_vmax=True,
                extinction=True,
            )
            ax.imshow(
                fresco_image,
                extent=[xmin, xmax, ymin, ymax],
                alpha=use_fresco,
                origin='lower',
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("%s [%s]" % (x_axis, length_unit))
        ax.set_ylabel("%s [%s]" % (y_axis, length_unit))

    if stop_mapper:
        mapper.stop()

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


def plot_stars(
        time,
        sph=None,
        stars=None,
        sinks=None,
        L=default_settings.L,
        N=None,
        filename=None,
        offset_x=0 | units.pc,
        offset_y=0 | units.pc,
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
    xmin += offset_x
    xmax += offset_x
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
        default=0,
        type=float,
        help='Central X coordinate (0 [pc])',
    )
    parser.add_argument(
        '-y',
        dest='y',
        default=0,
        type=float,
        help='Central Y coordinate (0 [pc])',
    )
    parser.add_argument(
        '-z',
        dest='z',
        default=0,
        type=float,
        help='Central Z coordinate (0 [pc])',
    )
    parser.add_argument(
        '-w',
        dest='w',
        default=default_settings.L,
        type=float,
        help='Width (None)',
    )
    parser.add_argument(
        '--com',
        dest='use_com',
        action='store_true',
        default=False,
        help='Center on center of mass [False]',
    )
    parser.add_argument(
        '--starscale',
        dest='starscale',
        type=float,
        help='starscale (%f)' % default_settings.starscale,
        default=default_settings.starscale,
    )
    return parser.parse_args()


def main():
    o = new_argument_parser()
    gasfilename = o.gasfilename
    starsfilename = o.starsfilename
    sinksfilename = o.sinksfilename
    imagefilename = o.imagefilename
    n = o.n
    offset_x = o.x | units.pc
    offset_y = o.y | units.pc
    offset_z = o.z | units.pc
    w = o.w
    starscale = o.starscale
    width = w | units.pc
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
        if hasattr(gas, "itype"):
            gas = gas[gas.itype == 1]
        # gas.h_smooth = gas.h
        default_temperature = 30 | units.K
        if not hasattr(gas, "u"):
            print("Setting temperature to %s" % default_temperature)
            gas.u = temperature_to_u(default_temperature)
    else:
        gas = Particles()

    mtot = gas.total_mass()
    com = mtot * gas.center_of_mass()
    if not sinks.is_empty():
        mtot += sinks.total_mass()
        com += sinks.total_mass() * sinks.center_of_mass()
    if not stars.is_empty():
        mtot += stars.total_mass()
        com += stars.total_mass() * stars.center_of_mass()
    com = com / mtot
    if o.use_com:
        offset_x = com[0]
        offset_y = com[1]
        offset_z = com[2]

    print(com.value_in(units.parsec))
    try:
        time = gas.get_timestamp()
    except AttributeError:
        time = 0.0 | units.Myr
    if time is None:
        time = 0.0 | units.Myr
    converter = nbody_system.nbody_to_si(
        # 1 | units.pc, 1 | units.MSun,
        default_settings.gas_rscale,
        default_settings.gas_mscale,
    )

    # gasproperties = ["density", "temperature"]
    gasproperties = ["density"]
    for gasproperty in gasproperties:
        L = o.w or default_settings.L
        N = o.n or default_settings.N
        figure, ax = plot_hydro_and_stars(
            time,
            stars=stars,
            sinks=sinks,
            gas=gas,
            width=width,
            L=L,
            N=N,
            image_size_scale=image_size_scale,
            filename=imagefilename+".pdf",
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
            title="time = %06.2f %s" % (
                time.value_in(units.Myr),
                units.Myr,
            ),
            gasproperties=[gasproperty],
            # colorbar=True,  # causes weird interpolation
            # alpha_sfe=0.02,
            # stars_are_sinks=False,
            thickness=None,
            starscale=starscale,
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
                circle = pyplot.Circle(
                    (cluster_x, cluster_y), s, color='r', fill=False,
                )
                ax.add_artist(circle)
        pyplot.savefig(
            gasproperty + "-" + imagefilename + ".png",
            dpi=default_settings.dpi,
        )


if __name__ == "__main__":
    main()
