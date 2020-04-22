#!/usr/bin/env python3
"""
Class for implementing a stellar wind
"""
from __future__ import print_function, division
import numpy

from amuse.units import units, nbody_system
# from amuse.datamodel.particles import Particles, Particle
from amuse.community.seba.interface import SeBa
# from amuse.community.fi.interface import Fi
from amuse.community.phantom.interface import Phantom
# from amuse.ic.gasplummer import new_plummer_gas_model
from amuse.io import write_set_to_file, read_set_from_file

from amuse.ext.masc import new_star_cluster
from amuse.ext import stellar_wind
from amuse.ext.molecular_cloud import molecular_cloud

from amuse.support.console import set_preferred_units

from plotting_class import plot_hydro_and_stars, temperature_to_u
from cooling_class import SimplifiedThermalModelEvolver  # , Cooling


COOLING = False  # True


def determine_short_timestep(sph, gas, h_min=0.1 | units.parsec):
    gamma = sph.parameters.gamma
    C_cour = sph.parameters.C_cour
    eni = gas.u.max()  # eni is the energy of the injected particle
    spsound = numpy.sqrt(gamma*(gamma-1.)*eni)
    return C_cour*h_min/spsound


def main():
    numpy.random.seed(42)
    evo_headstart = 2.0 | units.Myr
    dt_base = 0.0001 | units.Myr
    dt = dt_base
    time = 0 | units.Myr
    time_end = 8 | units.Myr
    Tmin = 10 | units.K

    stars = read_set_from_file("stars.amuse", "amuse")
    # stars = new_star_cluster(
    #     stellar_mass=1000 | units.MSun, effective_radius=7 | units.parsec)
    # stars.velocity *= 3
    # stars.vx += 0 | units.kms
    # stars.vy += 0 | units.kms

    NGas = 160000
    MGas = (NGas/8) | units.MSun
    M = stars.total_mass() + MGas
    R = stars.position.lengths().mean()
    converter = nbody_system.nbody_to_si(M, R)
    gasconverter = nbody_system.nbody_to_si(MGas, 20 | units.parsec)
    print(converter.to_si(1 | nbody_system.energy))
    # exit()
    # gas = new_plummer_gas_model(NGas, gasconverter)
    # gas = molecular_cloud(targetN=NGas, convert_nbody=gasconverter).result
    # gas.u = temperature_to_u(Tmin)
    gas = read_set_from_file("gas.amuse", "amuse")
    mms = stars[stars.mass == stars.mass.max()][0]
    print("Most massive star: %s" % mms.mass)
    print("Gas particle mass: %s" % gas[0].mass)

    evo = SeBa()
    # sph = Fi(converter, mode="openmp")
    sph = Phantom(converter, redirection="none")
    # print(sph.parameters)

    stars_in_evo = evo.particles.add_particles(stars)
    channel_stars_evo_from_code = stars_in_evo.new_channel_to(
        stars,
        attributes=[
            "age", "radius", "mass", "luminosity", "temperature",
            "stellar_type",
        ],
    )
    channel_stars_evo_from_code.copy()

    try:
        sph.parameters.timestep = dt
    except:
        print("SPH code doesn't support setting the timestep")
    sph.parameters.stopping_condition_maximum_density = \
        5e-16 | units.g * units.cm**-3
    try:
        sph.parameters.ieos = 2
    except:
        print("SPH code doesn't support setting ieos")
    sph.parameters.gamma = 5./3.
    # sph.parameters.beta = 1.
    # sph.parameters.C_cour = sph.parameters.C_cour / 4
    # sph.parameters.C_force = sph.parameters.C_force / 4
    print(sph.parameters)
    # stars_in_sph = stars.copy()  # sph.sink_particles.add_particles(stars)
    stars_in_sph = sph.sink_particles.add_particles(stars)
    channel_stars_grav_to_code = stars.new_channel_to(
        # sph.sink_particles,
        # sph.dm_particles,
        stars_in_sph,
        attributes=["mass"]
    )
    channel_stars_grav_from_code = stars_in_sph.new_channel_to(
        stars,
        attributes=["x", "y", "z", "vx", "vy", "vz"],
    )
    # We don't want to accrete gas onto the stars/sinks
    stars_in_sph.radius = 0 | units.RSun
    # stars_in_sph = sph.dm_particles.add_particles(stars)
    # try:
    #     sph.parameters.isothermal_flag = True
    #     sph.parameters.integrate_entropy_flag = False
    #     sph.parameters.gamma = 1
    # except:
    #     print("SPH code doesn't support setting isothermal flag")
    gas_in_code = sph.gas_particles.add_particles(gas)
    # channel_gas_to_code = gas.new_channel_to(
    #     gas_in_code,
    #     attributes=[
    #         "x", "y", "z", "vx", "vy", "vz", "u",
    #     ]
    # )
    # mass is never updated, and if sph is in isothermal mode u is not reliable
    channel_gas_from_code = gas_in_code.new_channel_to(
        gas,
        attributes=[
            "x", "y", "z", "vx", "vy", "vz", "density", "pressure", "rho",
            "u", "h_smooth",
        ],
    )
    channel_gas_from_code.copy()  # Initialise values for density etc

    sph_particle_mass = gas[0].mass  # 0.1 | units.MSun
    r_max = 0.1 | units.parsec

    wind = stellar_wind.new_stellar_wind(
        sph_particle_mass,
        mode="heating",
        r_max=r_max,
        derive_from_evolution=True,
        tag_gas_source=True,
        # target_gas=gas,
        # timestep=dt,
    )
    stars_in_wind = wind.particles.add_particles(stars)
    channel_stars_wind_to_code = stars.new_channel_to(
        stars_in_wind,
        attributes=[
            "age", "radius", "mass", "luminosity", "temperature",
            "stellar_type",
        ],
    )
    channel_stars_wind_to_code.copy()

    u_now = gas.u
    gas.du_dt = (u_now - u_now) / dt  # zero, but in the correct units

    # reference_mu = 2.2 | units.amu
    gasvolume = (4./3.) * numpy.pi * (
        gas.position - gas.center_of_mass()
    ).lengths().mean()**3
    rho0 = gas.total_mass() / gasvolume
    print(rho0.value_in(units.g * units.cm**-3))
    # exit()
    # cooling_flag = "thermal_model"
    # cooling = Cooling(
    cooling = SimplifiedThermalModelEvolver(
        gas,
        Tmin=Tmin,
        # T0=20 | units.K,
        # n0=rho0/reference_mu
    )
    cooling.model_time = sph.model_time

    start_mass = (
        stars.mass.sum()
        + (gas.mass.sum() if not gas.is_empty() else 0 | units.MSun)
    )
    step = 0
    com = stars_in_sph.center_of_mass()
    plot_hydro_and_stars(
        time,
        sph,
        stars=stars,
        sinks=None,
        L=50,
        N=200,
        image_size_scale=3,
        filename="phantom-coolthermalwindtestplot-%04i.png" % step,
        title="time = %06.2f %s" % (time.value_in(units.Myr), units.Myr),
        gasproperties=["density", "temperature"],
        colorbar=True,
        starscale=1,
        offset_x=com[0].value_in(units.parsec),
        offset_y=com[1].value_in(units.parsec),
    )

    dt = dt_base
    delta_t = 1 | units.day
    small_step = True
    while time < time_end:
        print("Gas mean u: %s" % (gas.u.mean().in_(units.erg/units.MSun)))
        step += 1
        evo.evolve_model(evo_headstart+time)
        channel_stars_evo_from_code.copy()
        channel_stars_grav_to_code.copy()
        if COOLING:
            # channel_stars_wind_to_code.copy()
            cooling.evolve_for(dt/2)
            # channel_stars_wind_from_code.copy()
        # channel_gas_to_code.copy()
        print(
            "min/max u in gas: %s %s" % (
                converter.to_nbody(gas_in_code.u.min()),
                converter.to_nbody(gas_in_code.u.max()),
            )
        )
        if small_step:
            sph.evolve_model(time - dt + delta_t)
        sph.evolve_model(time)
        channel_gas_from_code.copy()
        channel_stars_grav_from_code.copy()
        u_previous = u_now
        u_now = gas.u
        gas.du_dt = (u_now - u_previous) / dt

        channel_stars_wind_to_code.copy()
        wind.evolve_model(time)
        # channel_stars_wind_from_code.copy()
        if COOLING:
            cooling.evolve_for(dt/2)

        if wind.has_new_wind_particles():
            wind_p = wind.create_wind_particles()
            wind_p.h_smooth = gas.h_smooth.mean()
            # max_e = (1e44 | units.erg) / wind_p[0].mass
            # max_e = 10 * gas.u.mean()
            # max_e = (1.e48 | units.erg) / wind_p[0].mass
            # wind_p[wind_p.u > max_e].u = max_e
            # wind_p[wind_p.u > max_e].h_smooth = 0.1 | units.parsec
            # print(wind_p.position)
            print(
                "time: %s, wind energy: %s"
                % (time, (wind_p.u * wind_p.mass).sum())
            )
            print(
                "gas particles: %i (total mass %s)"
                % (len(wind_p), wind_p.total_mass())
            )
            # for windje in wind_p:
            #     # print(windje)
            #     source = stars[stars.key == windje.source][0]
            #     windje.position += source.position
            #     windje.velocity += source.velocity
            #     # print(source)
            #     # print(windje)
            # # exit()
            gas.add_particles(wind_p)
            gas_in_code.add_particles(wind_p)
            # for wp in wind_p:
            #     print(wp)
            print("Wind particles added")
            if True:  # wind_p.u.max() > gas_in_code.u.max():
                print("Setting dt to very short")
                small_step = True  # dt = 0.1 | units.yr
                h_min = gas.h_smooth.min()
                delta_t = determine_short_timestep(sph, wind_p, h_min=h_min)
                print("delta_t is set to %s" % delta_t.in_(units.yr))
        else:
            small_step = False
        print(
            "time: %s sph: %s dM: %s" % (
                time,
                sph.model_time,
                (
                    stars.total_mass()
                    + (
                        gas.total_mass()
                        if not gas.is_empty()
                        else (0 | units.MSun)
                    )
                    - start_mass
                )
            )
        )
        # com = sph.sink_particles.center_of_mass()
        # com = sph.dm_particles.center_of_mass()
        com = stars.center_of_mass()
        plot_hydro_and_stars(
            time,
            sph,
            # stars=sph.sink_particles,
            # stars=sph.dm_particles,
            stars=stars,
            sinks=None,
            L=50,
            N=200,
            image_size_scale=3,
            filename="phantom-coolthermalwindtestplot-%04i.png" % step,
            title="time = %06.2f %s" % (time.value_in(units.Myr), units.Myr),
            gasproperties=["density", "temperature"],
            colorbar=True,
            starscale=1,
            offset_x=com[0].value_in(units.parsec),
            offset_y=com[1].value_in(units.parsec),
        )
        write_set_to_file(gas, "gas.amuse", "amuse", append_to_file=False)
        write_set_to_file(stars, "stars.amuse", "amuse", append_to_file=False)
        time += dt
    return


if __name__ == "__main__":
    set_preferred_units(
        units.MSun, units.parsec, units.kms, units.erg, units.Myr,
        units.erg * units.MSun**-1,
    )
    main()
