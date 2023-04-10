#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for implementing a stellar wind
"""
import sys
import numpy
from numpy import pi

from amuse.units import units, nbody_system, constants
from amuse.datamodel.particles import Particle
from amuse.community.seba.interface import SeBa
# from amuse.community.fi.interface import Fi
from amuse.community.phantom.interface import Phantom
# from amuse.ic.gasplummer import new_plummer_gas_model
from amuse.io import write_set_to_file, read_set_from_file

from amuse.ext.masc import new_star_cluster
from amuse.ext import stellar_wind
from amuse.ext.molecular_cloud import molecular_cloud

from amuse.support.console import set_preferred_units

from ekster.plotting_class import plot_hydro_and_stars, temperature_to_u, u_to_temperature
from ekster.cooling_class import SimplifiedThermalModelEvolver  # , Cooling
from ekster.star_forming_region_class import new_kroupa_mass_distribution
from ekster import ekster_settings


COOLING = False


def determine_short_timestep(sph, gas, h_min=0.1 | units.parsec):
    gamma = sph.parameters.gamma
    C_cour = sph.parameters.C_cour
    eni = gas.u.max()  # eni is the energy of the injected particle
    spsound = numpy.sqrt(gamma*(gamma-1.)*eni)
    return C_cour*h_min/spsound


def main():
    settings = ekster_settings.Settings()
    numpy.random.seed(42)
    evo_headstart = 0.0 | units.Myr
    dt_base = 0.001 | units.Myr
    dt = dt_base
    time = 0 | units.Myr
    time_end = 8 | units.Myr
    Tmin = 22 | units.K

    gas_density = 5e3 | units.amu * units.cm**-3

    increase_vol = 2

    Ngas = increase_vol**3 * 10000
    Mgas = increase_vol**3 * 1000 | units.MSun  # Mgas = Ngas | units.MSun
    volume = Mgas / gas_density  # 4/3 * pi * r**3
    radius = (volume / (pi * 4/3))**(1/3)
    radius = increase_vol * radius  # 15 | units.parsec

    gasconverter = nbody_system.nbody_to_si(Mgas, radius)
    # gasconverter = nbody_system.nbody_to_si(1 | units.pc, 1 | units.MSun)
    # gasconverter = nbody_system.nbody_to_si(1e10 | units.cm, 1e10 | units.g)

    # NOTE: make stars first - so that it remains the same random
    # initialisation even when we change the number of gas particles
    if len(sys.argv) > 1:
        gas = read_set_from_file(sys.argv[1], "amuse")
        stars = read_set_from_file(sys.argv[2], "amuse")
        stars.position = stars.position * 3
    else:
        # stars = new_star_cluster(
        #     stellar_mass=1000 | units.MSun, effective_radius=3 | units.parsec
        # )
        # stars.velocity = stars.velocity * 2.0
        from amuse.datamodel import Particles
        Nstars = 100
        stars = Particles(Nstars)
        stars.position = [0, 0, 0] | units.pc
        stars.velocity = [0, 0, 0] | units.kms
        stars.mass = new_kroupa_mass_distribution(Nstars, mass_min=1 | units.MSun).reshape(Nstars)
        #  25 | units.MSun
        gas = molecular_cloud(targetN=Ngas, convert_nbody=gasconverter).result
        # gas.velocity = gas.velocity * 0.5
        gas.u = temperature_to_u(100 | units.K)
    # gas.x = gas.x
    # gas.y = gas.y
    # gas.z = gas.z
    # gas.h_smooth = (gas.mass / gas_density / (4/3) / pi)**(1/3)
    # print(gas.h_smooth.mean())
    # gas = read_set_from_file("gas_initial.hdf5", "amuse")
    # gas.density = gas_density
    # print(gas.h_smooth.mean())
    # exit()
    u_now = gas.u
    #print(gasconverter.to_nbody(gas[0].u))
    #print(constants.kB.value_in(units.erg * units.K**-1))
    #print((constants.kB * 6.02215076e23).value_in(units.erg * units.K**-1))

    #print(gasconverter.to_nbody(temperature_to_u(10 | units.K)))
    #tempiso = 2.d0/3.d0*ui/(Rg/gmwvar/uergg)
    # print(nbody_system.length**2 / nbody_system.time**2)
    # print(gasconverter.to_si(1 | nbody_system.length**2 / nbody_system.time**2).value_in(units.kms**2))
    # print(gasconverter.to_nbody(temperature_to_u(Tmin)))
    # Rg = (constants.kB * 6.02214076e23).value_in(units.erg * units.K**-1)
    # gmwvar = 1.2727272727
    # uergg = nbody_system.length**2 * nbody_system.time**-2
    # uergg = 6.6720409999999996E-8
    # print(Rg)
    # print(
    #     2.0/3.0*gasconverter.to_nbody(temperature_to_u(Tmin))/(Rg/gmwvar/uergg)
    # )
    # #tempiso, ui, Rg, gmwvar, uergg, udist, utime   1.7552962911187030E-018   2.5778500859241771E-003   83140000.000000000        1.2727272727272725        6.6720409999999996E-008   1.0000000000000000        3871.4231866737564
    # u = 3./2. * Tmin.value_in(units.K) * (Rg/gmwvar/uergg)
    # print(u)
    # print(
    #     2.0/3.0*u/(Rg/gmwvar/uergg)
    # )
    # print(u, Rg, gmwvar, uergg)
    # print(temperature_to_u(10 | units.K).value_in(units.kms**2))
    u = temperature_to_u(20 | units.K)
    #print(gasconverter.to_nbody(u))
    #print(u_to_temperature(u).value_in(units.K))
    # exit()
    # gas.u = u | units.kms**2
    # exit()
    
    # print(gasconverter.to_nbody(gas.u.mean()))
    # print(gasconverter.to_si(gas.u.mean()).value_in(units.kms**2))
    # exit()
    gas.du_dt = (u_now - u_now) / dt  # zero, but in the correct units

    # stars = read_set_from_file("stars.amuse", "amuse")
    # write_set_to_file(stars, 'stars.amuse', 'amuse', append_to_file=False)
    # stars.velocity *= 3
    # stars.vx += 0 | units.kms
    # stars.vy += 0 | units.kms

    M = stars.total_mass() + Mgas
    R = stars.position.lengths().mean()
    # converter = nbody_system.nbody_to_si(M, R)
    # exit()
    # gas = new_plummer_gas_model(Ngas, gasconverter)
    # gas = molecular_cloud(targetN=Ngas, convert_nbody=gasconverter).result
    # gas.u = temperature_to_u(Tmin)
    # gas = read_set_from_file("gas.amuse", "amuse")
    # print(stars.mass == stars.mass.max())
    print(len(stars.mass))
    print(len(stars.mass == stars.mass.max()))
    print(stars[0])
    print(stars[stars.mass == stars.mass.max()])
    mms = stars[stars.mass == stars.mass.max()]
    print("Most massive star: %s" % mms.mass)
    print("Gas particle mass: %s" % gas[0].mass)

    evo = SeBa()
    # sph = Fi(converter, mode="openmp")
    phantomconverter = nbody_system.nbody_to_si(
        settings.gas_rscale,
        settings.gas_mscale,
    )
    sph = Phantom(phantomconverter, redirection="none")
    sph.parameters.ieos = 2
    sph.parameters.icooling = 1
    sph.parameters.alpha = 0.1
    sph.parameters.gamma = 5/3
    sph.parameters.rho_crit = 1e17 | units.amu * units.cm**-3
    sph.parameters.h_soft_sinkgas = 0.1 | units.parsec
    sph.parameters.h_soft_sinksink = 0.1 | units.parsec
    sph.parameters.h_acc = 0.1 | units.parsec
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

    # try:
    #     sph.parameters.timestep = dt
    # except:
    #     print("SPH code doesn't support setting the timestep")
    sph.parameters.stopping_condition_maximum_density = \
        5e-16 | units.g * units.cm**-3
    # sph.parameters.beta = 1.
    # sph.parameters.C_cour = sph.parameters.C_cour / 4
    # sph.parameters.C_force = sph.parameters.C_force / 4
    print(sph.parameters)
    stars_in_sph = stars.copy()  # sph.sink_particles.add_particles(stars)
    # stars_in_sph = sph.sink_particles.add_particles(stars)
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
    # print(gasconverter.to_nbody(gas_in_code[0].u).value_in(nbody_system.specific_energy))
    # ui = temperature_to_u(10 | units.K)
    # Rg = constants.kB * 6.02214179e+23
    # gmwvar = (1.4/1.1) | units.g
    # uergg = 1.# | nbody_system.specific_energy
    # print("gmwvar = %s"%gasconverter.to_si(gmwvar))
    # print("Rg = %s"% gasconverter.to_si(Rg))
    # print("ui = %s"% gasconverter.to_si(ui))
    # #print("uergg = %s"% gasconverter.to_nbody(uergg))
    # print("uergg = %s" % gasconverter.to_si(1 | nbody_system.specific_energy).in_(units.cm**2 * units.s**-2))
    # print("****** %s" % ((2.0/3.0)*ui/(Rg/gmwvar/uergg)) + "*****")
    # print(gasconverter.to_nbody(Rg))
    # print((ui).in_(units.cm**2*units.s**-2))
    # #exit()
    
    # sph.evolve_model(1 | units.day)
    # write_set_to_file(sph.gas_particles, "gas_initial.hdf5", "amuse")
    # exit()

    channel_gas_to_code = gas.new_channel_to(
        gas_in_code,
        attributes=[
            "x", "y", "z", "vx", "vy", "vz", "u",
        ]
    )
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
            "x", "y", "z", "vx", "vy", "vz", "age", "radius", "mass",
            "luminosity", "temperature", "stellar_type",
        ],
    )
    channel_stars_wind_to_code.copy()

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
        # gas_in_code,
        gas,
        Tmin=Tmin,
        # T0=30 | units.K,
        # n0=rho0/reference_mu
    )
    cooling.model_time = sph.model_time
    # cooling_to_code = cooling.particles.new_channel_to(gas

    start_mass = (
        stars.mass.sum()
        + (gas.mass.sum() if not gas.is_empty() else 0 | units.MSun)
    )
    step = 0
    plotnr = 0
    com = stars_in_sph.center_of_mass()
    plot_hydro_and_stars(
        time,
        sph,
        stars=stars,
        sinks=None,
        # L=20,
        # N=100,
        filename="phantom-coolthermalwindtestplot-%04i.png" % step,
        title="time = %06.2f %s" % (time.value_in(units.Myr), units.Myr),
        gasproperties=["density", "temperature"],
        # colorbar=True,
        # starscale=1,
        offset_x=com[0].value_in(units.parsec),
        offset_y=com[1].value_in(units.parsec),
        thickness=5 | units.parsec,
    )

    dt = dt_base
    sph.parameters.time_step = dt
    delta_t = phantomconverter.to_si(2**(-16) | nbody_system.time)
    print("delta_t: %s" % delta_t.in_(units.day))
    # small_step = True
    small_step = False
    plot_every = 100
    subplot_factor = 10
    subplot_enabled = False
    subplot = 0
    while time < time_end:
        time += dt
        print("Gas mean u: %s" % (gas.u.mean().in_(units.erg/units.MSun)))
        print("Evolving to t=%s (%s)" % (time, gasconverter.to_nbody(time)))
        step += 1
        evo.evolve_model(evo_headstart+time)

        print(evo.particles.stellar_type.max())
        channel_stars_evo_from_code.copy()
        channel_stars_grav_to_code.copy()

        if COOLING:
            channel_gas_from_code.copy()
            cooling.evolve_for(dt/2)
            channel_gas_to_code.copy()
        print(
            "min/max temp in gas: %s %s" % (
                u_to_temperature(gas_in_code.u.min()).in_(units.K),
                u_to_temperature(gas_in_code.u.max()).in_(units.K),
            )
        )
        if small_step:
            # Take small steps until a full timestep is done.
            # Each substep is 2* as long as the last until dt is reached
            print("Doing small steps")
            # print(u_to_temperature(sph.gas_particles[0].u))
            # print(sph.gas_particles[0].u)
            old_dt = dt_base
            substeps = 2**8
            dt = old_dt / substeps
            dt_done = 0 * old_dt
            sph.parameters.time_step = dt
            print("adjusted dt to %s, base dt is %s" % (
                dt.in_(units.Myr),
                dt_base.in_(units.Myr),
                )
            )
            sph.evolve_model(sph.model_time + dt)
            dt_done += dt
            while dt_done < old_dt:
                sph.parameters.time_step = dt
                print("adjusted dt to %s, base dt is %s" % (
                    dt.in_(units.Myr),
                    dt_base.in_(units.Myr),
                    )
                )
                sph.evolve_model(sph.model_time + dt)
                dt_done += dt
                dt = min(2*dt, old_dt-dt_done)
                dt = max(dt, old_dt/substeps)
            dt = dt_base
            sph.parameters.time_step = dt
            print(
                "adjusted dt to %s" % sph.parameters.time_step.in_(units.Myr)
            )
            small_step = False
            print("Finished small steps")
            # print(u_to_temperature(sph.gas_particles[0].u))
            # print(sph.gas_particles[0].u)
            # exit()
        else:
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
            channel_gas_from_code.copy()
            cooling.evolve_for(dt/2)
            channel_gas_to_code.copy()

        if wind.has_new_wind_particles():
            subplot_enabled = True
            wind_p = wind.create_wind_particles()
            # nearest = gas.find_closest_particle_to(wind_p.x, wind_p.y, wind_p.z)
            # wind_p.h_smooth = nearest.h_smooth
            wind_p.h_smooth = 100 | units.au
            print("u: %s / T: %s" % (wind_p.u.mean(), u_to_temperature(wind_p.u.mean())))
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
                "wind temperature: %s"
                % (u_to_temperature(wind_p.u))
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
                # delta_t = determine_short_timestep(sph, wind_p, h_min=h_min)
                # print("delta_t is set to %s" % delta_t.in_(units.yr))
        # else:
        #     small_step = True
        print(
            "time: %s sph: %s dM: %s" % (
                time.in_(units.Myr),
                sph.model_time.in_(units.Myr),
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
        print("STEP: %i step%%plot_every: %i" % (step, step % plot_every))
        if step % plot_every == 0:
            plotnr = plotnr + 1
            plot_hydro_and_stars(
                time,
                sph,
                # stars=sph.sink_particles,
                # stars=sph.dm_particles,
                stars=stars,
                sinks=None,
                # L=20,
                # N=100,
                # image_size_scale=10,
                filename=f"phantom-coolthermalwindtestplot-{plotnr:04i}.png",
                title="time = %06.2f %s" % (time.value_in(units.Myr), units.Myr),
                gasproperties=["density", "temperature"],
                # colorbar=True,
                # starscale=1,
                offset_x=com[0].value_in(units.parsec),
                offset_y=com[1].value_in(units.parsec),
                thickness=5 | units.parsec,
            )
            # write_set_to_file(gas, "gas.amuse", "amuse", append_to_file=False)
            # write_set_to_file(stars, "stars.amuse", "amuse", append_to_file=False)
        elif (
                subplot_enabled
                and ((step % (plot_every/subplot_factor)) == 0)
        ):
            plotnr = plotnr + 1
            subplot += 1
            plot_hydro_and_stars(
                time,
                sph,
                # stars=sph.sink_particles,
                # stars=sph.dm_particles,
                stars=stars,
                sinks=None,
                # L=20,
                # N=100,
                # image_size_scale=10,
                filename="phantom-coolthermalwindtestplot-%04i.png" % plotnr,  # int(step/plot_every),
                title="time = %06.2f %s" % (time.value_in(units.Myr), units.Myr),
                gasproperties=["density", "temperature"],
                # colorbar=True,
                # starscale=1,
                offset_x=com[0].value_in(units.parsec),
                offset_y=com[1].value_in(units.parsec),
                thickness=5 | units.parsec,
            )
            if subplot % subplot_factor == 0:
                subplot_enabled = False
        print(
            "Average temperature of gas: %s" % (
                u_to_temperature(gas.u).mean().in_(units.K)
            )
        )
    return


if __name__ == "__main__":
    # set_preferred_units(
    #     units.MSun, units.parsec, units.kms, units.erg, units.Myr,
    #     units.erg * units.MSun**-1,
    # )
    main()
