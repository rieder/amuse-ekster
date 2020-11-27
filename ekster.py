#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for a star cluster embedded in a tidal field and a gaseous region
"""

import sys
import os
import logging
import pickle
import numpy
import concurrent.futures

from amuse.datamodel import ParticlesSuperset, Particles, Particle
from amuse.units import units, nbody_system  # , constants
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits
from amuse.units.quantities import VectorQuantity

from amuse.support.console import set_preferred_units
from amuse.io import write_set_to_file

from amuse.ext import stellar_wind
# from amuse.ext.masc import new_star_cluster
from amuse.ext.sink import new_sink_particles

from gas_class import GasCode
from sinks_class import should_a_sink_form  # , sfe_to_density
from star_cluster_class import StarCluster
from plotting_class import plot_hydro_and_stars  # , plot_stars
from plotting_class import u_to_temperature, temperature_to_u
from merge_recipes import form_new_star
from star_forming_region_class import (
    form_stars,
    form_stars_from_group, assign_sink_group,
)
from bridge import (
    Bridge, CalculateFieldForCodes,
)
import spiral_potential

import ekster_settings
import available_codes


set_preferred_units(
    units.Myr, units.kms, units.pc, units.MSun, units.g * units.cm**-3
)


def new_argument_parser(settings):
    "Parse command line arguments"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        dest='filename_stars',
        default=settings.filename_stars,
        help='file containing stars (optional) [%s]' % (
            settings.filename_stars
        ),
    )
    parser.add_argument(
        '-i',
        dest='filename_sinks',
        default=settings.filename_sinks,
        help='file containing sinks (optional) [%s]' % (
            settings.filename_sinks
        ),
    )
    parser.add_argument(
        '-g',
        dest='filename_gas',
        default=settings.filename_gas,
        help='file containing gas (optional) [%s]' % (
            settings.filename_gas
        ),
    )
    parser.add_argument(
        '-r',
        dest='filename_random',
        default=settings.filename_random,
        help='file containing random state (optional) [%s]' % (
            settings.filename_random
        ),
    )
    parser.add_argument(
        '-d',
        dest='rundir',
        default=settings.rundir,
        help='directory to store run in (optional) [%s]' % (
            settings.rundir
        ),
    )
    parser.add_argument(
        '-c',
        dest='settingfilename',
        default='settings.ini',
        help='settings file [settings.ini]',
    )
    parser.add_argument(
        '--setup',
        dest='setup',
        default="default",
        help='configuration setup [default]',
    )
    parser.add_argument(
        '--writesetup',
        dest='writesetup',
        default=False,
        action="store_true",
        help='write default settings file and exit',
    )
    return parser.parse_args()


class ClusterInPotential(
        StarCluster,
):
    """
    Stellar cluster in an external potential

    This class builds on StarCluster, and extends it with support for gas, an
    external tidal field, and star formation.
    """

    def __init__(
            self,
            stars=Particles(),
            gas=Particles(),
            sinks=Particles(),
            epsilon=0.1 | units.parsec,
            star_converter=None,
            gas_converter=None,
            logger=None,
            logger_level=None,
            settings=ekster_settings.Settings(),
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.settings = settings
        if logger_level is None:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logger_level)

        # Set begin time to previous begin time + previous model time
        self.begin_time = settings.begin_time + settings.model_time

        self.new_stars_added = False

        # FIXME because this is no longer true
        # This class stores particle data by itself, in contrast to StarCluster
        self.gas_state = "clean"
        self.gas_particles = Particles()
        self.dm_particles = Particles()
        # self.sink_particles = SinkParticles(
        #     Particles(),
        #     sink_radius=settings.minimum_sink_radius,
        # )
        self.sink_particles = Particles()
        self.star_particles = Particles()

        # mass_scale_stars = 200 | units.MSun  # or stars.mass.sum()
        length_scale_stars = 0.25 | units.parsec
        # resolve timescales on a factor 2^x times this
        time_scale_stars = 0.01 | units.Myr
        if star_converter is None:
            converter_for_stars = nbody_system.nbody_to_si(
                # mass_scale_stars,
                length_scale_stars,
                time_scale_stars,
            )
        else:
            converter_for_stars = star_converter

        self.logger.info("Initialising StarCluster")
        StarCluster.__init__(
            self,
            # stars=stars,
            converter=converter_for_stars,
            epsilon=epsilon,
            # star_code=available_codes.Hermite,
            # star_code=available_codes.ph4,
            star_code=settings.star_code,
        )
        self.logger.info("Initialised StarCluster")

        if not gas.is_empty():
            mass_scale_gas = gas.mass.sum()
            length_scale_gas = 100 | units.parsec
            if gas_converter is None:
                converter_for_gas = nbody_system.nbody_to_si(
                    mass_scale_gas,
                    length_scale_gas,
                )
            else:
                converter_for_gas = gas_converter
            self.logger.info("Initialising Gas")

            phantom_solarm = 1.9891e33 | units.g
            phantom_pc = 3.086e18 | units.cm
            phantom_gg = 6.672041e-8 | units.cm**3 * units.g**-1 * units.s**-2
            phantom_length = 0.1 * phantom_pc
            phantom_mass = 1.0 * phantom_solarm
            new_gas_converter = ConvertBetweenGenericAndSiUnits(
                # Phantom uses CGS units internally, scaled with G=1
                # So we need to make sure we use those same units here...
                phantom_length,  # 0.1 pc
                phantom_mass,  # 1.0 MSun
                (phantom_length**3 / (phantom_gg*phantom_mass))**0.5,
            )
            # new_gas_converter = nbody_system.nbody_to_si(
            #     settings.gas_rscale,
            #     settings.gas_mscale,
            # )
            self.isothermal_mode = False if settings.ieos != 1 else True
            self.gas_code = GasCode(
                converter=new_gas_converter,
                settings=settings,
            )
            self.sink_code = self.star_code
            print(self.gas_code.parameters)

            print("****Adding gas****")
            self.add_gas(gas)
            # print(self.gas_code.parameters)
            self.timestep = settings.timestep
            self.timestep_bridge = settings.timestep_bridge
            self.logger.info("Initialised Gas")

        self.wind_particles = Particles()
        self.wind = stellar_wind.new_stellar_wind(
            self.gas_particles.mass.min(),
            mode="heating",
            # mode="accelerate",
            # mode="simple",
            r_max=0.1 | units.parsec,
            derive_from_evolution=True,
            target_gas=self.wind_particles,
            timestep=self.timestep,
        )

        # We need to be careful here - stellar evolution needs to re-calculate
        # all the stars unfortunately...
        # self.add_stars(stars)
        if not stars.is_empty():
            self.star_particles.add_particles(stars)

            stars_with_original_mass = stars.copy()
            if not hasattr(stars_with_original_mass, "birth_mass"):
                self.logger.info(
                    "No birth mass recorded for stars, setting to current mass"
                )
                stars_with_original_mass.birth_mass = \
                    stars_with_original_mass.mass
            else:
                stars_with_original_mass.mass = \
                    stars_with_original_mass.birth_mass
            if not hasattr(stars_with_original_mass, "birth_time"):
                self.logger.info(
                    "No birth time recorded for stars, setting to current time"
                )
                stars_with_original_mass.birth_time = 0 | units.Myr
            epochs, indices = numpy.unique(
                stars_with_original_mass.birth_time,
                return_inverse=True,
            )

            self.evo_code.__begin_time = epochs[0]
            for i, time in enumerate(epochs):
                if time != epochs[0]:
                    self.evo_code.evolve_model(time)
                self.evo_code.particles.add_particles(
                    stars_with_original_mass[indices == i],
                )
            self.evo_code.evolve_model(self.begin_time)
            self.evo_code_stars = self.evo_code.particles
            self.sync_from_evo_code()

            self.star_code.particles.add_particles(stars)
            self.wind.particles.add_particles(stars)

        if not sinks.is_empty():
            self.add_sinks(sinks)

        if hasattr(spiral_potential, settings.Tide):
            Tide = getattr(spiral_potential, settings.Tide)
            self.logger.info("Using tidal field %s", settings.Tide)
            self.tidal_field = Tide(
                t_start=self.begin_time + settings.tide_time_offset,
                spiral_type=settings.tide_spiral_type,
            )
            self.logger.info(
                "Initialised tidal field at time %s",
                self.begin_time + settings.tide_time_offset
            )
        else:
            self.tidal_field = False
            self.logger.info(
                "Not using a tidal field"
            )

        self.epsilon = epsilon
        self.converter = converter_for_gas

        def new_field_tree_gravity_code(
                # code=BHTree,
                code=available_codes.Fi,
                # code=Petar,
        ):
            "Create a new field tree code"
            print("Creating field tree code")
            result = code(
                self.converter,
                # redirection="none",
                mode="openmp",
            )
            result.parameters.epsilon_squared = self.epsilon**2
            # result.parameters.timestep = 0.5 * self.timestep
            return result

        def new_field_direct_gravity_code(
                code=available_codes.FastKick,
        ):
            "Create a new field direct code"
            print("Creating field direct code")
            result = code(
                self.converter,
                # redirection="none",
                number_of_workers=8,
            )
            result.parameters.epsilon_squared = self.epsilon**2
            return result

        def new_field_code(
                code,
                mode="direct",
        ):
            " something"
            if mode == "tree":
                new_field_gravity_code = new_field_tree_gravity_code
            elif mode == "direct":
                new_field_gravity_code = new_field_direct_gravity_code
            else:
                new_field_gravity_code = new_field_direct_gravity_code
            result = CalculateFieldForCodes(
                new_field_gravity_code,
                [code],
            )
            return result

        to_gas_codes = []
        if self.tidal_field:
            to_gas_codes.append(self.tidal_field)
        to_gas_codes.append(
            # self.star_code,
            new_field_code(
                self.star_code,
                # mode="direct",
                mode="tree",
            )
        )
        to_stars_codes = []
        if self.tidal_field:
            to_stars_codes.append(self.tidal_field)
        to_stars_codes.append(
            # self.gas_code,
            new_field_code(
                self.gas_code,
                # mode="direct",
                mode="tree",
            )
        )

        self.system = Bridge(
            timestep=(
                self.timestep_bridge
            ),
            # use_threading=True,
            use_threading=False,
        )
        self.system.add_system(
            self.star_code,
            partners=to_stars_codes,
            do_sync=True,
            # zero_smoothing=True,  # for petar
        )
        self.system.add_system(
            self.gas_code,
            partners=to_gas_codes,
            do_sync=True,
            h_smooth_is_eps=True,
        )
        # self.gas_code.parameters.time_step = 0.025 * self.timestep
        self.gas_code.parameters.time_step = self.timestep_bridge/2

    def sync_from_gas_code(self):
        """
        copy relevant attributes changed by gas/sink/dm dynamics
        """
        from_gas_attributes = [
            "x", "y", "z", "vx", "vy", "vz",
            "density", "h_smooth",
        ]
        if not self.isothermal_mode:
            from_gas_attributes.append("u")
        from_dm_attributes = [
            "x", "y", "z", "vx", "vy", "vz",
        ]
        channel_from_gas = \
            self.gas_code.gas_particles.new_channel_to(self.gas_particles)
        channel_from_dm = \
            self.gas_code.dm_particles.new_channel_to(self.dm_particles)
        channel_from_gas.copy_attributes(
            from_gas_attributes,
        )
        if not self.dm_particles.is_empty():
            channel_from_dm.copy_attributes(
                from_dm_attributes,
            )

    def sync_to_gas_code(self):
        """
        copy potentially modified attributes that matter to gas/sink/dm
        dynamics
        """
        channel_to_gas = \
            self.gas_particles.new_channel_to(self.gas_code.gas_particles)
        channel_to_dm = \
            self.dm_particles.new_channel_to(self.gas_code.dm_particles)
        channel_to_gas.copy_attributes(
            ["x", "y", "z", "vx", "vy", "vz", "u"]
        )
        if not self.dm_particles.is_empty():
            channel_to_dm.copy_attributes(
                ["x", "y", "z", "vx", "vy", "vz"]
            )

    def sync_from_evo_code(self):
        """
        copy relevant attributes changed by stellar evolution
        """
        from_stellar_evolution_attributes = [
            "radius", "luminosity", "temperature", "age",
            "stellar_type",  # NOTE: ignoring mass for now!
        ]
        channel_from_star_evo = \
            self.evo_code.particles.new_channel_to(self.star_particles)
        channel_from_star_evo.copy_attributes(
            from_stellar_evolution_attributes
        )

    def sync_to_evo_code(self):
        """
        copy potentially modified attributes that matter to stellar evolution
        """
        to_stellar_evolution_attributes = [
            "mass", "age",
        ]
        channel_to_star_evo = \
            self.star_particles.new_channel_to(self.evo_code.particles)
        channel_to_star_evo.copy_attributes(
            to_stellar_evolution_attributes
        )

    def sync_from_wind_code(self):
        """
        copy star properties from wind
        """
        channel_from_wind = \
            self.wind.particles.new_channel_to(self.star_particles)
        channel_from_wind.copy()

    def sync_to_wind_code(self):
        """
        copy star properties to wind
        """
        to_wind_attributes = [
            "x", "y", "z", "vx", "vy", "vz", "age", "radius", "mass",
            "luminosity", "temperature", "stellar_type",
        ]
        channel_to_wind = \
            self.star_particles.new_channel_to(self.wind.particles)
        channel_to_wind.copy_attributes(to_wind_attributes)

    def sync_from_star_code(self):
        """
        copy relevant attributes changed by stellar gravity
        """
        from_stellar_gravity_attributes = [
            "x", "y", "z", "vx", "vy", "vz"
        ]
        from_sink_attributes = [
            "x", "y", "z", "vx", "vy", "vz", "mass",
        ]
        channel_from_star_dyn = \
            self.star_code.particles.new_channel_to(self.star_particles)
        channel_from_star_dyn.copy_attributes(
            from_stellar_gravity_attributes
        )
        channel_from_sinks = \
            self.sink_code.particles.new_channel_to(
                self.sink_particles)
        if not self.sink_particles.is_empty():
            channel_from_sinks.copy_attributes(
                from_sink_attributes,
            )

    def sync_to_star_code(self):
        """
        copy potentially modified attributes that matter to stellar gravity
        """
        to_stellar_gravity_attributes = [
            "x", "y", "z", "vx", "vy", "vz", "mass", "radius",
        ]
        channel_to_star_dyn = \
            self.star_particles.new_channel_to(self.star_code.particles)
        if not self.star_particles.is_empty():
            channel_to_star_dyn.copy_attributes(
                to_stellar_gravity_attributes
            )
        channel_to_sinks = \
            self.sink_particles.new_channel_to(
                self.sink_code.particles
            )
        if not self.sink_particles.is_empty():
            # print(self.sink_particles)
            # print(self.sink_code.particles)
            channel_to_sinks.copy_attributes(
                to_stellar_gravity_attributes
            )

    @property
    def particles(self):
        "Return all particles"
        return ParticlesSuperset(
            self.star_particles,
            self.dm_particles,
            self.sink_particles,
            self.gas_particles,
        )

    @property
    def code(self):
        "Return the main code - the Bridge system in this case"
        return self.system

    def resolve_sinks(
            self,
            density_override_factor=10,
    ):
        "Identify high-density gas, and form sink(s) when needed"
        settings = self.settings
        dump_saved = False
        removed_gas = Particles()
        mass_initial = (
            self.gas_particles.total_mass()
            + (
                self.sink_particles.total_mass()
                if not self.sink_particles.is_empty()
                else (0 | units.MSun)
            )
            + (
                self.star_particles.total_mass()
                if not self.star_particles.is_empty()
                else (0 | units.MSun)
            )
        )
        maximum_density = (
            self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select_array(
            lambda density: density > maximum_density,
            ["density"],
        ).copy().sorted_by_attribute("density").reversed()

        current_max_density = self.gas_particles.density.max()
        print(
            "Max gas density: %s (%.3f critical)"
            % (current_max_density, current_max_density/maximum_density),
        )
        print(
            "Number of gas particles above maximum density (%s): %i" % (
                maximum_density.in_(units.g * units.cm**-3),
                len(high_density_gas),
            )
        )
        self.logger.info(
            "Number of gas particles above maximum density (%s): %i",
            maximum_density.in_(units.g * units.cm**-3),
            len(high_density_gas),
        )
        new_sinks = Particles()
        while not high_density_gas.is_empty():
            i = 0
            origin_gas = high_density_gas[i]
            form_sink = False

            if origin_gas in removed_gas:
                high_density_gas.remove_particle(origin_gas)
                continue
            if (
                origin_gas.density/maximum_density
                > density_override_factor
            ):
                print(
                    "Sink formation override: "
                    "gas density is %s (> %s), forming sink" % (
                        origin_gas.density.in_(units.g * units.cm**-3),
                        (
                            density_override_factor
                            * maximum_density
                        ).in_(units.g * units.cm**-3),
                    )
                )
                self.logger.info(
                    "Sink formation override: gas density is %s (> %s), "
                    "forming sink",
                    origin_gas.density.in_(units.g * units.cm**-3),
                    (
                        density_override_factor
                        * maximum_density
                    ).in_(units.g * units.cm**-3),
                )
                form_sink = True
            else:
                try:
                    form_sink, not_forming_message = should_a_sink_form(
                        origin_gas.as_set(), self.gas_particles,
                        # check_thermal=self.isothermal_mode,
                        accretion_radius=settings.minimum_sink_radius,
                    )
                    form_sink = form_sink[0]
                except TypeError as te:
                    print(te)
                    print(origin_gas)
                    form_sink = False
                    not_forming_message = \
                        "Something went wrong (TypeError)"
                    dump_file_id = numpy.random.randint(100000000)
                    key = origin_gas.key
                    dumpfile = "dump-%08i-stars-key%i.hdf5" % (
                        dump_file_id,
                        key,
                    )
                    if not dump_saved:
                        print("saving gas dump to %s" % dumpfile)
                        write_set_to_file(
                            self.gas_particles, dumpfile, "amuse",
                        )
                        dump_saved = True
                    exit()
            if not form_sink:
                self.logger.info(
                    "Not forming a sink at t= %s - not meeting the"
                    " requirements: %s",
                    self.physical_time,
                    not_forming_message,
                )
                high_density_gas.remove_particle(origin_gas)
            else:  # try:
                print("Forming a sink from particle %s" % origin_gas.key)

                # NOTE: we create a new sink core *without*
                # removing/accreting the gas seed!  This is because it is
                # accreted later on and we don't want to duplicate its
                # accretion.
                new_sink = Particle()
                new_sink.sink_number = (
                    0 if self.sink_particles.is_empty()
                    else max(self.sink_particles.sink_number)+1
                )
                new_sink.birth_time = self.physical_time
                new_sink.initialised = False

                # Should do accretion here, and calculate the radius from
                # the average density, stopping when the jeans radius
                # becomes larger than the accretion radius!

                # Setting the radius to something that will lead to
                # >~150MSun per sink would be ideal.  So this radius is
                # related to the density.
                minimum_sink_radius = settings.minimum_sink_radius
                # desired_sink_mass = settings.desired_sink_mass
                # desired_sink_radius = max(
                #     (desired_sink_mass / origin_gas.density)**(1/3),
                #     minimum_sink_radius,
                # )

                # new_sink.radius = desired_sink_radius
                new_sink.radius = minimum_sink_radius
                # Average of accreted gas is better but for isothermal this
                # is fine
                # new_sink.u = origin_gas.u

                # NOTE: this gets overwritten when gas is accreted, so
                # probably this is misleading code...
                if self.isothermal_mode:
                    new_sink.u = temperature_to_u(
                        settings.isothermal_gas_temperature
                    )
                else:
                    new_sink.u = origin_gas.u
                # new_sink.u = 0 | units.kms**2

                new_sink.accreted_mass = 0 | units.MSun
                o_x, o_y, o_z = origin_gas.position

                new_sink.position = origin_gas.position
                new_sink.mass = origin_gas.mass  # 0 | units.MSun
                new_sink.velocity = origin_gas.velocity
                self.remove_gas(origin_gas.as_set())
                if origin_gas in high_density_gas:
                    high_density_gas.remove_particle(origin_gas)
                else:
                    print("already removed origin gas")
                new_sink = new_sink_particles(new_sink.as_set())
                # print(self.sink_particles)

                # Note: this *will* delete the accreted gas (all within
                # sink_radius)!
                print("accreting")
                # print(new_sink)
                accreted_gas = new_sink.accrete(self.gas_particles)
                print("done accreting")
                # new_sink.initial_density = origin_gas.density
                new_sink.initial_density = (
                    new_sink.mass
                    / (4/3 * numpy.pi * new_sink.radius**3)
                )

                if accreted_gas.is_empty():
                    self.logger.info("Empty gas so no sink")
                    exit()
                    high_density_gas.remove_particle(origin_gas)
                else:
                    self.logger.info(
                        "Number of accreted gas particles: %i",
                        len(accreted_gas)
                    )

                    # Track how much the sink accretes over time
                    new_sink.accreted_mass = new_sink.mass

                    # Sink's "internal energy" is the velocity dispersion
                    # of the infalling gas
                    new_sink.u = (
                        accreted_gas.velocity
                        - accreted_gas.center_of_mass_velocity()
                    ).lengths_squared().mean()

                    # Since accretion removes gas particles, we should
                    # remove them from the code too
                    # removed_gas.add_particles(accreted_gas.copy())
                    # self.remove_gas(accreted_gas)

                    # Which high-density gas particles are accreted and
                    # should no longer be considered?
                    accreted_high_density_gas = \
                        high_density_gas.get_intersecting_subset_in(
                            accreted_gas
                        )
                    high_density_gas.remove_particles(
                        accreted_high_density_gas
                    )
                    new_sinks.add_particle(new_sink)
                    self.logger.info(
                        "Added sink %i with mass %s and radius %s",
                        new_sink.sink_number,
                        new_sink.mass.in_(units.MSun),
                        new_sink.radius.in_(units.parsec),
                    )
                    # except:
                    #     print("Could not add another sink")
        self.add_sinks(new_sinks)
        self.gas_particles.synchronize_to(
            self.gas_code.gas_particles
        )
        del high_density_gas
        mass_final = (
            self.gas_particles.total_mass()
            + (
                self.sink_particles.total_mass()
                if not self.sink_particles.is_empty()
                else (0 | units.MSun)
            )
            + (
                self.star_particles.total_mass()
                if not self.star_particles.is_empty()
                else (0 | units.MSun)
            )
        )
        if abs(mass_initial - mass_final) >= self.gas_particles[0].mass:
            print("WARNING: mass is not conserved in sink formation!")
            self.logger.info(
                "WARNING: mass is not conserved in sink formation!"
            )

    def resolve_sink_accretion(self):
        # FIXME: this only makes "temporary" sinks - some attributes like
        # angular momentum may be lost afterwards
        sinks = new_sink_particles(
            self.sink_particles.sorted_by_attribute("mass").reversed()
        )  # Allowing them to accrete
        sink_mass_before_accretion = sinks.mass
        # accreted_gas = sinks.accrete(self.gas_particles)
        sink_mass_after_accretion = sinks.mass
        delta_mass = sink_mass_after_accretion - sink_mass_before_accretion
        sinks.accreted_mass += delta_mass

        # Need to keep the sink density the same i.e. expand radius if it's
        # been accreting!  Otherwise we'll form new stars in a VERY small
        # radius which is really bad...
        sinks.radius = (
            (sinks.mass / sinks.initial_density)
            / (4/3 * numpy.pi)
        )**(1/3)
        for i, sink in enumerate(sinks):
            self.logger.info(
                "Sink %i accreted %s, resizing to a radius of %s",
                sink.sink_number, delta_mass[i], sink.radius
            )
        # Remove accreted gas
        self.gas_particles.synchronize_to(self.gas_code.gas_particles)
        # Sync sinks
        self.sync_to_star_code()

    def add_sink(self, sink):
        self.sink_code.particles.add_particle(sink)
        # self.sink_particles.add_sink(sink)
        self.sink_particles.add_particle(sink)
        # self.sink_code.commit_particles()

    def add_sinks(self, sinks):
        if not sinks.is_empty():
            self.sink_code.particles.add_particles(sinks)
            # self.sink_code.parameters_to_default(star_code=settings.star_code)
            # self.sink_particles.add_sinks(sinks)
            self.sink_particles.add_particles(sinks)
            # self.sink_code.commit_particles()

    def remove_sinks(self, sinks):
        self.sink_code.particles.remove_particles(sinks)
        self.sink_particles.remove_particles(sinks)

    def add_gas(self, gas):
        self.gas_code.gas_particles.add_particles(gas)
        self.gas_particles.add_particles(gas)
        # print("Added gas - evolving for 10s")
        # self.gas_code.evolve_model(10 | units.s)
        # print("Done")

    def remove_gas(self, gas_):
        # Workaround: to prevent either from not being removed.
        gas = gas_.copy()
        self.gas_code.gas_particles.remove_particles(gas)
        self.gas_particles.remove_particles(gas)
        del gas

    def add_stars(self, stars):
        if not stars.is_empty():
            self.star_particles.add_particles(stars)
            self.evo_code_stars = self.evo_code.particles.add_particles(stars)
            self.sync_from_evo_code()
            self.star_code.particles.add_particles(stars)
            # self.star_code.parameters_to_default(star_code=settings.star_code)
            # self.star_code.commit_particles()
            self.wind.particles.add_particles(stars)

    def remove_stars(self, stars):
        self.star_code.particles.remove_particles(stars)
        self.evo_code.particles.remove_particles(stars)
        self.wind.particles.remove_particles(stars)
        self.star_particles.remove_particles(stars)

    def resolve_star_formation(
            self,
            stop_star_forming_time=10. | units.Myr,
            shrink_sinks=True,
            # shrink_sinks=False,
    ):
        settings = self.settings
        if self.physical_time >= stop_star_forming_time:
            self.logger.info(
                "No star formation since time > %s",
                stop_star_forming_time
            )
            return False
        self.logger.info("Resolving star formation")
        mass_before = (
            self.gas_particles.total_mass()
            + (
                self.sink_particles.total_mass()
                if not self.sink_particles.is_empty()
                else (0 | units.MSun)
            )
            + (
                self.star_particles.total_mass()
                if not self.star_particles.is_empty()
                else (0 | units.MSun)
            )
        )
        # max_new_stars_per_timestep = 500
        formed_stars = False
        stellar_mass_formed = 0 | units.MSun

        if settings.star_formation_method == "grouped":
            # Grouping of sinks here
            self.logger.info('Assigning groups to the sinks...')
            self.sink_particles = (
                self.sink_particles.sorted_by_attribute('mass')
            ).reversed()
            for i, sink in enumerate(self.sink_particles):
                sink = assign_sink_group(
                    sink,
                    self.sink_particles,
                    group_radius=settings.group_distance,
                    group_speed=(
                        settings.group_speed_mach
                        * self.gas_code.parameters.polyk.sqrt()
                    ),
                    group_age=settings.group_age,
                    logger=self.logger
                )
                self.logger.info(
                    "Sink %i in group #%i", sink.key, sink.in_group
                )

            # Form stars according to groups
            self.logger.info('Forming stars according to groups...')
            number_of_groups = self.sink_particles.in_group.max()
            for i in range(number_of_groups):
                i += 1   # Change to one-based index
                self.logger.info(
                    "Processing group %i (out of %i)",
                    i, number_of_groups
                )
                new_stars = form_stars_from_group(
                    group_index=i,
                    sink_particles=self.sink_particles,
                    local_sound_speed=self.gas_code.parameters.polyk.sqrt(),
                    logger=self.logger,
                    randomseed=numpy.random.randint(2**32-1),
                    shrink_sinks=shrink_sinks
                )

                if new_stars is not None:
                    formed_stars = True
                    self.add_stars(new_stars)
                    stellar_mass_formed += new_stars.total_mass()

        else:
            self.logger.info("Looping over %i sinks", len(self.sink_particles))

            multithread = False
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = []
                for i, sink in enumerate(self.sink_particles):
                    # TODO: this loop needs debugging/checking...
                    # self.logger.info("Processing sink %i, with mass %s",
                    # sink.sink_number, sink.mass)
                    self.logger.info("Processing sink %i", sink.key)
                    local_sound_speed = self.gas_code.parameters.polyk.sqrt()
                    if multithread:
                        results.append(
                            executor.submit(
                                form_stars,
                                sink,
                                local_sound_speed=local_sound_speed,
                                logger=self.logger,
                                randomseed=numpy.random.randint(2**32-1),
                            )
                        )
                    else:
                        results.append(
                            form_stars(
                                sink,
                                local_sound_speed=local_sound_speed,
                                logger=self.logger,
                                randomseed=numpy.random.randint(2**32-1),
                            )
                        )
                for sink, result in zip(self.sink_particles, results):
                    if multithread:
                        result = result.result()
                    sink.mass = result[0].mass
                    sink.next_primary_mass = result[0].next_primary_mass
                    sink.initialised = result[0].initialised
                    self.logger.info(
                        "Mass remaining in sink: %s - next star to form: %s",
                        sink.mass, sink.next_primary_mass
                    )
                    new_stars = result[1]
                    if not new_stars.is_empty():
                        formed_stars = True
                        new_stars.birth_time = self.physical_time
                        self.add_stars(new_stars)
                        stellar_mass_formed += new_stars.total_mass()

                    if shrink_sinks:
                        # After forming stars, shrink the sink's (accretion)
                        # radius to prevent it from accreting relatively far
                        # away gas and moving a lot
                        sink.radius = (
                            (sink.mass / sink.initial_density)
                            / (4/3 * numpy.pi)
                        )**(1/3)
                        sink.sink_radius = sink.radius
                    self.logger.info(
                        "Shrinking sink %i to a radius of %s",
                        sink.sink_number, sink.radius
                    )

        mass_after = (
            self.gas_particles.total_mass()
            + (
                self.sink_particles.total_mass()
                if not self.sink_particles.is_empty()
                else (0 | units.MSun)
            )
            + (
                self.star_particles.total_mass()
                if not self.star_particles.is_empty()
                else (0 | units.MSun)
            )
        )
        self.logger.debug(
            "dM = %s, mFormed = %s ",
            (mass_after - mass_before).in_(units.MSun),
            stellar_mass_formed.in_(units.MSun),
        )
        if abs(mass_before - mass_after) >= self.gas_particles[0].mass:
            print("WARNING: mass not conserved in star formation!")
            self.logger.info("WARNING: mass not conserved in star formation!")
        self.star_code.evolve_model(self.star_code.model_time)
        self.sync_to_star_code()
        return formed_stars

    def stellar_feedback(self):
        # Deliberately not doing anything here yet
        return

    def evolve_model(self, real_tend):
        #TODO
        "Evolve system to specified time"
        settings = self.settings
        relative_tend = real_tend - self.begin_time

        # dmax = self.gas_code.parameters.stopping_condition_maximum_density
        Myr = units.Myr

        self.logger.info(
            "Evolving to time %s",
            real_tend.in_(Myr),
        )
        # self.model_to_evo_code.copy()
        # self.model_to_gas_code.copy()

        density_limit_detection = \
            self.gas_code.stopping_conditions.density_limit_detection
        # density_limit_detection.enable()
        density_limit_detection.disable()

        # maximum_density = (
        #     self.gas_code.parameters.stopping_condition_maximum_density
        # )

        print("Starting loop")
        time_unit = units.Myr
        print(
            self.model_time.in_(time_unit),
            real_tend.in_(time_unit),
            self.system.timestep.in_(time_unit),
        )
        step = 0
        minimum_steps = 1
        while (
                (self.physical_time
                 < (real_tend - self.system.timestep))
                or (step < minimum_steps)
        ):
            step += 1
            print("Step %i" % step)
            # evo_time = self.evo_code.model_time
            # self.model_to_star_code.copy()
            if not self.star_particles.is_empty():
                evo_timestep = self.evo_code.particles.time_step.min()
                self.logger.info(
                    "Smallest evo timestep: %s", evo_timestep.in_(Myr)
                )
            # time = min(
            #     evo_time+evo_timestep,
            #     relative_tend,
            # )

            print("Evolving to %s" % real_tend.in_(Myr))
            self.logger.info("Evolving to %s", real_tend.in_(Myr))
            if not self.star_particles.is_empty():
                self.logger.info("Stellar evolution...")
                self.evo_code.evolve_model(real_tend)
                self.sync_from_evo_code()

            self.logger.info("System...")

            print("Evolving system")
            print("Gas state is ", self.gas_state)
            print(
                "Gas timestep = %s"
                % self.gas_code.parameters.time_step.in_(units.Myr)
            )

            if self.gas_state == "modified":
                print("Gas modified, smaller step!")
                # If new gas was added, as a precaution, we will switch to
                # shorter timesteps in the gas code.
                # Since we don't want to shorten the Bridge timestep, need to
                # be creative here...
                system_dt = self.system.timestep
                gas_dt = system_dt/2  # self.gas_code.parameters.time_step
                substeps = 2**16
                gas_small_dt = gas_dt / substeps
                self.gas_code.parameters.time_step = gas_small_dt

                self.sync_to_gas_code()
                # Bridge kick 1
                self.system.kick_codes(system_dt/2)

                # Drift gas with initially small and then increasingly long
                # timestep
                self.gas_code.evolve_model(
                    self.gas_code.model_time + gas_small_dt
                )
                while gas_dt - gas_small_dt > gas_dt / substeps:
                    self.gas_code.parameters.time_step = gas_small_dt
                    self.gas_code.evolve_model(
                        self.gas_code.model_time + gas_small_dt
                    )
                    gas_small_dt = 2*gas_small_dt

                # Bridge drift
                self.system.drift_codes(
                    self.system.time + self.system.timestep
                )
                self.system.channels.copy()
                self.system.time += self.system.timestep
                # Bridge kick 2
                self.system.kick_codes(system_dt/2)

                self.gas_code.parameters.time_step = gas_dt
                self.gas_state = "clean"
                print("Setting gas state to ", self.gas_state)

                # finally, continue
            else:
                self.sync_to_gas_code()
            self.logger.info("Pre system evolve")
            self.logger.info(
                "Stellar code is at time %s", self.star_code.model_time
            )
            star_code = getattr(available_codes, settings.star_code)
            self.star_code.parameters_to_default(star_code=star_code)
            self.system.evolve_model(relative_tend)
            self.logger.info("Post system evolve")
            self.logger.info(
                "Stellar code is at time %s", self.star_code.model_time
            )

            self.sync_from_gas_code()
            self.sync_from_star_code()

            if settings.use_wind and not self.star_particles.is_empty():
                self.sync_to_wind_code()
                self.wind.evolve_model(relative_tend)
                self.sync_from_wind_code()

            # if self.wind.has_new_wind_particles():
            #     wind_p = self.wind.create_wind_particles()
            if not self.wind_particles.is_empty():
                wind_p = self.wind_particles
                print(
                    "\n\nAdding %i wind particles with <T> %s\n\n"
                    % (len(wind_p), u_to_temperature(wind_p.u).mean())
                )
                self.logger.info(
                    "Adding %i wind particle(s) at t=%s, <T>=%s",
                    len(wind_p),
                    real_tend.in_(units.Myr),
                    u_to_temperature(wind_p.u).mean()
                )

                # This initial guess for h_smooth is used by Phantom to
                # determine the timestep bin this particle is in.
                # Small value: short timestep.
                # But also: too small value -> problem getting the density to
                # converge...
                # wind_p.h_smooth = 1 * (wind_p.mass/rhomax)**(1/3)
                wind_p.h_smooth = 0.1 | units.pc

                self.add_gas(wind_p)
                wind_p.remove_particles(wind_p)

            print("number of gas particles: %i" % len(self.gas_particles))
            if (
                    self.gas_code.model_time
                    < (relative_tend - self.system.timestep)
            ):
                print("******Evolving gas code a bit more")
                self.gas_code.evolve_model(relative_tend)
                self.sync_from_gas_code()
            print("Evolved system")

            if not self.sink_particles.is_empty():
                for i, sink in enumerate(self.sink_particles):
                    self.logger.info(
                        "sink %i's radius = %s, mass = %s",
                        sink.sink_number,
                        sink.radius.in_(units.parsec),
                        sink.mass.in_(units.MSun),
                    )
                print("Accreting gas")
                self.resolve_sink_accretion()
            if self.model_time < self.system.timestep:
                check_for_new_sinks = False
            else:
                check_for_new_sinks = True

            if check_for_new_sinks:
                print("Checking for new sinks")
                n_sink = len(self.sink_particles)
                self.resolve_sinks()
                if len(self.sink_particles) == n_sink:
                    self.logger.info("No new sinks")
                    # check_for_new_sinks = False
                else:
                    self.logger.info("New sink formed")

            if not self.sink_particles.is_empty():
                print("Forming stars")
                formed_stars = self.resolve_star_formation()
                if formed_stars and self.star_code is available_codes.ph4:
                    self.star_code.zero_step_mode = True
                    self.star_code.evolve_model(relative_tend)
                    self.star_code.zero_step_mode = False
                print("Formed stars")
                if not self.star_particles.is_empty():
                    self.logger.info(
                        "Average mass of stars: %s. "
                        "Average mass of sinks: %s.",
                        self.star_particles.mass.mean().in_(units.MSun),
                        self.sink_particles.mass.mean().in_(units.MSun),
                    )
                    self.logger.info(
                        "Total mass of stars: %s. "
                        "Total mass of sinks: %s.",
                        self.star_particles.total_mass().in_(units.MSun),
                        self.sink_particles.total_mass().in_(units.MSun),
                    )
            else:
                print("No sinks (yet?)")

            self.logger.info(
                "Now we have %i stars; %i sinks and %i gas, %i particles"
                " in total.",
                len(self.star_particles),
                len(self.sink_particles),
                len(self.gas_particles),
                (
                    len(self.gas_code.particles)
                    + len(self.star_code.particles)
                ),
            )

            self.logger.info(
                "Evo time is now %s",
                self.evo_code.model_time.in_(Myr)
            )
            self.logger.info(
                "Bridge time is now %s", (
                    (self.physical_time).in_(Myr)
                )
            )
            # self.evo_code_to_model.copy()
            # Check for stopping conditions

            self.feedback_timestep = self.timestep
            if not self.star_particles.is_empty():
                self.stellar_feedback()

            self.logger.info(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s",
                real_tend.in_(units.Myr),
                (self.physical_time).in_(Myr),
                (self.begin_time + self.gas_code.model_time).in_(Myr),
                (self.begin_time + self.star_code.model_time).in_(Myr),
                (self.evo_code.model_time).in_(Myr),
            )
            print(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s" % (
                    real_tend.in_(units.Myr),
                    (self.physical_time).in_(Myr),
                    (self.begin_time + self.gas_code.model_time).in_(Myr),
                    (self.begin_time + self.star_code.model_time).in_(Myr),
                    (self.evo_code.model_time).in_(Myr),
                )
            )

        # self.gas_code_to_model.copy()
        # self.star_code_to_model.copy()

    def get_gravity_at_point(self, *args, **kwargs):
        force = VectorQuantity(
            [0, 0, 0],
            unit=units.m * units.s**-2,
        )
        for parent in [self.star_code, self.gas_code, self.tidal_field]:
            if parent:
                force += parent.get_gravity_at_point(*args, **kwargs)
        return force

    @property
    def model_time(self):
        "Return time of the system since starting this run"
        return self.system.model_time

    @property
    def physical_time(self):
        "Return physical time of the system"
        return self.system.model_time + self.begin_time


def main(
        args, seed=22, have_stars=False, have_gas=False, have_sinks=False,
        nsteps=None, settings=ekster_settings.Settings()
):
    "Simulate an embedded star cluster (sph + dynamics + evolution)"
    from amuse.io import read_set_from_file
    from plotting_class import temperature_to_u
    from _version import version
    import signal

    if args.setup != "default":
        settings = ekster_settings.read_config(
            settings, args.settingfilename, args.setup,
        )

    def graceful_exit(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_exit)

    logger = logging.getLogger(__name__)

    if args.filename_gas is not None:
        filename_gas = args.filename_gas
    else:
        filename_gas = settings.filename_gas
    if args.filename_stars is not None:
        filename_stars = args.filename_stars
    else:
        filename_stars = settings.filename_stars
    if args.filename_sinks is not None:
        filename_sinks = args.filename_sinks
    else:
        filename_sinks = settings.filename_sinks
    if args.filename_random is not None:
        filename_random = args.filename_random
    else:
        filename_random = settings.filename_random
    if args.rundir is not None:
        rundir = args.rundir
    else:
        rundir = settings.rundir

    if not os.path.exists(rundir):
        os.makedirs(rundir)
    # TODO: get time stamp from gas, stars, or sinks
    # Default for the initial spiral gas is 1.4874E+15 seconds

    if filename_random is None or filename_random == "None":
        numpy.random.seed(seed)
    else:
        state_file = open(filename_random, 'rb')
        pickled_state = state_file.read()
        state_file.close()
        randomstate = pickle.loads(pickled_state)
        numpy.random.set_state(randomstate.get_state())
    run_prefix = rundir + "/"

    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    logging.basicConfig(
        filename="%sekster.log" % run_prefix,
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
    logger.info("git revision: %s", version())

    star_converter = nbody_system.nbody_to_si(
        # settings.star_mscale,
        settings.star_rscale,
        settings.timestep,
    )

    gas_converter = nbody_system.nbody_to_si(
        settings.gas_mscale,
        settings.gas_rscale,
    )

    if filename_stars is not None and filename_stars != "None":
        stars = read_set_from_file(filename_stars, "amuse", close_file=True,)
        have_stars = True
    else:
        from amuse.ext.masc import new_star_cluster
        stars = new_star_cluster(
            number_of_stars=10,
            initial_mass_function='kroupa',
            effective_radius=3.0 | units.parsec,
        )
        # stars.mass = stars.mass
        stars.birth_mass = stars.mass
        stars.birth_time = 0 | units.Myr  # Should be gas time??
        stars.x += 250 | units.pc
        stars.vx += 0.5 | units.kms
        # have_stars = False
    if filename_gas is not None and filename_gas != "None":
        print("reading gas")
        gas = read_set_from_file(filename_gas, "amuse", close_file=True,)
        if hasattr(gas, 'u'):
            if gas.u.in_base().unit is units.K:
                temp = gas.u
                del gas.u
                gas.u = temperature_to_u(temp)
        else:
            u = temperature_to_u(settings.isothermal_gas_temperature)
            gas.u = u
        # if hasattr(gas, 'pressure'):
        #     del gas.pressure

        print("Using %i particles" % len(gas))
        have_gas = True
        print(len(gas))
    else:
        from amuse.ext.molecular_cloud import molecular_cloud
        if not have_stars:
            stars = new_star_cluster(
                number_of_stars=2048,
                initial_mass_function='kroupa',
                lower_mass_limit=1.0 | units.MSun,
                effective_radius=2.0 | units.parsec,
            )
            stars.x += 4 | units.parsec
            stars.add_particles(new_star_cluster(
                number_of_stars=2048,
                initial_mass_function='kroupa',
                lower_mass_limit=1.0 | units.MSun,
                effective_radius=2.0 | units.parsec,
            ))
            stars.x -= 2 | units.pc
            stars.birth_mass = stars.mass
            stars.birth_time = 0 | units.Myr  # ??
            have_stars = True
        gas_density = 2e-18 | units.g * units.cm**-3
        increase_vol = 5
        Ngas = increase_vol**3 * 1000
        Mgas = increase_vol**3 * 1000 | units.MSun  # Mgas = Ngas | units.MSun
        volume = Mgas / gas_density  # 4/3 * pi * r**3
        radius = (volume / (units.pi * 4/3))**(1/3)
        radius = increase_vol * radius  # 15 | units.parsec
        gasconverter = nbody_system.nbody_to_si(Mgas, radius)
        gas = molecular_cloud(targetN=Ngas, convert_nbody=gasconverter).result
        gas.u = temperature_to_u(30 | units.K)
        have_gas = True

    if filename_sinks != "None" and filename_sinks is not None:
        sinks = read_set_from_file(filename_sinks, "amuse", close_file=True,)
        have_sinks = True

    if not (have_stars or have_gas or have_sinks):
        print("No particles!")
        exit()

    model = ClusterInPotential(
        stars=stars if have_stars else Particles(),
        gas=gas if have_gas else Particles(),
        sinks=sinks if have_sinks else Particles(),
        star_converter=star_converter,
        gas_converter=gas_converter,
        settings=settings,
    )
    model.sync_from_gas_code()

    timestep = settings.timestep
    starting_step = settings.step + 1
    print("Forming sinks")
    if nsteps is None:
        nsteps = 2000
    for step in range(starting_step, nsteps):
        time_unit = units.Myr
        print(
            "PT: %s MT: %s HT: %s GT: %s ET: %s" % (
                model.physical_time.in_(time_unit),
                model.model_time.in_(time_unit),
                model.gas_code.model_time.in_(time_unit),
                model.star_code.model_time.in_(time_unit),
                model.evo_code.model_time.in_(time_unit),
            )
        )
        time = (1+step) * timestep
        while model.physical_time < time - (1 | units.day):
            model.evolve_model(time)
        print(
            "Evolved to %s" % model.physical_time.in_(time_unit)
        )
        print(
            "Number of particles - gas: %i sinks: %i stars: %i" % (
                len(model.gas_particles),
                len(model.sink_particles),
                len(model.star_particles),
            )
        )
        if not model.sink_particles.is_empty():
            print(
                "Most massive sink: %s" % (
                    model.sink_particles.mass.max().in_(units.MSun),
                )
            )
            print(
                "Sinks centre of mass: %s" % (
                    model.sink_particles.center_of_mass().in_(units.parsec),
                )
            )
        if not model.star_particles.is_empty():
            print(
                "Most massive star: %s" % (
                    model.star_particles.mass.max().in_(units.MSun),
                )
            )
            print(
                "Stars centre of mass: %s" % (
                    model.star_particles.center_of_mass().in_(units.parsec),
                )
            )
        print(
            "Gas centre of mass: %s" % (
                model.gas_particles.center_of_mass().in_(units.parsec),
            )
        )
        dmax_now = model.gas_particles.density.max()
        dmax_stop = \
            model.gas_code.parameters.stopping_condition_maximum_density
        print(
            "Maximum density / stopping density = %s" % (
                dmax_now
                / dmax_stop,
            )
        )
        logger.info("Max density: %s", dmax_now.in_(units.g * units.cm**-3))
        logger.info(
            "Max density / sink formation density: %s", dmax_now/dmax_stop
        )

        logger.info("Making plot")
        print("Creating plot")
        mtot = model.gas_particles.total_mass()
        com = mtot * model.gas_particles.center_of_mass()
        if not model.sink_particles.is_empty():
            mtot += model.sink_particles.total_mass()
            com += (
                model.sink_particles.total_mass()
                * model.sink_particles.center_of_mass()
            )
        if not model.star_particles.is_empty():
            mtot += model.star_particles.total_mass()
            com += (
                model.star_particles.total_mass()
                * model.star_particles.center_of_mass()
            )
        com = com / mtot
        print("Centre of gas mass: %s" % com)
        if not model.star_particles.is_empty():
            print(
                "Centre of stars mass: %s"
                % model.star_particles.center_of_mass()
            )
        plotname = "%sdensity-%04i.png" % (run_prefix, step)
        offset_x_index = ["x", "y", "z"].index(settings.plot_xaxis)
        offset_y_index = ["x", "y", "z"].index(settings.plot_yaxis)
        offset_z_index = ["x", "y", "z"].index(settings.plot_zaxis)
        plot_hydro_and_stars(
            model.physical_time,
            stars=model.star_particles,
            sinks=model.sink_particles,
            gas=model.gas_particles,
            filename=plotname,
            title="time = %06.2f %s" % (
                model.physical_time.value_in(units.Myr),
                units.Myr,
            ),
            x_axis=settings.plot_xaxis,
            y_axis=settings.plot_yaxis,
            z_axis=settings.plot_zaxis,
            offset_x=com[offset_x_index],
            offset_y=com[offset_y_index],
            offset_z=com[offset_z_index],
            gasproperties=["density", ],
            # stars_are_sinks=True,
            # stars_are_sinks=False,
            # alpha_sfe=model.alpha_sfe,
            settings=settings,
        )

        if settings.write_backups:
            if not model.gas_particles.is_empty():
                settings.filename_gas = "%sgas-%04i.hdf5" % (run_prefix, step)
            if not model.star_particles.is_empty():
                settings.filename_stars = "%sstars-%04i.hdf5" % (
                    run_prefix, step)
            if not model.sink_particles.is_empty():
                settings.filename_sinks = "%ssinks-%04i.hdf5" % (
                    run_prefix, step)
            settings.filename_random = \
                "%srandomstate-%04i.pkl" % (run_prefix, step)
            settings.step = step
            settings.begin_time = model.begin_time
            settings.model_time = model.model_time
            settings.physical_time = model.physical_time
            ekster_settings.write_config(
                settings, "resume-%04i.ini" % step, "resume"
            )
            logger.info("Writing snapshots")
            print("Writing snapshots")
            randomstate = numpy.random.RandomState()
            pickled_random_state = pickle.dumps(randomstate)
            state_file = open(
                settings.filename_random,
                "wb"
            )
            state_file.write(pickled_random_state)
            state_file.close()
            if not model.gas_particles.is_empty():
                model.gas_particles.collection_attributes.timestamp = (
                    model.gas_code.model_time + model.begin_time
                )
                write_set_to_file(
                    model.gas_particles,
                    settings.filename_gas,
                    "amuse",
                    append_to_file=False,
                    # version='2.0',
                    # return_working_copy=False,
                    close_file=True,
                )
            if not model.sink_particles.is_empty():
                model.sink_particles.collection_attributes.timestamp = (
                    model.gas_code.model_time + model.begin_time
                )
                write_set_to_file(
                    model.sink_particles,
                    settings.filename_sinks,
                    "amuse",
                    append_to_file=False,
                    # version='2.0',
                    # return_working_copy=False,
                    close_file=True,
                )
            if not model.star_particles.is_empty():
                model.star_particles.collection_attributes.timestamp = (
                    model.star_code.model_time + model.begin_time
                )
                write_set_to_file(
                    model.star_particles,
                    settings.filename_stars,
                    "amuse",
                    append_to_file=False,
                    # version='2.0',
                    # return_working_copy=False,
                    close_file=True,
                )
    return model


if __name__ == "__main__":
    settings = ekster_settings.Settings()
    args = new_argument_parser(settings)
    if args.writesetup:
        ekster_settings.write_config(
            settings, args.settingfilename, args.setup
        )
        exit()
    model = main(args, settings=settings)
