#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for a star cluster embedded in a tidal field and a gaseous region
"""

import sys
import os
import logging
import argparse
import pickle
import signal
import concurrent.futures
import numpy

from amuse.datamodel import ParticlesSuperset, Particles, Particle
from amuse.units import units, nbody_system  # , constants
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits
from amuse.units.quantities import VectorQuantity

from amuse.support.console import set_preferred_units
from amuse.io import write_set_to_file
from amuse.io import read_set_from_file

from amuse.ext import stellar_wind
# from amuse.ext.masc import new_star_cluster
from amuse.ext.sink import new_sink_particles

from amuse.ext.ekster._version import version
from amuse.ext.ekster.gas_class import GasCode
FEEDBACK_ENABLED = False
if FEEDBACK_ENABLED:
    from amuse.ext.ekster.feedback_class import main_stellar_feedback
from amuse.ext.ekster.stellar_dynamics_class import StellarDynamicsCode
from amuse.ext.ekster.stellar_evolution_class import StellarEvolutionCode
from amuse.ext.ekster.sinks_class import should_a_sink_form  # , sfe_to_density
from amuse.ext.ekster.plotting_class import plot_hydro_and_stars  # , plot_stars
from amuse.ext.ekster.plotting_class import (
    u_to_temperature, temperature_to_u,
    gas_mean_molecular_weight,
)
from amuse.ext.ekster.star_forming_region_class import (
    form_stars,
    form_stars_from_group, assign_sink_group,
)
from amuse.ext.ekster.bridge import (
    Bridge, CalculateFieldForCodes,
)
from amuse.ext.ekster import ekster_settings
from amuse.ext.ekster import spiral_potential
from amuse.ext.ekster import available_codes


set_preferred_units(
    units.Myr, units.kms, units.pc, units.MSun, units.g * units.cm**-3
)


def new_argument_parser(settings):
    "Parse command line arguments"
    parser = argparse.ArgumentParser()
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
        # StarCluster,
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

        self.new_stars_added = False

        # FIXME because this is no longer true
        # This class stores particle data by itself, in contrast to StarCluster
        self.gas_state = "clean"
        self.gas_particles = Particles()
        self.dm_particles = Particles()
        self.sink_particles = Particles()
        self.star_particles = Particles()

        if star_converter is None:
            converter_for_stars = nbody_system.nbody_to_si(
                settings.star_rscale,
                settings.timestep_bridge,  # 1 nbody time == 1 timestep
            )
        else:
            converter_for_stars = star_converter

        if not gas.is_empty():
            if gas_converter is None:
                converter_for_gas = nbody_system.nbody_to_si(
                    settings.gas_mscale,
                    settings.gas_rscale,
                )
            else:
                converter_for_gas = gas_converter
            self.logger.info("Initialising Gas")

            # phantom_solarm = 1.9891e33 | units.g
            # phantom_pc = 3.086e18 | units.cm
            phantom_gg = 6.672041e-8 | units.cm**3 * units.g**-1 * units.s**-2
            phantom_mass = 1.0 | units.MSun
            phantom_time = 60 * 60 * 24 * 365.25 * 1e6 | units.s
            phantom_converter = ConvertBetweenGenericAndSiUnits(
                # Phantom uses CGS units internally, scaled with G=1
                # So we need to make sure we use those same units here...
                # Also, Phantom's value for G is not the same as AMUSE's...
                (phantom_time**2 * phantom_gg * phantom_mass)**(1/3),
                phantom_mass,  # 1.0 MSun
                phantom_time,  # 1 Julian Myr
            )
            self.isothermal_mode = False if settings.ieos != 1 else True
            try:
                gas_time_offset = gas.get_timestamp()
            except AttributeError:
                gas_time_offset = 0 | units.Myr
            if gas_time_offset is None:
                gas_time_offset = 0 | units.yr
            self.gas_code = GasCode(
                converter=phantom_converter,
                settings=settings,
                time_offset=gas_time_offset,
            )

            print("****Adding gas****")
            self.add_gas(gas)
            self.timestep = settings.timestep
            self.timestep_bridge = settings.timestep_bridge
            self.logger.info("Initialised Gas")

        if settings.wind_enabled:
            self.wind_particles = Particles()
            self.wind = stellar_wind.new_stellar_wind(
                sph_particle_mass=self.gas_particles.mass.min(),
                mode=settings.wind_type,
                timestep=self.timestep,
                r_max=settings.wind_r_max,
                derive_from_evolution=True,
                tag_gas_source=True,
                grid_type="regular",
                rotate=True,
                target_gas=self.wind_particles,
                # acceleration_function="rsquared",
            )
            self.wind.model_time = settings.model_time

        # We need to be careful here - stellar evolution needs to re-calculate
        # all the stars unfortunately...
        # self.add_stars(stars)
        try:
            stars_time_offset = stars.get_timestamp()
        except AttributeError:
            stars_time_offset = None
        if stars_time_offset is None:
            try:
                stars_time_offset = sinks.get_timestamp()
            except AttributeError:
                stars_time_offset = None
        if stars_time_offset is None:
            stars_time_offset = 0 | units.yr

        self.star_code = StellarDynamicsCode(
            converter=converter_for_stars,
            star_code=settings.star_code,
            logger=self.logger,
            settings=settings,
            redirection=settings.code_redirection,
            time_offset=stars_time_offset,
            stop_after_each_step=settings.stop_after_each_step,
        )
        self.sink_code = self.star_code
        self.star_code.parameters.epsilon_squared = \
            settings.epsilon_stars**2
        evo_code = getattr(available_codes, settings.evo_code)
        if not hasattr(stars, "birth_time"):
            stars.birth_time = 0 | units.Myr
            if not stars.is_empty():
                print("WARNING: stars did not have a birth time - using 0 Myr")
        self.evo_code = StellarEvolutionCode(
            evo_code=evo_code,
            logger=self.logger,
            settings=settings,
            # redirection=settings.code_redirection,
            time_offset=(
                stars.birth_time.min() if not stars.is_empty()
                else 0 | units.yr
            ),
        )

        if not stars.is_empty():
            self.star_particles.add_particles(stars)

            stars_with_original_mass = stars.copy()
            if not hasattr(stars_with_original_mass, "birth_mass"):
                self.logger.info(
                    "No birth mass recorded for stars, setting to current"
                    " mass"
                )
                stars_with_original_mass.birth_mass = \
                    stars_with_original_mass.mass
            else:
                stars_with_original_mass.mass = \
                    stars_with_original_mass.birth_mass
            if not hasattr(stars_with_original_mass, "birth_time"):
                self.logger.info(
                    "No birth time recorded for stars, setting to current"
                    " time"
                )
                stars_with_original_mass.birth_time = 0 | units.Myr
            epochs, indices = numpy.unique(
                stars_with_original_mass.birth_time,
                return_inverse=True,
            )

            for i, time in enumerate(epochs):
                if time != epochs[0]:
                    self.evo_code.evolve_model(time)
                self.evo_code.particles.add_particles(
                    stars_with_original_mass[indices == i],
                )
            self.evo_code.evolve_model(settings.model_time)
            self.evo_code_stars = self.evo_code.particles
            self.sync_from_evo_code()

            self.star_code.particles.add_particles(stars)
            if settings.wind_enabled:
                self.wind.particles.add_particles(stars)

        if not sinks.is_empty():
            self.add_sinks(sinks)

        if hasattr(spiral_potential, settings.tide):
            tidal_field = getattr(spiral_potential, settings.tide)
            self.logger.info("Using tidal field %s", settings.tide)
            self.tidal_field = tidal_field(
                t_start=settings.model_time + settings.tide_time_offset,
                spiral_type=settings.tide_spiral_type,
            )
            self.logger.info(
                "Initialised tidal field at time %s",
                settings.model_time + settings.tide_time_offset
            )
        else:
            self.tidal_field = False
            self.logger.info(
                "Not using a tidal field"
            )

        self.converter = converter_for_gas

        def new_field_tree_gravity_code(
                code=available_codes.Fi,
        ):
            "Create a new field tree code"
            print("Creating field tree code")
            result = code(
                self.converter,
                redirection="none",
                mode="openmp",
            )
            result.parameters.epsilon_squared = max(
                settings.epsilon_stars,
                settings.epsilon_gas,
            )**2
            # result.parameters.timestep = 0.5 * self.timestep
            return result

        def new_field_direct_gravity_code(
                code=available_codes.Fastkick,
        ):
            "Create a new field direct code"
            print("Creating field direct code")
            result = code(
                self.converter,
                # redirection="none",
                number_of_workers=8,
            )
            result.parameters.epsilon_squared = max(
                settings.epsilon_stars,
                settings.epsilon_gas,
            )**2
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
                verbose=True,
            )
            return result

        to_gas_kickers = []
        if (
            settings.wind_enabled
            and settings.wind_type == "accelerate"
        ):
            to_gas_kickers.append(self.wind)
        if self.tidal_field:
            to_gas_kickers.append(self.tidal_field)
        to_gas_kickers.append(
            # self.star_code,
            new_field_code(
                self.star_code,
                mode=settings.field_code_type,
            )
        )

        to_stars_kickers = []
        if self.tidal_field:
            to_stars_kickers.append(self.tidal_field)
        to_stars_kickers.append(
            # self.gas_code,
            new_field_code(
                self.gas_code,
                mode=settings.field_code_type,
            )
        )

        self.system = Bridge(
            timestep=(
                self.timestep_bridge
            ),
            # use_threading=True,
            use_threading=False,
        )
        self.system.time = settings.model_time
        self.system.add_system(
            self.star_code,
            partners=to_stars_kickers,
            do_sync=True,
            # zero_smoothing=True,  # for petar
        )
        self.system.add_system(
            self.gas_code,
            partners=to_gas_kickers,
            do_sync=True,
            h_smooth_is_eps=True,
        )
        self.gas_code.parameters.time_step = self.timestep_bridge

    def sync_from_gas_code(self):
        """
        copy relevant attributes changed by gas/sink/dm dynamics
        """
        from_gas_attributes = [
            "x", "y", "z", "vx", "vy", "vz",
            "density", "h_smooth",
        ]
        thermal_gas_attributes = [
            "u",  # "h2ratio", "hi_abundance", "proton_abundance",
            # "electron_abundance", "co_abundance",
        ]
        if not self.isothermal_mode:
            for attribute in thermal_gas_attributes:
                from_gas_attributes.append(attribute)
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
            "stellar_type",
        ]
        if self.settings.evo_stars_lose_mass or self.settings.wind_enabled:
            from_stellar_evolution_attributes.append("mass")
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
            [
                self.star_particles,
                self.dm_particles,
                self.sink_particles,
                self.gas_particles,
            ]
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
            settings.density_threshold
            # self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select_array(
            lambda density: density > maximum_density,
            ["density"],
        ).copy().sorted_by_attribute("density").reversed()

        current_max_density = self.gas_particles.density.max()
        print(
            f"Max gas density: {current_max_density} "
            f"({current_max_density/maximum_density:.3f} critical)"
        )
        print(
            "Number of gas particles above maximum density "
            f"({maximum_density.in_(units.g * units.cm**-3)}): "
            f"{len(high_density_gas)}"
        )
        self.logger.info(
            "Number of gas particles above maximum density (%s): %i",
            maximum_density.in_(units.g * units.cm**-3),
            len(high_density_gas),
        )
        new_sinks = Particles()
        while not high_density_gas.is_empty():
            i = 0
            print(f"Checking gas core {i} ({len(high_density_gas)} remain)")
            self.logger.info(
                "Checking gas core %i (%i remain)",
                i,
                len(high_density_gas),
            )
            origin_gas = high_density_gas[i]
            form_sink = False

            if origin_gas in removed_gas:
                high_density_gas.remove_particle(origin_gas)
                continue
            if (
                origin_gas.density/maximum_density
                > density_override_factor
            ):
                unit_density = units.g * units.cm**-3
                supercritical_density = (
                    density_override_factor * maximum_density
                )
                print(
                    "Sink formation override: "
                    f"gas density is {origin_gas.density.in_(unit_density)} "
                    f"(> {(supercritical_density).in_(unit_density)})"
                    ", forming sink"
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
                    form_sink, not_forming_message, neighbours = \
                        should_a_sink_form(
                            origin_gas.as_set(), self.gas_particles,
                            # check_thermal=self.isothermal_mode,
                            accretion_radius=settings.minimum_sink_radius,
                        )
                    form_sink = form_sink[0]
                except TypeError as type_error:
                    print(type_error)
                    print(origin_gas)
                    form_sink = False
                    not_forming_message = \
                        "Something went wrong (TypeError)"
                    dump_file_id = numpy.random.randint(100000000)
                    key = origin_gas.key
                    dumpfile = f"dump-{dump_file_id}-stars-key{key}.amuse"
                    if not dump_saved:
                        print(f"saving gas dump to {dumpfile}")
                        write_set_to_file(
                            self.gas_particles, dumpfile, "amuse",
                        )
                        dump_saved = True
                    sys.exit()
            if not form_sink:
                self.logger.info(
                    "Not forming a sink at t= %s - not meeting the"
                    " requirements: %s",
                    self.model_time,
                    not_forming_message,
                )
                high_density_neighbours = \
                    neighbours.get_intersecting_subset_in(high_density_gas)
                high_density_gas.remove_particle(origin_gas)
                high_density_gas.remove_particles(high_density_neighbours)
            else:  # try:
                print(f"Forming a sink from particle {origin_gas.key}")

                # NOTE: we create a new sink core *without*
                # removing/accreting the gas seed!  This is because it is
                # accreted later on and we don't want to duplicate its
                # accretion.
                new_sink = Particle()
                last_sink_number = 0
                if not self.sink_particles.is_empty():
                    last_sink_number = max(self.sink_particles.sink_number)
                if not new_sinks.is_empty():
                    last_sink_number = max(new_sinks.sink_number)

                setattr(new_sink, "sink_number", last_sink_number+1)
                setattr(new_sink, "birth_time", self.model_time)
                setattr(new_sink, "initialised", False)

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
                setattr(new_sink, "radius", minimum_sink_radius)
                # Average of accreted gas is better but for isothermal this
                # is fine
                # new_sink.u = origin_gas.u

                # NOTE: this gets overwritten when gas is accreted, so
                # probably this is misleading code...
                if self.isothermal_mode:
                    setattr(
                        new_sink, "u", temperature_to_u(
                            settings.isothermal_gas_temperature,
                            gmmw=gas_mean_molecular_weight(0.5),
                        )
                    )
                else:
                    setattr(
                        new_sink, "u", origin_gas.u
                    )
                # new_sink.u = 0 | units.kms**2

                setattr(new_sink, "accreted_mass", 0 | units.MSun)
                # o_x, o_y, o_z = origin_gas.position

                setattr(new_sink, "position", origin_gas.position)
                setattr(new_sink, "mass", origin_gas.mass)  # 0 | units.MSun
                setattr(new_sink, "velocity", origin_gas.velocity)
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

                try:
                    assert not accreted_gas.is_empty()
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
                except AssertionError:
                    self.logger.warning("No gas accreted??")
                    high_density_gas.remove_particle(origin_gas)
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
        self.gas_particles.add_particles(gas)
        self.gas_code.gas_particles.add_particles(gas)
        self.sync_from_gas_code()
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
            if self.settings.wind_enabled:
                self.wind.particles.add_particles(stars)

            # self.star_code.parameters_to_default(star_code=settings.star_code)
            # self.star_code.commit_particles()

    def remove_stars(self, stars):
        self.star_code.particles.remove_particles(stars)
        self.evo_code.particles.remove_particles(stars)
        if settings.wind_enabled:
            self.wind.particles.remove_particles(stars)
        self.star_particles.remove_particles(stars)

    def resolve_star_formation(
            self,
            stop_star_forming_time=10. | units.Myr,
            shrink_sinks=True,
            # shrink_sinks=False,
    ):
        settings = self.settings
        if self.model_time >= stop_star_forming_time:
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
                print(f"Group {i} of {number_of_groups}")
                self.logger.info(
                    "Processing group %i (out of %i)",
                    i, number_of_groups
                )
                new_stars = form_stars_from_group(
                    group_index=i,
                    sink_particles=self.sink_particles,
                    lower_mass_limit=settings.stars_lower_mass_limit,
                    upper_mass_limit=settings.stars_upper_mass_limit,
                    local_sound_speed=self.gas_code.parameters.polyk.sqrt(),
                    logger=self.logger,
                    randomseed=numpy.random.randint(2**32-1),
                    shrink_sinks=shrink_sinks,
                )

                if new_stars is not None:
                    formed_stars = True
                    new_stars.birth_time = self.model_time
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
                    # FIXME
                    # local_sound_speed = self.gas_code.parameters.polyk.sqrt()
                    local_sound_speed = 100 | units.kms
                    if multithread:
                        results.append(
                            executor.submit(
                                form_stars,
                                sink,
                                lower_mass_limit=(
                                    settings.stars_lower_mass_limit
                                ),
                                upper_mass_limit=(
                                    settings.stars_upper_mass_limit
                                ),
                                local_sound_speed=local_sound_speed,
                                logger=self.logger,
                                randomseed=numpy.random.randint(2**32-1),
                            )
                        )
                    else:
                        results.append(
                            form_stars(
                                sink,
                                lower_mass_limit=(
                                    settings.stars_lower_mass_limit
                                ),
                                upper_mass_limit=(
                                    settings.stars_upper_mass_limit
                                ),
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
                        new_stars.birth_time = self.model_time
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
        # self.gas_particles = test_stellar_feedback(
        #     gas_=self.gas_particles,
        #     stars=self.star_particles
        # )
        # print('Stellar feedback implemented.')

        if (
            FEEDBACK_ENABLED
            and self.settings.feedback_enabled
            and not self.star_particles.is_empty()
        ):

            # STARBENCH
            # self.star_particles.luminosity = (
            #     1e49 * 13.6 | units.eV * units.s**-1
            # )

            self.gas_particles = main_stellar_feedback(
                gas=self.gas_particles,
                stars_=self.star_particles,
                time=self.model_time,
                mass_cutoff=self.settings.feedback_mass_threshold,
                temp_range=[
                    self.settings.isothermal_gas_temperature.value_in(units.K),
                    20000
                ] | units.K,
                logger=self.logger,
                randomseed=numpy.random.randint(2**32-1),
            )

    def evolve_model(self, end_time):
        "Evolve system to specified time"
        time_unit = units.Myr
        settings = self.settings
        start_time = self.model_time
        timestep = self.system.timestep

        # dmax = self.gas_code.parameters.stopping_condition_maximum_density

        self.logger.info(
            "Evolving to time %s",
            end_time.in_(time_unit),
        )
        # self.model_to_evo_code.copy()
        # self.model_to_gas_code.copy()

        try:
            density_limit_detection = \
                self.gas_code.stopping_conditions.density_limit_detection
            # density_limit_detection.enable()
            density_limit_detection.disable()
        except:
            print("No support for density limit detection")

        # maximum_density = (
        #     self.gas_code.parameters.stopping_condition_maximum_density
        # )

        print("Starting loop")
        print(
            start_time.in_(time_unit),
            end_time.in_(time_unit),
            timestep.in_(time_unit),
        )
        substep = 0
        number_of_steps = int(
            0.5 +
            (end_time - start_time) / timestep
        )
        while substep < number_of_steps:
            substep += 1
            evolve_to_time = start_time + substep*timestep
            print("Substep %i/%i" % (substep, number_of_steps))
            if not self.star_particles.is_empty():
                evo_timestep = self.evo_code.particles.time_step.min()
                self.logger.info(
                    "Smallest evo timestep: %s", evo_timestep.in_(time_unit)
                )

            print("Evolving to %s" % evolve_to_time.in_(time_unit))
            self.logger.info("Evolving to %s", evolve_to_time.in_(time_unit))
            if not self.star_particles.is_empty():
                self.logger.info("Stellar evolution...")
                self.evo_code.evolve_model(evolve_to_time)
                self.sync_from_evo_code()
                self.sync_to_star_code()

            self.logger.info("System...")

            self.sync_to_gas_code()
            self.logger.info("Pre system evolve")
            self.logger.info(
                "Stellar code is at time %s", self.star_code.model_time
            )
            star_code = getattr(available_codes, settings.star_code)
            self.star_code.parameters_to_default(star_code=star_code)
            self.gas_code.parameters.time_step = self.system.timestep
            print(f"Gas code time step: {self.gas_code.parameters.time_step}")
            self.system.evolve_model(evolve_to_time)
            self.logger.info("Post system evolve")
            self.logger.info(
                "Stellar code is at time %s", self.star_code.model_time
            )

            self.sync_from_gas_code()
            self.sync_from_star_code()

            if settings.wind_enabled and not self.star_particles.is_empty():
                self.sync_to_wind_code()
                self.wind.evolve_model(evolve_to_time)
                self.sync_from_wind_code()

                wind_p = self.wind_particles
                if not wind_p.is_empty():
                    mean_wind_temperature = u_to_temperature(
                        wind_p.u,
                        gmmw=gas_mean_molecular_weight(0),
                    ).mean()
                    print(
                        f"\n\nAdding {len(wind_p)} wind particles with "
                        f"<T> {mean_wind_temperature}\n\n"
                    )
                    self.logger.info(
                        "Adding %i wind particle(s) at t=%s, <T>=%s",
                        len(wind_p),
                        self.wind.model_time.in_(units.Myr),
                        mean_wind_temperature
                    )

                    # This initial guess for h_smooth is used by Phantom to
                    # determine the timestep bin this particle is in.
                    # Small value: short timestep.
                    # But also: too small value -> problem getting the density
                    # to converge...
                    # wind_p.h_smooth = 1 * (wind_p.mass/rhomax)**(1/3)
                    wind_p.h_smooth = 0.1 | units.pc

                    self.add_gas(wind_p)
                    wind_p.remove_particles(wind_p)

                print(f"number of gas particles: {len(self.gas_particles)}")
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
                self.resolve_sinks(
                    density_override_factor=settings.density_override_factor,
                )
                if len(self.sink_particles) == n_sink:
                    self.logger.info("No new sinks")
                    # check_for_new_sinks = False
                else:
                    self.logger.info("New sink formed")

            if not self.sink_particles.is_empty():
                print("Forming stars")
                formed_stars = self.resolve_star_formation()
                if formed_stars and self.star_code is available_codes.Ph4:
                    self.star_code.zero_step_mode = True
                    self.star_code.evolve_model(evolve_to_time)
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
                self.evo_code.model_time.in_(time_unit)
            )
            self.logger.info(
                "Bridge time is now %s", (
                    (self.model_time).in_(time_unit)
                )
            )
            # self.evo_code_to_model.copy()
            # Check for stopping conditions

            self.feedback_timestep = self.timestep
            if not self.star_particles.is_empty():
                self.stellar_feedback()

            self.logger.info(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s",
                evolve_to_time.in_(time_unit),
                (self.model_time).in_(time_unit),
                (self.gas_code.model_time).in_(time_unit),
                (self.star_code.model_time).in_(time_unit),
                (self.evo_code.model_time).in_(time_unit),
            )
            print(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s" % (
                    evolve_to_time.in_(time_unit),
                    (self.model_time).in_(time_unit),
                    (self.gas_code.model_time).in_(time_unit),
                    (self.star_code.model_time).in_(time_unit),
                    (self.evo_code.model_time).in_(time_unit),
                )
            )

    def get_gravity_at_point(self, *args, **kwargs):
        force = VectorQuantity(
            [0, 0, 0],
            unit=units.m * units.s**-2,
        )
        for parent in [self.star_code, self.gas_code, self.tidal_field]:
            if parent:
                force += parent.get_gravity_at_point(*args, **kwargs)
        return force

    def get_potential_at_point(self, *args, **kwargs):
        potential = 0 | units.m**2 * units.s**2
        for parent in [self.star_code, self.gas_code, self.tidal_field]:
            if parent:
                potential += parent.get_potential_at_point(*args, **kwargs)
        return potential

    @property
    def time_model(self):
        "Return time of the system since starting this run"
        return self.system.model_time

    @property
    def model_time(self):
        return self.time_model

    def stop(self):
        self.star_code.stop()
        self.gas_code.stop()
        self.evo_code.stop()
        return


def main(
        args, seed=22,
        nsteps=None, settings=ekster_settings.Settings()
):
    "Simulate an embedded star cluster (sph + dynamics + evolution)"

    if args.setup != "default":
        settings = ekster_settings.read_config(
            settings, args.settingfilename, args.setup,
        )

    def graceful_exit(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_exit)

    logger = logging.getLogger(__name__)

    rundir = settings.rundir
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    run_prefix = rundir + "/"

    filename_random = settings.filename_random
    if filename_random is None or filename_random == "None":
        numpy.random.seed(seed)
    else:
        with open(filename_random, 'rb') as state_file:
            pickled_state = state_file.read()
        randomstate = pickle.loads(pickled_state)
        numpy.random.set_state(randomstate.get_state())

    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    logging.basicConfig(
        filename=f"{run_prefix}ekster.log",
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
    logger.info("git revision: %s", version())

    star_converter = nbody_system.nbody_to_si(
        settings.star_rscale,
        settings.timestep,
    )

    gas_converter = nbody_system.nbody_to_si(
        settings.gas_mscale,
        settings.gas_rscale,
    )

    filename_gas = settings.filename_gas
    if filename_gas is not None and filename_gas != "None":
        gas = read_set_from_file(filename_gas, "amuse", close_file=True,)
        if hasattr(gas, 'u'):
            if gas.u.in_base().unit is units.K:
                temp = gas.u
                del gas.u
                if hasattr(gas, 'h2ratio'):
                    h2ratio = gas.h2ratio
                else:
                    # Assume cold gas by default - could base this on
                    # temperature though!
                    h2ratio = 0.5
                gas.u = temperature_to_u(
                    temp,
                    gmmw=gas_mean_molecular_weight(h2ratio),
                )
        else:
            u = temperature_to_u(
                settings.isothermal_gas_temperature,
                gmmw=gas_mean_molecular_weight(0.5),
            )
            gas.u = u
    else:
        print(
            "No gas read, generating standard initial conditions instead"
        )
        from amuse.ext.molecular_cloud import molecular_cloud
        gas_density = 1e-20 | units.g * units.cm**-3
        increase_vol = 2
        Ngas = increase_vol**3 * 1000 * 5
        Mgas = increase_vol**3 * 1000 | units.MSun
        volume = Mgas / gas_density
        radius = (volume / (units.pi * 4/3))**(1/3)
        radius = increase_vol * radius
        gasconverter = nbody_system.nbody_to_si(Mgas, radius)
        gas = molecular_cloud(targetN=Ngas, convert_nbody=gasconverter).result
        gas.u = temperature_to_u(
            30 | units.K,
            gmmw=gas_mean_molecular_weight(0.5),
        )
        gas.collection_attributes.timestamp = 0 | units.yr

    have_sinks = False
    filename_sinks = settings.filename_sinks
    if filename_sinks != "None" and filename_sinks is not None:
        sinks = read_set_from_file(filename_sinks, "amuse", close_file=True,)
        have_sinks = True

    have_stars = False
    filename_stars = settings.filename_stars
    if filename_stars is not None and filename_stars != "None":
        stars = read_set_from_file(filename_stars, "amuse", close_file=True,)
        have_stars = True

    model = ClusterInPotential(
        gas=gas,
        sinks=sinks if have_sinks else Particles(),
        stars=stars if have_stars else Particles(),
        star_converter=star_converter,
        gas_converter=gas_converter,
        settings=settings,
    )
    model.sync_from_gas_code()

    timestep = settings.timestep
    starting_step = settings.step
    final_step = settings.number_of_steps
    for step in range(starting_step, final_step):
        time_unit = units.Myr
        print(
            f"MT: {model.model_time.in_(time_unit)} "
            f"HT: {model.gas_code.model_time.in_(time_unit)} "
            f"GT: {model.star_code.model_time.in_(time_unit)} "
            f"ET: {model.evo_code.model_time.in_(time_unit)}"
        )
        time = step * timestep

        model.evolve_model(time)

        print(
            f"Step {step:04d}: evolved to "
            f"{model.model_time.in_(time_unit)}"
        )
        print(
            f"Number of particles - gas: {len(model.gas_particles)} "
            f"sinks: {len(model.sink_particles)} "
            f"stars: {len(model.star_particles)} "
        )
        if not model.sink_particles.is_empty():
            print(
                "Most massive sink: "
                f"{model.sink_particles.mass.max().in_(units.MSun)}"
            )
            print(
                "Sinks centre of mass: "
                f"{model.sink_particles.center_of_mass().in_(units.parsec)}"
            )
        if not model.star_particles.is_empty():
            print(
                "Most massive star: "
                f"{model.star_particles.mass.max().in_(units.MSun)}"
            )
            print(
                "Stars centre of mass: "
                f"{model.star_particles.center_of_mass().in_(units.parsec)}"
            )
        print(
            "Gas centre of mass: "
            f"{model.gas_particles.center_of_mass().in_(units.parsec)}"
        )
        dmax_now = model.gas_particles.density.max()
        dmax_stop = \
            settings.density_threshold
        #    model.gas_code.parameters.stopping_condition_maximum_density
        print(
            f"Maximum density / stopping density = {dmax_now / dmax_stop}"
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
        print(f"Centre of gas mass: {com}")
        if not model.star_particles.is_empty():
            print(
                "Centre of stars mass: "
                f"{model.star_particles.center_of_mass()}"
            )
        offset_x_index = ["x", "y", "z"].index(settings.plot_xaxis)
        offset_y_index = ["x", "y", "z"].index(settings.plot_yaxis)
        offset_z_index = ["x", "y", "z"].index(settings.plot_zaxis)
        if settings.plot_density:
            plotname = f"{run_prefix}density-{step:04d}.png"
            plot_hydro_and_stars(
                model.model_time,
                stars=model.star_particles,
                sinks=model.sink_particles,
                gas=model.gas_particles,
                filename=plotname,
                title=(
                    f"time = {model.model_time.value_in(units.Myr):06.2f} "
                    f"{units.Myr}"
                ),
                x_axis=settings.plot_xaxis,
                y_axis=settings.plot_yaxis,
                z_axis=settings.plot_zaxis,
                offset_x=com[offset_x_index],
                offset_y=com[offset_y_index],
                offset_z=com[offset_z_index],
                gasproperties=["density", ],
                settings=settings,
            )
        if settings.plot_temperature and not settings.ieos == 1:
            plotname = f"{run_prefix}temperature-{step:04d}.png"
            plot_hydro_and_stars(
                model.model_time,
                stars=model.star_particles,
                sinks=model.sink_particles,
                gas=model.gas_particles,
                filename=plotname,
                title=(
                    f"time = {model.model_time.value_in(units.Myr):06.2f} "
                    f"{units.Myr}",
                ),
                x_axis=settings.plot_xaxis,
                y_axis=settings.plot_yaxis,
                z_axis=settings.plot_zaxis,
                offset_x=com[offset_x_index],
                offset_y=com[offset_y_index],
                offset_z=com[offset_z_index],
                gasproperties=["temperature", ],
                settings=settings,
            )

        if (
            settings.write_backups
            and (
                step != starting_step
                or starting_step == 0
            )
        ):
            if not model.gas_particles.is_empty():
                settings.filename_gas = f"{run_prefix}gas-{step:04d}.amuse"
            if not model.star_particles.is_empty():
                settings.filename_stars = f"{run_prefix}stars-{step:04d}.amuse"
            if not model.sink_particles.is_empty():
                settings.filename_sinks = f"{run_prefix}sinks-{step:04d}.amuse"
            settings.filename_random = \
                f"{run_prefix}randomstate-{step:04d}.pkl"
            settings.step = step
            settings.model_time = model.model_time
            ekster_settings.write_config(
                settings, f"{run_prefix}resume-{step:04d}.ini", "resume"
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
                    model.gas_code.model_time
                )
                write_set_to_file(
                    model.gas_particles,
                    settings.filename_gas,
                    "amuse",
                    append_to_file=False,
                    # version='2.0',
                    # return_working_copy=False,
                    compression="gzip",
                    close_file=True,
                )
            if not model.sink_particles.is_empty():
                model.sink_particles.collection_attributes.timestamp = (
                    model.sink_code.model_time
                )
                write_set_to_file(
                    model.sink_particles,
                    settings.filename_sinks,
                    "amuse",
                    append_to_file=False,
                    # version='2.0',
                    # return_working_copy=False,
                    compression="gzip",
                    close_file=True,
                )
            if not model.star_particles.is_empty():
                model.star_particles.collection_attributes.timestamp = (
                    model.star_code.model_time
                )
                write_set_to_file(
                    model.star_particles,
                    settings.filename_stars,
                    "amuse",
                    append_to_file=False,
                    # version='2.0',
                    # return_working_copy=False,
                    compression="gzip",
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
        sys.exit()
    model = main(args, settings=settings)
