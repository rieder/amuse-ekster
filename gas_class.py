#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Class for a gas system"
import logging
import numpy

from amuse.community.fi.interface import Fi
from amuse.community.phantom.interface import Phantom
# from amuse.datamodel import Particles, Particle
from amuse.units import units, nbody_system, constants

from basic_class import BasicCode
from cooling_class import SimplifiedThermalModelEvolver
# from plotting_class import plot_hydro_and_stars
# from sinks_class import accrete_gas  # , SinkParticles
# from amuse.ext.sink import SinkParticles

import default_settings


def sfe_to_density(e_loc, alpha=0.02):
    "Calculate density needed for specified star formation efficiency"
    rho = 100 * (e_loc / alpha)**2 | units.MSun * units.parsec**-3
    return rho


def density_to_sfe(rho, alpha=0.02):
    "Calculate star formation efficiency for specified density"
    sfe = alpha * (rho.value_in(100 * units.MSun * units.parsec**-3))**0.5
    return sfe


class GasCode(BasicCode):
    """Wraps around gas code, supports star formation and cooling"""

    def __init__(
            self,
            sph_code=Phantom,
            # sph_code=Fi,
            converter=None,
            logger=None,
            internal_star_formation=False,
            # cooling_type="thermal_model",
            cooling_type="default",
            begin_time=0.0 | units.Myr,
            **keyword_arguments
    ):
        self.typestr = "Hydro"
        # self.namestr = sph_code.__name__
        self.__name__ = "GasCode"
        self.logger = logger or logging.getLogger(__name__)
        self.internal_star_formation = internal_star_formation
        if converter is not None:
            self.unit_converter = converter
        else:
            self.unit_converter = nbody_system.nbody_to_si(
                default_settings.gas_mscale,
                default_settings.gas_rscale,
            )
        if begin_time is None:
            begin_time = 0. | units.Myr
        self.__begin_time = self.unit_converter.to_si(begin_time)

        self.cooling_type = cooling_type

        self.epsilon = default_settings.epsilon_gas
        # self.density_threshold = (5e-20 | units.g * units.cm**-3)
        # self.density_threshold = (5e5 | units.amu * units.cm**-3)
        self.density_threshold = default_settings.density_threshold
        print(
            "Density threshold for sink formation: %s (%s / %s)" % (
                self.density_threshold.in_(units.MSun * units.parsec**-3),
                self.density_threshold.in_(units.g * units.cm**-3),
                self.density_threshold.in_(units.amu * units.cm**-3),
            )
        )
        self.logger.info(
            "Density threshold for sink formation: %s (%s / %s)",
            self.density_threshold.in_(units.MSun * units.parsec**-3),
            self.density_threshold.in_(units.g * units.cm**-3),
            self.density_threshold.in_(units.amu * units.cm**-3),
        )
        # self.density_threshold = (1 | units.MSun) / (self.epsilon)**3
        self.code = sph_code(
            # self.unit_converter if sph_code is not Phantom else None,
            self.unit_converter,
            redirection="none",
            **keyword_arguments
        )
        self.parameters = self.code.parameters
        if sph_code is Fi:
            self.parameters.use_hydro_flag = True
            self.parameters.self_gravity_flag = True
            # Maybe make these depend on the converter?
            self.parameters.periodic_box_size = \
                100 * default_settings.gas_rscale
            self.parameters.timestep = default_settings.timestep * 0.5
            self.parameters.verbosity = 0
            self.parameters.integrate_entropy_flag = False
            self.parameters.stopping_condition_maximum_density = \
                self.density_threshold
        elif sph_code is Phantom:
            self.parameters.alpha = default_settings.alpha
            # self.parameters.gamma = 5./3.
            self.parameters.gamma = default_settings.gamma
            self.parameters.ieos = default_settings.ieos
            self.parameters.icooling = default_settings.icooling
            mu = self.parameters.mu  # mean molecular weight
            temperature = 10 | units.K
            polyk = (
                constants.kB
                * temperature
                / mu
            )
            self.parameters.polyk = polyk
            self.parameters.rho_crit = 20*self.density_threshold
            self.parameters.stopping_condition_maximum_density = \
                self.density_threshold
            self.parameters.h_soft_sinkgas = default_settings.epsilon_gas
            self.parameters.h_soft_sinksink = default_settings.epsilon_gas
            self.parameters.h_acc = 0.01 | units.parsec

        if self.cooling_type == "thermal_model":
            if sph_code is Fi:
                # Have to do our own cooling
                # self.parameters.isothermal_flag = True
                # self.parameters.gamma = 1
                self.parameters.isothermal_flag = False
                self.parameters.radiation_flag = False
                self.parameters.gamma = 5./3.
            # elif sph_code is Phantom:
                # self.parameters.ieos = "adiabatic"
            self.cooling = SimplifiedThermalModelEvolver(
                self.gas_particles
            )
            self.cooling.model_time = self.model_time
        elif self.cooling_type == "default":
            self.cooling = False
            if sph_code is Fi:
                self.parameters.gamma = 1.
                self.parameters.isothermal_flag = True
                self.parameters.radiation_flag = False

        # self.get_gravity_at_point = self.code.get_gravity_at_point
        # self.get_potential_at_point = self.code.get_potential_at_point
        # self.get_hydro_state_at_point = self.code.get_hydro_state_at_point

    # @property
    def get_potential_at_point(self, eps, x, y, z, **keyword_arguments):
        """Return potential at specified point"""
        return self.code.get_potential_at_point(
            eps, x, y, z, **keyword_arguments
        )

    def get_gravity_at_point(self, eps, x, y, z, **keyword_arguments):
        """Return gravity at specified point"""
        return self.code.get_gravity_at_point(
            eps, x, y, z, **keyword_arguments
        )

    def get_hydro_state_at_point(self, eps, x, y, z, **keyword_arguments):
        """Return hydro state at specified point"""
        return self.code.get_hydro_state_at_point(
            eps, x, y, z, **keyword_arguments
        )

    @property
    def model_time(self):
        """Return the current time"""
        return self.code.model_time + self.__begin_time

    @property
    def stopping_conditions(self):
        """Return stopping conditions"""
        return self.code.stopping_conditions

    @property
    def gas_particles(self):
        """Return all gas particles"""
        return self.code.gas_particles

    @property
    def dm_particles(self):
        """Return all dm particles"""
        return self.code.dm_particles

    @property
    def sink_particles(self):
        """Return all sink particles"""
        return self.code.sink_particles
        # return self.code.dm_particles

    @property
    def particles(self):
        """Return all particles"""
        return self.code.particles

    # @property
    def stop(self):
        """Stop the simulation code"""
        return self.code.stop

    def evolve_model(self, real_end_time):
        """
        Evolve model, and manage these stopping conditions:
        - density limit detection
        --> form stars when this happens

        Use the code time step to advance until 'end_time' is reached.
        Each step, also do cooling (if enabled).

        - returns immediately if the code would not evolve a step, i.e. when
          (end_time - model_time) is smaller than half a code timestep (for
          Fi). Because we don't want to do cooling then either!
        """
        end_time = real_end_time - self.__begin_time
        time_unit = real_end_time.unit
        print("Evolve gas until %s" % end_time.in_(time_unit))

        # if code_name is Fi:
        # timestep = 0.005 | units.Myr  # self.code.parameters.timestep
        timestep = default_settings.timestep * 0.5
        # if self.code.model_time >= (end_time - timestep/2):
        #     return
        # if code_name is something_else:
        # some_other_condition

        density_limit_detection = \
            self.code.stopping_conditions.density_limit_detection
        if self.internal_star_formation:
            density_limit_detection.enable()

        # Do cooling with a leapfrog scheme
        first = True
        if self.cooling and first:
            self.cooling.evolve_for(timestep/2)
            first = False

        # short offset
        while self.code.model_time < (end_time - timestep*0.0001):
            if self.cooling and not first:
                self.cooling.evolve_for(timestep)
            next_time = self.code.model_time + timestep
            # temp = self.code.gas_particles[0].u
            print(
                "Calling evolve_model of code (timestep: %s end_time: %s" % (
                    timestep.in_(units.Myr), next_time.in_(units.Myr),
                )
            )
            self.code.evolve_model(next_time)
            print("evolve_model of code is done")
            if density_limit_detection.is_set():
                # We don't want to do star formation in this bit of code (since
                # we can't tell other codes about it). But we must make sure to
                # synchronise the cooling up to this point before we return,
                # otherwise we will cool too much (or too little)...
                # So we should probably:
                # - set the time step short enough to make this not a problem
                # - Finish our loop here for this timestep by doing a final
                # half-timestep cooling
                # NOTE: make sure code is not doing sub-'timestep' steps or
                # this will not work as expected!
                if self.cooling:
                    self.cooling.evolve_for(timestep/2)
                return
            # while density_limit_detection.is_set():
            #     self.resolve_starformation()
            #     self.code.evolve_model(next_time)
        # Make sure cooling and the code are synchronised when the loop ends
        if self.cooling:
            self.cooling.evolve_for(timestep/2)

        return

    def resolve_starformation(self):
        """
        Form stars from gas denser than 'density_threshold'.
        Stars are added to the 'dm_particles' particleset in the code.
        For use with an external stellar dynamics code: todo.
        """
        high_density_gas = self.gas_particles.select_array(
            lambda rho: rho > self.density_threshold,
            ["rho"],
        )
        # Other selections?
        new_stars = high_density_gas.copy()
        self.gas_particles.remove_particles(high_density_gas)
        self.logger.info("Removed %i former gas particles", len(new_stars))

        new_stars.birth_age = self.model_time
        self.dm_particles.add_particles(new_stars)
        self.logger.info("Added %i new star particles", len(new_stars))

        self.logger.info("Resolved star formation")

    @property
    def potential_energy(self):
        """Return the potential energy in the code"""
        return self.code.potential_energy

    @property
    def kinetic_energy(self):
        """Return the kinetic energy in the code"""
        return self.code.kinetic_energy

    @property
    def thermal_energy(self):
        """Return the thermal energy in the code"""
        return self.code.thermal_energy


def main():
    "Test class with a molecular cloud"
    from amuse.ext.molecular_cloud import molecular_cloud
    converter = nbody_system.nbody_to_si(
        100000 | units.MSun,
        10.0 | units.parsec,
    )
    numpy.random.seed(11)
    temperature = 30 | units.K
    from plotting_class import temperature_to_u
    u = temperature_to_u(temperature)
    gas = molecular_cloud(targetN=200000, convert_nbody=converter).result
    gas.u = u
    print("We have %i gas particles" % (len(gas)))

    # gastwo = molecular_cloud(targetN=100000, convert_nbody=converter).result
    # gastwo.u = u

    # gastwo.x += 12 | units.parsec
    # gastwo.y += 3 | units.parsec
    # gastwo.z -= 1 | units.parsec
    # gastwo.vx -= ((5 | units.parsec) / (1 | units.Myr))

    # gas.add_particles(gastwo)
    print("Number of gas particles: %i" % (len(gas)))

    model = GasCode(
        converter=converter,
        # internal_cooling=False,
    )
    model.gas_particles.add_particles(gas)
    print(model.parameters)
    # print(model.gas_code.gas_particles[0])
    # exit()
    timestep = 0.01 | units.Myr  # model.gas_code.parameters.timestep

    times = [] | units.Myr
    # kinetic_energies = [] | units.J
    # potential_energies = [] | units.J
    # thermal_energies = [] | units.J
    time = 0 | units.Myr
    step = 0
    while time < 0.3 | units.Myr:
        time += timestep
        print("Starting at %s, evolving to %s" % (
            model.model_time.in_(units.Myr),
            time.in_(units.Myr),
        )
              )
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time.in_(units.Myr))
        print(
            "Maximum density / stopping density = %s" % (
                model.gas_particles.density.max()
                / model.parameters.stopping_condition_maximum_density,
            )
        )
        if not model.sink_particles.is_empty():
            print(
                "Largest sink mass: %s" % (
                    model.sink_particles.mass.max().in_(units.MSun)
                )
            )

        times.append(time)
        step += 1


if __name__ == "__main__":
    main()
