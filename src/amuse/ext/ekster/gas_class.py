#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Class for a gas system"
import logging
import numpy

# from amuse.datamodel import Particles, Particle
from amuse.units import units, nbody_system, constants
from amuse.datamodel import Particles

from amuse.ext.ekster import available_codes
from amuse.ext.ekster.available_codes import Fi, Phantom, Mizuki
from amuse.ext.ekster.basic_class import BasicCode
# from ekster.plotting_class import plot_hydro_and_stars
# from ekster.sinks_class import accrete_gas  # , SinkParticles
# from amuse.ext.sink import SinkParticles


def sfe_to_density(e_loc, alpha=0.02):
    "Calculate density needed for specified star formation efficiency"
    rho = 100 * (e_loc / alpha)**2 | units.MSun * units.parsec**-3
    return rho


def density_to_sfe(rho, alpha=0.02):
    "Calculate star formation efficiency for specified density"
    sfe = alpha * (rho.value_in(100 * units.MSun * units.parsec**-3))**0.5
    return sfe


class GasCode(BasicCode):
    """Wraps around gas code"""

    def __init__(
            self,
            sph_code=Phantom,    
            # sph_code=Fi,
            converter=None,
            logger=None,
            internal_star_formation=False,
            time_offset=0.0 | units.Myr,
            settings=None,  # ekster_settings.Settings(),
            **keyword_arguments
    ):
        try:
            sph_code_name = settings.sph_code
            print(sph_code_name)
            sph_code = getattr(available_codes, sph_code_name)
        except:
            print("Using legacy sph_code value")
            exit()

        self.typestr = "Hydro"
        # self.namestr = sph_code.__name__
        self.__name__ = "GasCode"
        self.logger = logger or logging.getLogger(__name__)
        if settings is None:
            from ekster.ekster_settings import Settings
            settings = Settings()
            print("WARNING: using default settings!")
            logger.info("WARNING: using default settings!")
        self.internal_star_formation = internal_star_formation
        if converter is not None:
            self.unit_converter = converter
        else:
            self.unit_converter = nbody_system.nbody_to_si(
                settings.gas_mscale,
                settings.timestep_bridge,
            )
        if time_offset is None:
            time_offset = 0. | units.Myr
        self.__time_offset = self.unit_converter.to_si(time_offset)

        self.epsilon = settings.epsilon_gas
        self.density_threshold = settings.density_threshold
        print(
            "Density threshold for sink formation: %s (%s; %s)" % (
                self.density_threshold.in_(units.MSun * units.parsec**-3),
                self.density_threshold.in_(units.g * units.cm**-3),
                self.density_threshold.in_(units.amu * units.cm**-3),
            )
        )
        self.logger.info(
            "Density threshold for sink formation: %s (%s; %s)",
            self.density_threshold.in_(units.MSun * units.parsec**-3),
            self.density_threshold.in_(units.g * units.cm**-3),
            self.density_threshold.in_(units.amu * units.cm**-3),
        )
        self.code = sph_code(
            self.unit_converter,
            redirection=settings.code_redirection,
            **keyword_arguments
        )
        self.parameters = self.code.parameters
        if sph_code is Fi:
            self.parameters.use_hydro_flag = True
            self.parameters.self_gravity_flag = True
            # Maybe make these depend on the converter?
            self.parameters.periodic_box_size = \
                100 * settings.gas_rscale
            self.parameters.timestep = settings.timestep_bridge
            self.parameters.verbosity = 0
            self.parameters.integrate_entropy_flag = False
            self.parameters.stopping_condition_maximum_density = \
                self.density_threshold
        elif sph_code is Mizuki:
            self.parameters.time_step = settings.timestep_bridge
        elif sph_code is Phantom:
            self.parameters.alpha = settings.alpha
            self.parameters.beta = settings.beta
            # self.parameters.gamma = 5./3.
            self.parameters.gamma = settings.gamma
            self.parameters.ieos = settings.ieos
            self.parameters.icooling = settings.icooling
            mu = self.parameters.mu  # mean molecular weight
            temperature = settings.isothermal_gas_temperature
            polyk = (
                constants.kB
                * temperature
                / mu
            )
            self.parameters.polyk = polyk
            self.parameters.rho_crit = 20*self.density_threshold
            self.parameters.stopping_condition_maximum_density = \
                self.density_threshold
            self.parameters.h_soft_sinkgas = settings.epsilon_gas
            self.parameters.h_soft_sinksink = settings.epsilon_gas
            self.parameters.h_acc = settings.h_acc
            self.parameters.time_step = settings.timestep_bridge #/ 2

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

    def commit_particles(self):
        return self.code.commit_particles()

    @property
    def model_time(self):
        """Return the current time"""
        return self.code.model_time + self.__time_offset

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
        try:
            return self.code.dm_particles
        except:
            return Particles()

    @property
    def sink_particles(self):
        """Return all sink particles"""
        try:
            return self.code.sink_particles
        except:
            return Particles()
        # return self.code.dm_particles

    @property
    def particles(self):
        """Return all particles"""
        return self.code.particles

    def stop(self):
        """Stop the simulation code"""
        return self.code.stop

    def evolve_model(self, end_time):
        """
        Evolve model, and manage these stopping conditions:
        - density limit detection
        --> form stars when this happens

        Use the code time step to advance until 'end_time' is reached.

        - returns immediately if the code would not evolve a step, i.e. when
          (end_time - model_time) is smaller than half a code timestep (for
          Fi).
        """
        time_unit = end_time.unit
        print("Evolve gas until %s" % end_time.in_(time_unit))

        # if code_name is Fi:
        # timestep = 0.005 | units.Myr  # self.code.parameters.timestep
        timestep = self.code.parameters.time_step
        # if self.code.model_time >= (end_time - timestep/2):
        #     return
        # if code_name is something_else:
        # some_other_condition

        density_limit_detection = \
            self.code.stopping_conditions.density_limit_detection
        if self.internal_star_formation:
            density_limit_detection.enable()

        # short offset
        time_fraction = 2**-14 * timestep
        while (
                self.model_time < (end_time - time_fraction)
        ):
            next_time = self.code.model_time + timestep
            # temp = self.code.gas_particles[0].u
            print(
                "Calling evolve_model of code (timestep: %s end_time: %s" % (
                    timestep.in_(units.Myr), next_time.in_(units.Myr),
                )
            )
            self.code.evolve_model(next_time)
            print("evolve_model of code is done")

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
    from ekster.plotting_class import temperature_to_u
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
