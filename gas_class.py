"Class for a gas system"
from __future__ import print_function, division
import logging
import numpy

from amuse.community.fi.interface import Fi
from amuse.datamodel import Particles
from amuse.units import units, nbody_system

from basic_class import BasicCode
from cooling_class import SimplifiedThermalModelEvolver
from plotting_class import plot_hydro_and_stars


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
            converter=None,
            sph_code=Fi,
            logger=None,
            internal_star_formation=False,
            **keyword_arguments
    ):
        self.typestr = "Hydro"
        self.namestr = sph_code.__name__
        self.logger = logger or logging.getLogger(__name__)
        self.internal_star_formation = internal_star_formation
        if converter is not None:
            self.converter = converter
        else:
            self.converter = nbody_system.nbody_to_si(
                1 | units.MSun,
                1 | units.parsec,
            )
        # self.gas_particles = Particles()
        # self.dm_particles = Particles()

        self.cooling_flag = "thermal_model"
        self.epsilon = 0.05 | units.parsec
        self.density_threshold = (1 | units.MSun) / (self.epsilon)**3
        # self.density_threshold = 10**-12 | units.g * units.cm**-3
        if sph_code is Fi:
            self.code = sph_code(
                self.converter,
                mode="openmp",
                **keyword_arguments
            )
            self.parameters = self.code.parameters
            self.parameters.begin_time = 0.0 | units.Myr
            self.parameters.use_hydro_flag = True
            self.parameters.self_gravity_flag = True
            # Have to do our own cooling
            self.parameters.isothermal_flag = True
            self.parameters.integrate_entropy_flag = False
            self.parameters.gamma = 1
            # Maybe make these depend on the converter?
            self.parameters.periodic_box_size = 10 | units.kpc
            self.parameters.timestep = 0.01 | units.Myr
        self.parameters.stopping_condition_maximum_density = \
            self.density_threshold

        # self.get_gravity_at_point = self.code.get_gravity_at_point
        # self.get_potential_at_point = self.code.get_potential_at_point
        # self.get_hydro_state_at_point = self.code.get_hydro_state_at_point

        if self.cooling_flag == "thermal_model":
            self.cooling = SimplifiedThermalModelEvolver(
                self.code.gas_particles
            )
            self.cooling.model_time = self.code.model_time
        else:
            self.cooling = False

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

    # @property
    # def model_time(self):
    #     """Return the current time"""
    #     return self.code.model_time

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

    # @property
    # def particles(self):
    #     """Return all particles"""
    #     return self.code.particles

    # @property
    # def stop(self):
    #     """Stop the simulation code"""
    #     return self.code.stop

    def evolve_model(self, end_time):
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

        # if code_name is Fi:
        timestep = self.code.parameters.timestep
        half_timestep = timestep / 2
        if self.model_time >= (end_time - timestep/2):
            return
        # if code_name is something_else:
        # some_other_condition

        density_limit_detection = \
            self.code.stopping_conditions.density_limit_detection
        if self.internal_star_formation:
            density_limit_detection.enable()

        # Do cooling with a leapfrog scheme
        first = True
        if self.cooling and first:
            self.cooling.evolve_for(half_timestep)
            first = False
        while self.model_time < (end_time - half_timestep):
            if self.cooling and not first:
                self.cooling.evolve_for(timestep)
            next_time = self.model_time + timestep
            self.code.evolve_model(next_time)

            if (
                    density_limit_detection.is_set()
            ):
                if self.internal_star_formation:
                    self.resolve_starformation()
                else:
                    break

        # Make sure cooling and the code are synchronised when the loop ends
        if self.cooling:
            self.cooling.evolve_for(half_timestep)

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


class Gas(object):
    """Gas object"""

    def __init__(
            self,
            gas=None,
            converter=None,
            # mass_scale=1000 | units.MSun,
            # length_scale=100 | units.parsec,
            epsilon=0.05 | units.parsec,
            alpha_sfe=0.04,
            internal_cooling=False,
            logger=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.alpha_sfe = alpha_sfe
        if converter is not None:
            self.gas_converter = converter
        else:
            mass_scale = gas.mass.sum()
            # should be something related to length spread?
            length_scale = 100 | units.parsec
            self.gas_converter = nbody_system.nbody_to_si(
                mass_scale,
                length_scale,
            )
        # We're simulating gas as gas, not just "dark matter".
        if gas is None:
            self.gas_particles = Particles()
        else:
            self.gas_particles = gas
        # from amuse.community.fi.interface import Fi
        self.gas_code = Fi(
            self.gas_converter,
            mode="openmp",
            # channel_type="sockets",
            # redirection="none",
        )
        # from amuse.community.gadget2.interface import Gadget2
        # self.gas_code = Gadget2(
        #     self.gas_converter,
        #     # redirection="none",
        # )
        self.gas_code.parameters.epsilon_squared = (epsilon)**2
        self.gas_code.parameters.timestep = min(
            self.gas_converter.to_si(
                0.01 | nbody_system.time,  # 0.05 | units.Myr
            ),
            0.01 | units.Myr,
        )
        self.gas_code.gas_particles.add_particles(self.gas_particles)
        self.model_to_gas_code = self.gas_particles.new_channel_to(
            self.gas_code.gas_particles,
        )
        self.gas_code_to_model = self.gas_code.gas_particles.new_channel_to(
            self.gas_particles,
        )

        if internal_cooling:
            self.gas_code.parameters.gamma = 5./3.
            self.gas_code.parameters.isothermal_flag = False
            # self.gas_code.parameters.radiation_flag = True
            # self.gas_code.parameters.star_formation_flag = True

            self.cooling = False
        else:
            # We want to control cooling ourselves
            # from cooling_class import Cooling
            # from cooling_class import SimplifiedThermalModelEvolver
            self.gas_code.parameters.isothermal_flag = False
            self.gas_code.parameters.radiation_flag = False
            self.gas_code.parameters.gamma = 5./3.
            self.cooling = SimplifiedThermalModelEvolver(
                self.gas_code.gas_particles,
            )
            self.cooling.model_time = self.gas_code.model_time
            # self.cooling = False

        # Sensible to set these, even though they are already default
        self.gas_code.parameters.use_hydro_flag = True
        self.gas_code.parameters.self_gravity_flag = True
        self.gas_code.parameters.verbosity = 0
        self.gas_code.parameters.integrate_entropy_flag = False

        # We want to stop and create stars when a certain density is reached
        # print(c.stopping_conditions.supported_conditions())
        self.gas_code.parameters.stopping_condition_maximum_density = \
            (1 | units.MSun) / (epsilon**3)
        # sfe_to_density(1, alpha=self.alpha_sfe)
        logging.info(
            "The maximum density is set to %s",
            self.gas_code.parameters.stopping_condition_maximum_density.in_(
                units.amu * units.cm**-3
            )
        )
        self.gas_stopping_conditions = \
            self.gas_code.stopping_conditions.density_limit_detection
        self.gas_stopping_conditions.enable()

        # self.code = self.gas_code
        # self.particles = self.gas_particles

    @property
    def model_time(self):
        """Return the current time in the code"""
        return self.gas_code.model_time

    def get_gravity_at_point(self, *args, **kwargs):
        """Return gravity at specified location"""
        return self.gas_code.get_gravity_at_point(*args, **kwargs)

    # @property
    # def particles(self):
    #     return self.code.particles

    def resolve_collision(self):
        """Collide two particles - determine what should happen"""

        return self.merge_particles()

    def merge_particles(self):
        """Resolve collision by merging"""
        return

    def resolve_starformation(self):
        self.logger.info("Resolving star formation")
        high_density_gas = self.gas_particles.select_array(
            lambda rho: rho > sfe_to_density(
                1, alpha=self.alpha_sfe,
            ),
            ["rho"],
        )
        # Other selections?
        new_stars = high_density_gas.copy()
        # print(len(new_stars))
        self.logger.debug("Removing %i former gas particles", len(new_stars))
        self.gas_code.particles.remove_particles(high_density_gas)
        self.gas_particles.remove_particles(high_density_gas)
        # self.gas_particles.synchronize_to(self.gas_code.gas_particles)
        self.logger.debug("Removed %i former gas particles", len(new_stars))
        try:
            star_code_particles = self.star_code.particles
        except AttributeError:
            star_code_particles = self.gas_code.dm_particles
        new_stars.birth_age = self.gas_code.model_time
        self.logger.debug("Adding %i new stars to star code", len(new_stars))
        star_code_particles.add_particles(new_stars)
        self.logger.debug("Added %i new stars to star code", len(new_stars))
        self.logger.debug("Adding %i new stars to model", len(new_stars))
        try:
            self.star_particles.add_particles(new_stars)
            logger.debug("Added %i new stars to model", len(new_stars))
        except NameError:
            self.logger.debug("No stars in this model")
        self.logger.debug("Adding new stars to evolution code")
        try:
            self.evo_code.particles.add_particles(new_stars)
            self.logger.debug("Added new stars to evolution code")
        except AttributeError:
            self.logger.debug("No evolution code exists")
        self.logger.info(
            "Resolved star formation, formed %i stars" % len(new_stars)
        )

    def evolve_model(self, tend):
        "Evolve model to specified time and synchronise model"
        self.model_to_gas_code.copy()
        tstart = self.gas_code.model_time
        dt = tend - tstart
        while (
                abs(tend - self.model_time)
                >= self.gas_code.parameters.timestep
        ):
            if self.cooling:
                print("Cooling gas")
                self.cooling.evolve_for(dt/2)
            print(
                "Evolving to %s (now at %s)" % (
                    tend.in_(units.Myr),
                    self.model_time.in_(units.Myr),
                )
            )
            self.gas_code.evolve_model(tend)
            # Check stopping conditions
            while self.gas_stopping_conditions.is_set():
                print("Gas code stopped - max density reached")
                self.gas_code_to_model.copy()
                self.resolve_starformation()
                print("Now we have %i stars" % len(self.gas_code.dm_particles))
                print("And we have %i gas" % len(self.gas_code.gas_particles))
                print("A total of %i particles" % len(self.gas_code.particles))
                self.gas_code.evolve_model(tend)
            # ? self.gas_code_to_model.copy()
            if self.cooling:
                print("Cooling gas again")
                self.cooling.evolve_for(dt/2)
        self.gas_code_to_model.copy()

    def stop(self):
        "Stop code"
        self.gas_code.stop()


def main():
    "Test class with a molecular cloud"
    from amuse.ext.molecular_cloud import molecular_cloud
    converter = nbody_system.nbody_to_si(
        500 | units.MSun,
        2.0 | units.parsec,
    )
    numpy.random.seed(11)
    temperature = 10 | units.K
    from plotting_class import temperature_to_u
    u = temperature_to_u(temperature)
    gas = molecular_cloud(targetN=100000, convert_nbody=converter).result
    gas.u = u

    gastwo = molecular_cloud(targetN=100000, convert_nbody=converter).result
    gastwo.u = u

    gas.x -= 2 | units.parsec
    gastwo.x += 2 | units.parsec
    gastwo.y += 0 | units.parsec
    gastwo.z -= 0 | units.parsec
    gas.vx += 1.2 | units.kms
    gastwo.vx -= 1.2 | units.kms
    # gas.vx += ((3 | units.parsec) / (2 | units.Myr))
    # gastwo.vx -= ((3 | units.parsec) / (2 | units.Myr))

    gas.add_particles(gastwo)
    print("Number of gas particles: %i" % (len(gas)))

    model = GasCode(converter=converter)
    print(model.parameters)
    timestep = model.parameters.timestep
    model.gas_particles.add_particles(gas)

    times = [] | units.Myr
    kinetic_energies = [] | units.J
    potential_energies = [] | units.J
    thermal_energies = [] | units.J
    time = 0 | units.Myr
    step = 0
    while time < 4 | units.Myr:
        time += timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time)
        print(
            "Maximum density / stopping density = %s" % (
                model.gas_particles.density.max()
                / model.parameters.stopping_condition_maximum_density,
            )
        )

        plotname = "gastest-%04i.png" % (step)
        print("Creating plot")
        plot_hydro_and_stars(
            model.model_time,
            model,
            model.dm_particles,
            L=8,
            filename=plotname,
            title="time = %06.1f %s" % (
                model.model_time.value_in(units.Myr),
                units.Myr,
            ),
            gasproperties=["density", "temperature"],
            colorbar=True,
        )
        times.append(time)
        ekin = model.kinetic_energy
        starkin = model.dm_particles.kinetic_energy()
        gaskin = model.gas_particles.kinetic_energy()
        print(
            "Kinetic energy ratios: all=%s star=%s gas=%s" % (
                (gaskin+starkin)/ekin,
                starkin/ekin,
                gaskin/ekin,
            )
        )
        kinetic_energies.append(model.kinetic_energy)
        potential_energies.append(model.potential_energy)
        thermal_energies.append(model.thermal_energy)
        step += 1
    from plot_energy import energy_plot
    energy_plot(
        times, kinetic_energies, potential_energies, thermal_energies,
        "gas_energies.png",
    )


def _main():
    "Test class with a molecular cloud"
    from amuse.ext.molecular_cloud import molecular_cloud
    converter = nbody_system.nbody_to_si(
        5000 | units.MSun,
        5.0 | units.parsec,
    )
    numpy.random.seed(11)
    temperature = 30 | units.K
    from plotting_class import temperature_to_u
    u = temperature_to_u(temperature)
    gas = molecular_cloud(targetN=500000, convert_nbody=converter).result
    gas.u = u

    gastwo = molecular_cloud(targetN=500000, convert_nbody=converter).result
    gastwo.u = u

    gastwo.x += 12 | units.parsec
    gastwo.y += 3 | units.parsec
    gastwo.z -= 1 | units.parsec
    gastwo.vx -= ((5 | units.parsec) / (2 | units.Myr))

    gas.add_particles(gastwo)
    print("Number of gas particles: %i" % (len(gas)))

    model = Gas(
        gas=gas, converter=converter, internal_cooling=False,
        internal_star_formation=True,
    )
    model.epsilon = 0.05 | units.parsec
    model.parameters.stopping_condition_maximum_density = \
        10**-12 | units.g * units.cm**-3
    print(model.gas_code.parameters)
    timestep = model.gas_code.parameters.timestep

    times = [] | units.Myr
    kinetic_energies = [] | units.J
    potential_energies = [] | units.J
    thermal_energies = [] | units.J
    time = 0 | units.Myr
    step = 0
    while time < 4 | units.Myr:
        time += timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time)
        print(
            "Maximum density / stopping density = %s" % (
                model.gas_particles.density.max()
                / model.gas_code.parameters.stopping_condition_maximum_density,
            )
        )

        plotname = "gastest-%04i.png" % (step)
        print("Creating plot")
        plot_hydro_and_stars(
            model.model_time,
            model.gas_code,
            model.gas_code.dm_particles,
            L=20,
            filename=plotname,
            title="time = %06.1f %s" % (
                model.model_time.value_in(units.Myr),
                units.Myr,
            ),
            gasproperties=["density", "temperature"],
            colorbar=True,
        )
        times.append(time)
        ekin = model.gas_code.kinetic_energy
        starkin = model.gas_code.dm_particles.kinetic_energy()
        gaskin = model.gas_code.gas_particles.kinetic_energy()
        print(
            "Kinetic energy ratios: all=%s star=%s gas=%s" % (
                (gaskin+starkin)/ekin,
                starkin/ekin,
                gaskin/ekin,
            )
        )
        kinetic_energies.append(model.gas_code.kinetic_energy)
        potential_energies.append(model.gas_code.potential_energy)
        thermal_energies.append(model.gas_code.thermal_energy)
        step += 1
    from plot_energy import energy_plot
    energy_plot(
        times, kinetic_energies, potential_energies, thermal_energies,
        "gas_energies.png",
    )


if __name__ == "__main__":
    main()
