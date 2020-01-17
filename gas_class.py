"Class for a gas system"
from __future__ import print_function, division
import logging
import numpy

from amuse.community.fi.interface import Fi
from amuse.community.phantom.interface import Phantom
from amuse.datamodel import Particles, Particle
from amuse.units import units, nbody_system, constants

from basic_class import BasicCode
from cooling_class import SimplifiedThermalModelEvolver
from plotting_class import plot_hydro_and_stars  # , u_to_temperature
# from sinks_class import accrete_gas  # , SinkParticles
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
                1.0e5 | units.MSun,
                5.0 | units.parsec,
            )
        if begin_time is None:
            begin_time = 0. | units.Myr
        self.__begin_time = self.unit_converter.to_si(begin_time)

        self.cooling_type = cooling_type

        self.epsilon = 0.1 | units.parsec
        # self.density_threshold = (5e-20 | units.g * units.cm**-3)
        self.density_threshold = (1e7 | units.amu * units.cm**-3)
        print(
            "Density threshold for sink formation: %s (%s)" % (
                self.density_threshold.in_(units.MSun * units.parsec**-3),
                self.density_threshold.in_(units.g * units.cm**-3),
            )
        )
        # self.density_threshold = (1 | units.MSun) / (self.epsilon)**3
        self.code = sph_code(
            self.unit_converter,
            redirection="none",
            **keyword_arguments
        )
        self.parameters = self.code.parameters
        if sph_code is Fi:
            self.parameters.use_hydro_flag = True
            self.parameters.self_gravity_flag = True
            # Maybe make these depend on the converter?
            self.parameters.periodic_box_size = 10 | units.kpc
            self.parameters.timestep = 0.01 | units.Myr
            self.parameters.verbosity = 0
            self.parameters.integrate_entropy_flag = False
            self.parameters.stopping_condition_maximum_density = \
                self.density_threshold
        elif sph_code is Phantom:
            self.parameters.alpha = 0.1  # art. viscosity parameter (min)
            # self.parameters.gamma = 5./3.
            self.parameters.gamma = 1.0
            self.parameters.ieos = 1  # isothermal
            # self.parameters.ieos = 2  # adiabatic
            mu = self.parameters.mu  # mean molecular weight
            temperature = 10 | units.K
            polyk = (
                constants.kB
                * temperature
                / mu
            )
            self.parameters.polyk = polyk
            self.parameters.rho_crit = 0*self.density_threshold
            self.parameters.stopping_condition_maximum_density = \
                self.density_threshold
            self.parameters.h_soft_sinkgas = 0.1 | units.parsec
            self.parameters.h_soft_sinksink = 0.1 | units.parsec
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
        timestep = 0.005 | units.Myr  # self.code.parameters.timestep
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
        while self.code.model_time < end_time:
            if self.cooling and not first:
                self.cooling.evolve_for(timestep)
            next_time = self.code.model_time + timestep
            # temp = self.code.gas_particles[0].u
            print("Calling evolve_model of code")
            self.code.evolve_model(next_time)
            print("evolve_model of code is done")
            # temp2 = self.code.gas_particles[0].u
            # if temp != temp2:
            #     print(
            #         "Temperature of particle 0 changed: %s -> %s" % (
            #             u_to_temperature(temp),
            #             u_to_temperature(temp2),
            #         )
            #     )
            # tempavg = self.code.gas_particles.u.mean()
            # tempstd = self.code.gas_particles.u.std()
            # print("temp avg: %s std: %s"%(u_to_temperature(tempavg),
            #                               tempstd))
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
            # internal_cooling=False,
            logger=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.alpha_sfe = alpha_sfe
        if converter is not None:
            self.gas_converter = converter
        else:
            # mass_scale = gas.mass.sum()
            # # should be something related to length spread?
            # length_scale = 100 | units.parsec
            mass_scale = 1.0e5 | units.MSun
            length_scale = 5.0 | units.parsec
            self.gas_converter = nbody_system.nbody_to_si(
                mass_scale,
                length_scale,
            )
        # We're simulating gas as gas, not just "dark matter".
        if gas is None:
            self.gas_particles = Particles()
        else:
            self.gas_particles = gas
            self.gas_particles.h_smooth = 100 | units.AU
        self.sink_particles = Particles()
        # from amuse.community.fi.interface import Fi
        self.gas_code = GasCode(
            # Fi,
            Phantom,
            self.gas_converter,
            # mode="openmp",
            # cooling_type="thermal_model",
            cooling_type="default",
            # channel_type="sockets",
            # redirection="none",
        )
        # from amuse.community.gadget2.interface import Gadget2
        # self.gas_code = Gadget2(
        #     self.gas_converter,
        #     # redirection="none",
        # )
        # self.gas_code.parameters.epsilon_squared = (epsilon)**2
        # self.gas_code.parameters.timestep = min(
        #     self.gas_converter.to_si(
        #         0.01 | nbody_system.time,  # 0.05 | units.Myr
        #     ),
        #     0.01 | units.Myr,
        # )
        print("Adding particles: %i" % len(self.gas_particles))
        self.gas_code.gas_particles.add_particles(self.gas_particles)
        print("Particles added: %i" % len(self.gas_code.gas_particles))
        self.model_to_gas_code = self.gas_particles.new_channel_to(
            self.gas_code.gas_particles,
        )
        self.gas_code_to_model = self.gas_code.gas_particles.new_channel_to(
            self.gas_particles,
        )
        self.sinks_to_model = self.gas_code.sink_particles.new_channel_to(
            self.sink_particles,
        )

        # if internal_cooling:
        #     self.gas_code.parameters.gamma = 5./3.
        #     self.gas_code.parameters.isothermal_flag = False
        #     # self.gas_code.parameters.radiation_flag = True
        #     # self.gas_code.parameters.star_formation_flag = True

        #     self.cooling = False
        # else:
        #     # We want to control cooling ourselves
        #     # from cooling_class import Cooling
        #     # from cooling_class import SimplifiedThermalModelEvolver
        #     self.gas_code.parameters.isothermal_flag = False
        #     self.gas_code.parameters.radiation_flag = False
        #     self.gas_code.parameters.gamma = 5./3.
        #     self.cooling = SimplifiedThermalModelEvolver(
        #         self.gas_code.gas_particles,
        #     )
        #     self.cooling.model_time = self.gas_code.model_time
        #     # self.cooling = False

        # Sensible to set these, even though they are already default
        # self.gas_code.parameters.use_hydro_flag = True
        # self.gas_code.parameters.self_gravity_flag = True
        # self.gas_code.parameters.verbosity = 0
        # self.gas_code.parameters.integrate_entropy_flag = False

        # We want to stop and create stars when a certain density is reached
        # print(c.stopping_conditions.supported_conditions())
        # self.gas_code.parameters.stopping_condition_maximum_density = \
        #     (1 | units.MSun) / (epsilon**3)
        # sfe_to_density(1, alpha=self.alpha_sfe)
        logging.info(
            "The maximum density is set to %s",
            self.gas_code.parameters.stopping_condition_maximum_density.in_(
                units.amu * units.cm**-3
            )
        )

        # self.code = self.gas_code
        # self.particles = self.gas_particles

    @property
    def model_time(self):
        """
        Return the current time in the code modified with the begin time
        """
        return self.gas_code.model_time + self.__begin_time

    def get_gravity_at_point(self, *args, **kwargs):
        """Return gravity at specified location"""
        return self.gas_code.get_gravity_at_point(*args, **kwargs)

    # @property
    # def particles(self):
    #     return self.code.particles

    # def resolve_collision(self):
    #     """Collide two particles - determine what should happen"""

    #     return self.merge_particles()

    # def merge_particles(self):
    #     """Resolve collision by merging"""
    #     return

    def sink_accretion(self):
        self.logger.info("Resolving accretion of matter onto sinks")
        for i, sink in enumerate(self.sink_particles):
            xs, ys, zs = sink.position
            r2s = sink.radius**2
            accreted_gas = self.gas_particles.select(
                lambda x, y, z:
                ((x-xs)**2 + (y-ys)**2 + (z-zs)**2) < r2s,
                ["x", "y", "z"]
            )
            sink_and_accreted_gas = Particles()
            sink_and_accreted_gas.add_particle(sink)
            sink_and_accreted_gas.add_particles(accreted_gas)
            new_position = sink_and_accreted_gas.center_of_mass()
            new_velocity = sink_and_accreted_gas.center_of_mass_velocity()
            new_mass = sink_and_accreted_gas.total_mass()
            sink.position = new_position
            sink.velocity = new_velocity
            sink.mass = new_mass
            # if not accreted_tas.is_empty():
            #     cm = sink.position * sink.mass
            #     p = sink.velocity * sink.mass
            #     sink.mass += to_be_accreted.total_mass()
            #     sink.position = (
            #         cm + (
            #             to_be_accreted.center_of_mass()
            #             * to_be_accreted.total_mass()
            #         )
            #     ) / sink.mass
            #     sink.velocity = (
            #         p + to_be_accreted.total_momentum()
            #     ) / sink.mass
            #     # sink.spin
            self.remove_gas(accreted_gas)

    def sink_merger(self):
        self.logger.info("Resolving sink mergers")
        sinks = self.sink_particles.sorted_by_attribute('mass').reversed()
        for i, sink in enumerate(sinks):
            xs, ys, zs = sink.position
            r2s = sink.radius**2
            to_be_accreted = sinks[i:].select(
                lambda x, y, z:
                (x-xs)**2
                + (y-ys)**2
                + (z-zs)**2
                < r2s,
                ["x", "y", "z"]
            )
            if not to_be_accreted.is_empty():
                cm = sink.position * sink.mass
                p = sink.velocity * sink.mass
                sink.mass += to_be_accreted.total_mass()
                sink.position = (
                    cm + (
                        to_be_accreted.center_of_mass()
                        * to_be_accreted.total_mass()
                    )
                ) / sink.mass
                sink.velocity = (
                    p + to_be_accreted.total_momentum()
                ) / sink.mass
                # sink.spin
                self.remove_sinks(to_be_accreted)

    def add_sink(self, sink):
        self.gas_code.sink_particles.add_particle(sink)
        self.sink_particles.add_particle(sink)

    def remove_sinks(self, sinks):
        self.gas_code.sink_particles.remove_particles(sinks)
        self.sink_particles.remove_particles(sinks)

    def add_gas(self, gas):
        self.gas_code.gas_particles.add_particles(gas)
        self.gas_particles.add_particles(gas)

    def remove_gas(self, gas):
        self.gas_code.gas_particles.remove_particles(gas)
        self.gas_particles.remove_particles(gas)

    def sink_formation(self):
        maximum_density = (
            self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select_array(
            lambda rho:
            rho > maximum_density,
            ["rho"],
        )
        for i, origin_gas in enumerate(high_density_gas):
            try:
                new_sink = Particle()
                new_sink.radius = 0.01 | units.parsec
                # 100 | units.AU  # or something related to mass?
                new_sink.accreted_mass = 0 | units.MSun
                o_x, o_y, o_z = origin_gas.position

                # accreted_gas = self.gas_particles.select_array(
                #     lambda x, y, z:
                #     ((x-o_x)**2 + (y-o_y)**2 + (z-o_z)**2) <
                #     new_sink.radius**2,
                #     ["x", "y", "z"]
                # )
                accreted_gas = Particles()
                accreted_gas.add_particle(origin_gas)
                new_sink.position = accreted_gas.center_of_mass()
                new_sink.velocity = accreted_gas.center_of_mass_velocity()
                new_sink.mass = accreted_gas.total_mass()
                new_sink.accreted_mass = (
                    accreted_gas.total_mass() - origin_gas.mass
                )
                self.remove_gas(accreted_gas)
                self.add_sink(new_sink)
                print("Added sink %i" % i)
            except:
                raise("Could not add another sink")

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
        new_stars.birth_age = self.model_time
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
        stopping_density = \
            self.gas_code.parameters.stopping_condition_maximum_density
        self.gas_stopping_conditions = \
            self.gas_code.stopping_conditions.density_limit_detection
        self.gas_stopping_conditions.enable()
        self.model_to_gas_code.copy()
        # tstart = self.gas_code.model_time
        # dt = tend - tstart
        timestep = 0.01 | units.Myr  # self.gas_code.parameters.timestep
        # while (
        #         abs(tend - self.model_time)
        #         >= self.gas_code.parameters.timestep
        # ):
        while self.model_time < (tend - timestep/2):
            # if self.cooling:
            #     print("Cooling gas")
            #     self.cooling.evolve_for(dt/2)
            print(
                "Evolving to %s (now at %s)" % (
                    tend.in_(units.Myr),
                    self.model_time.in_(units.Myr),
                )
            )
            self.gas_code.evolve_model(tend)
            dead_gas = self.gas_code.gas_particles.select(
                lambda x: x <= 0.,
                ["h_smooth"]
            )
            print("Number of dead/accreted gas particles: %i" % len(dead_gas))
            self.remove_gas(dead_gas)
            # Check stopping conditions
            number_of_checks = 0
            while (
                    self.gas_stopping_conditions.is_set()
                    and (number_of_checks < 1)
            ):
                number_of_checks += 1
                print("Gas code stopped - max density reached")
                self.gas_code_to_model.copy()
                self.sinks_to_model.copy()
                # self.sink_accretion()
                self.sink_formation()
                # self.resolve_starformation()
                print(
                    "Now we have %i stars"
                    % len(self.gas_code.sink_particles)
                )
                print(
                    "And we have %i gas"
                    % len(self.gas_code.gas_particles)
                )
                print(
                    "A total of %i particles"
                    % len(self.gas_code.particles)
                )
                self.gas_code.evolve_model(tend)
                dead_gas = self.gas_code.gas_particles.select(
                    lambda x: x <= 0.,
                    ["h_smooth"]
                )
                print(
                    "Number of dead/accreted gas particles: %i"
                    % len(dead_gas)
                )
                self.remove_gas(dead_gas)

                print(len(self.gas_code.gas_particles))
                print(
                    self.gas_code.gas_particles.density.max()
                    / stopping_density
                )

            # ? self.gas_code_to_model.copy()
            # if self.cooling:
            #     print("Cooling gas again")
            #     self.cooling.evolve_for(dt/2)
        self.gas_code_to_model.copy()
        self.sinks_to_model.copy()

    def stop(self):
        "Stop code"
        self.gas_code.stop()


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

        # plotname = "gastest-%04i.png" % (step)
        # print("Creating plot")
        # plot_hydro_and_stars(
        #     model.model_time,
        #     model,
        #     model.sink_particles,
        #     L=20,
        #     N=150,
        #     filename=plotname,
        #     title="time = %06.1f %s" % (
        #         model.model_time.value_in(units.Myr),
        #         units.Myr,
        #     ),
        #     gasproperties=["density"],  # , "temperature"],
        #     colorbar=True,
        # )
        times.append(time)
        # ekin = model.gas_code.kinetic_energy
        # starkin = model.gas_code.dm_particles.kinetic_energy()
        # gaskin = model.gas_code.gas_particles.kinetic_energy()
        # print(
        #     "Kinetic energy ratios: all=%s star=%s gas=%s" % (
        #         (gaskin+starkin)/ekin,
        #         starkin/ekin,
        #         gaskin/ekin,
        #     )
        # )
        # kinetic_energies.append(model.gas_code.kinetic_energy)
        # potential_energies.append(model.gas_code.potential_energy)
        # thermal_energies.append(model.gas_code.thermal_energy)
        step += 1
    # from plot_energy import energy_plot
    # energy_plot(
    #     times, kinetic_energies, potential_energies, thermal_energies,
    #     "gas_energies.png",
    # )


if __name__ == "__main__":
    main()
