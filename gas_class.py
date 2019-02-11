"Class for a gas system"
from __future__ import print_function, division
import logging
import numpy

from amuse.community.fi.interface import Fi
from amuse.datamodel import Particles
from amuse.units import units, nbody_system

from cooling_class import SimplifiedThermalModelEvolver
from plotting_class import plot_hydro_and_stars

logger = logging.getLogger(__name__)


def sfe_to_density(e_loc, alpha=0.02):
    "Calculate density needed for specified star formation efficiency"
    rho = 100 * (e_loc / alpha)**2 | units.MSun * units.parsec**-3
    return rho


def density_to_sfe(rho, alpha=0.02):
    "Calculate star formation efficiency for specified density"
    sfe = alpha * (rho.value_in(100 * units.MSun * units.parsec**-3))**0.5
    return sfe


class SPH(object):
    "SPH object"

    def __init__(
            self,
            sph_code,
            converter=None,
    ):
        self.typestr = "Hydro"
        self.namestr = sph_code.__name__
        if converter is not None:
            self.converter = converter
        else:
            self.converter = nbody_system.nbody_to_si(
                1 | units.MSun,
                100 | units.parsec,
            )
        self.gas_particles = Particles()
        self.dm_particles = Particles()

        self.cooling_flag = "thermal_model"
        self.epsilon = 0.05 | units.parsec
        self.density_threshold = (1 | units.MSun) / (self.epsilon)**3
        if sph_code is Fi:
            self.code = sph_code(
                self.converter,
                mode="openmp",
            )
            self.code.parameters.begin_time = 0.0 | units.Myr
            self.code.parameters.use_hydro_flag = True
            self.code.parameters.self_gravity_flag = True
            # Have to do our own cooling
            self.code.parameters.isothermal_flag = True
            self.code.parameters.integrate_entropy_flag = False
            self.code.parameters.gamma = 1
            # Maybe make these depend on the converter?
            self.code.parameters.periodic_box_size = 10 | units.kpc
            self.code.parameters.timestep = 0.01 | units.Myr
        self.code.parameters.stopping_condition_maximum_density = \
            self.density_threshold

        self.parameters = self.code.parameters
        self.get_gravity_at_point = self.code.get_gravity_at_point
        self.get_potential_at_point = self.code.get_potential_at_point
        self.get_hydro_state_at_point = self.code.get_hydro_state_at_point

        self.channel_to_gas = self.code.gas_particles.new_channel_to(
            self.gas_particles
        )
        self.channel_from_gas = self.gas_particles.new_channel_to(
            self.code.gas_particles
        )
        self.channel_to_dm = self.code.dm_particles.new_channel_to(
            self.dm_particles
        )
        self.channel_from_dm = self.dm_particles.new_channel_to(
            self.code.dm_particles
        )

        if self.cooling_flag == "thermal_model":
            self.cooling = SimplifiedThermalModelEvolver(
                self.code.gas_particles
            )
            self.cooling.model_time = self.code.model_time

    @property
    def model_time(self):
        return self.code.model_time

    @property
    def gas_particles(self):
        return self.code.gas_particles

    @property
    def stop(self):
        return self.code.stop

    def evolve_model(self, end_time):
        density_limit_detection = \
            self.code.stopping_conditions.density_limit_detection
        density_limit_detection.enable()

        start_time = self.code.model_time
        dt = end_time - start_time
        timestep = self.code.parameters.timestep
        


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
    ):
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
        from amuse.community.fi.interface import Fi
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
            self.cooling = False
        else:
            # We want to control cooling ourselves
            # from cooling_class import Cooling
            from cooling_class import SimplifiedThermalModelEvolver
            self.gas_code.parameters.gamma = 1
            self.gas_code.parameters.isothermal_flag = True
            self.cooling = SimplifiedThermalModelEvolver(
                self.gas_code.gas_particles,
            )
            self.cooling.model_time = self.gas_code.model_time

        # Sensible to set these, even though they are already default
        self.gas_code.parameters.use_hydro_flag = True
        self.gas_code.parameters.self_gravity_flag = True
        self.gas_code.parameters.verbosity = 0

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
        "Return the current time in the code"
        return self.gas_code.model_time

    def get_gravity_at_point(self, *args, **kwargs):
        "Return gravity at specified location"
        return self.gas_code.get_gravity_at_point(*args, **kwargs)

    # @property
    # def particles(self):
    #     return self.code.particles

    def resolve_starformation(self):
        logger.info("Resolving star formation")
        high_density_gas = self.gas_particles.select_array(
            lambda rho: rho > sfe_to_density(
                1, alpha=self.alpha_sfe,
            ),
            ["rho"],
        )
        # Other selections?
        new_stars = high_density_gas.copy()
        # print(len(new_stars))
        logger.info("Removing %i former gas particles", len(new_stars))
        self.gas_code.particles.remove_particles(high_density_gas)
        self.gas_particles.remove_particles(high_density_gas)
        # self.gas_particles.synchronize_to(self.gas_code.gas_particles)
        logger.info("Removed %i former gas particles", len(new_stars))
        try:
            star_code_particles = self.star_code.particles
        except AttributeError:
            star_code_particles = self.gas_code.dm_particles
        new_stars.birth_age = self.gas_code.model_time
        logger.info("Adding %i new stars to star code", len(new_stars))
        star_code_particles.add_particles(new_stars)
        logger.info("Added %i new stars to star code", len(new_stars))
        logger.info("Adding %i new stars to model", len(new_stars))
        self.star_particles.add_particles(new_stars)
        logger.info("Added %i new stars to model", len(new_stars))
        logger.info("Adding new stars to evolution code")
        try:
            self.evo_code.particles.add_particles(new_stars)
            logger.info("Added new stars to evolution code")
        except AttributeError:
            logger.info("No evolution code exists")
            pass
        logger.info("Resolved star formation")

    def evolve_model(self, tend):
        "Evolve model to specified time and synchronise model"
        self.model_to_gas_code.copy()
        dt = tend - self.model_time
        if self.cooling:
            print("Cooling gas")
            self.cooling.evolve_for(dt/2)
        while (
                abs(tend - self.model_time) > self.gas_code.parameters.timestep
        ):
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
        self.gas_code_to_model.copy()
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
        200000 | units.MSun,
        9.0 | units.parsec,
    )
    numpy.random.seed(11)
    gas = molecular_cloud(targetN=100000, convert_nbody=converter).result
    print("Number of gas particles: %i" % (len(gas)))

    model = Gas(gas=gas, converter=converter, internal_cooling=False)
    print(model.gas_code.parameters)
    timestep = 0.1 | units.Myr
    for step in range(20):
        time = step * timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time)
        print(
            "Maximum density / stopping density = %s" % (
                model.gas_particles.density.max()
                / model.gas_code.parameters.stopping_condition_maximum_density,
            )
        )
        
        plotname = "gastest-%04i.png" % (step)
        logger.info("Creating plot")
        plot_hydro_and_stars(
            model.model_time,
            model.gas_code,
            model.gas_code.dm_particles,
            L=40,
            filename=plotname,
            title="time = %06.1f %s" % (
                model.model_time.value_in(units.Myr),
                units.Myr,
            ),
            gasproperties=["density", "temperature"],
            colorbar=True,
        )


if __name__ == "__main__":
    main()
