"Class for a star cluster embedded in a tidal field and a gaseous region"
from __future__ import print_function, division
import logging
from amuse.community.fastkick.interface import FastKick
from amuse.datamodel import ParticlesSuperset
from amuse.units import units, nbody_system
from amuse.units.quantities import VectorQuantity
from amuse.io import write_set_to_file

from bridge import (
    Bridge, CalculateFieldForCodes,
)
from gas_class import Gas  # , sfe_to_density
from star_cluster_class import StarCluster
from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)
from plotting_class import plot_hydro_and_stars

Tide = TimeDependentSpiralArmsDiskModel
logger = logging.getLogger(__name__)


class ClusterInPotential(
        StarCluster,
        Gas,
):
    """Stellar cluster in an external potential"""

    def __init__(
            self,
            stars=None,
            gas=None,
            # epsilon=200 | units.AU,
            epsilon=0.1 | units.parsec,
            star_converter=None,
            gas_converter=None,
    ):
        self.objects = (StarCluster, Gas)
        mass_scale_stars = stars.mass.sum()
        length_scale_stars = 3 | units.parsec
        if star_converter is None:
            converter_for_stars = nbody_system.nbody_to_si(
                mass_scale_stars,
                length_scale_stars,
            )
        else:
            converter_for_stars = star_converter
        logger.info("Initialising StarCluster")
        StarCluster.__init__(
            self, stars=stars, converter=converter_for_stars, epsilon=epsilon,
        )
        logger.info("Initialised StarCluster")

        mass_scale_gas = gas.mass.sum()
        length_scale_gas = 100 | units.parsec
        if gas_converter is None:
            converter_for_gas = nbody_system.nbody_to_si(
                mass_scale_gas,
                length_scale_gas,
            )
        else:
            converter_for_gas = gas_converter
        logger.info("Initialising Gas")
        Gas.__init__(
            self, gas=gas, converter=converter_for_gas, epsilon=epsilon,
            internal_cooling=False,
        )
        self.gas_code.parameters.timestep = 0.005 | units.Myr
        logger.info("Initialised Gas")
        logger.info("Creating Tide object")
        self.tidal_field = Tide()
        logger.info("Created Tide object")

        self.epsilon = epsilon
        self.converter = converter_for_gas

        def new_field_gravity_code():
            "Create a new field code"
            result = FastKick(
                self.converter,
                # redirection="file",
                # redirect_file=(
                #     p.dir_codelogs + "/field_gravity_code.log"
                #     ),
                mode="cpu",
                number_of_workers=2,
            )
            result.parameters.epsilon_squared = (self.epsilon)**2
            return result

        def new_field_code(
                code,
        ):
            " something"
            # result = CalculateFieldForCodesUsingReinitialize(
            result = CalculateFieldForCodes(
                new_field_gravity_code,
                [code],
                # required_attributes=[
                #     'mass', 'radius',
                #     'x', 'y', 'z',
                #     'vx', 'vy', 'vz',
                # ]
                # required_attributes=[
                #     'mass', 'u',
                #     'x', 'y', 'z',
                #     'vx', 'vy', 'vz',
                # ]
            )
            return result

        to_gas_codes = [
            self.star_code,
            self.tidal_field,
        ]
        to_stars_codes = [
            # new_field_code(
            self.gas_code,
            # ),
            self.tidal_field,
        ]

        self.system = Bridge(
            timestep=(
                2 * self.gas_code.parameters.timestep
            ),
            use_threading=False,
        )
        self.system.add_system(
            self.star_code,
            partners=to_stars_codes,
            do_sync=True,
        )
        self.system.add_system(
            self.gas_code,
            partners=to_gas_codes,
            do_sync=True,
            # do_sync=False,
        )
        # self.system.timestep = 2 * self.gas_code.parameters.timestep
        # 0.05 | units.Myr

    # @property
    # def gas_particles(self):
    #     "Return gas particles"
    #     return self.gas_particles

    # @property
    # def star_particles(self):
    #     "Return star particles"
    #     return self.star_particles

    @property
    def particles(self):
        "Return all particles"
        return ParticlesSuperset(
            self.star_particles,
            self.gas_particles,
        )

    @property
    def code(self):
        "Return the main code - the Bridge system in this case"
        return self.system

    def resolve_starformation(self):
        logger.info("Resolving star formation")
        high_density_gas = self.gas_particles.select_array(
            lambda rho:
            rho > self.gas_code.parameters.stopping_condition_maximum_density,
            # sfe_to_density(
            #     1, alpha=self.alpha_sfe,
            # ),
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

        star_code_particles = self.star_code.particles
        new_stars.birth_age = self.gas_code.model_time
        logger.info("Adding %i new stars to star code", len(new_stars))
        star_code_particles.add_particles(new_stars)
        logger.info("Added %i new stars to star code", len(new_stars))
        logger.info("Adding %i new stars to model", len(new_stars))
        self.star_particles.add_particles(new_stars)
        logger.info("Added %i new stars to model", len(new_stars))
        logger.info("Adding new stars to evolution code")
        self.evo_code.particles.add_particles(new_stars)
        logger.info("Added new stars to evolution code")
        logger.info("Resolved star formation")

    def evolve_model(self, tend):
        "Evolve system to specified time"
        logger.info(
            "Evolving %s to time %s",
            self.__name__,
            tend,
        )
        self.model_to_evo_code.copy()
        self.model_to_gas_code.copy()
        while abs(self.model_time - tend) >= self.system.timestep:
            evo_time = self.evo_code.model_time
            self.model_to_star_code.copy()
            evo_timestep = self.evo_code.particles.time_step.min()
            logger.info(
                "Smallest evo timestep: %s", evo_timestep.in_(units.Myr)
            )
            time = min(
                evo_time+evo_timestep,
                tend,
            )
            logger.info("Evolving to %s", time.in_(units.Myr))
            logger.info("Stellar evolution...")
            self.evo_code.evolve_model(time)
            dt_cooling = time - self.gas_code.model_time
            if self.cooling:
                logger.info("Cooling gas...")
                self.cooling.evolve_for(dt_cooling/2)
            logger.info("System...")

            self.system.evolve_model(time)
            while self.gas_stopping_conditions.is_set():
                logger.info("Gas code stopped - max density reached")
                self.gas_code_to_model.copy()
                self.resolve_starformation()
                logger.info(
                    "Now we have %i stars", len(self.star_particles),
                )
                logger.info(
                    "And we have %i gas", len(self.gas_particles),
                )
                logger.info(
                    "A total of %i particles", len(self.gas_code.particles),
                )
                self.gas_code.evolve_model(
                    time
                    # self.gas_code.model_time
                    # + self.gas_code.parameters.timestep
                )
            self.system.evolve_model(time)

            if self.cooling:
                logger.info("Second cooling gas...")
                self.cooling.evolve_for(dt_cooling/2)

            logger.info(
                "Evo time is now %s", self.evo_code.model_time.in_(units.Myr)
            )
            logger.info(
                "Bridge time is now %s", (
                    self.system.model_time.in_(units.Myr)
                )
            )
            self.evo_code_to_model.copy()
            # Check for stopping conditions
        self.gas_code_to_model.copy()
        self.star_code_to_model.copy()

    def get_gravity_at_point(self, *args, **kwargs):
        force = VectorQuantity(
            [0, 0, 0],
            unit=units.m * units.s**-2,
        )
        for parent in self.objects:
            force += parent.get_gravity_at_point(*args, **kwargs)
        return force

    # @property
    # def particles(self):
    #     return self.code.particles

    @property
    def model_time(self):
        "Return time of the system"
        return self.system.model_time


def main():
    "Simulate an embedded star cluster (sph + dynamics + evolution)"
    import numpy
    from amuse.ic.plummer import new_plummer_model
    from amuse.ic.salpeter import new_salpeter_mass_distribution
    from amuse.ext.molecular_cloud import molecular_cloud

    numpy.random.seed(14)

    logging_level = logging.INFO
    logging.basicConfig(
        filename="embedded_star_cluster_info.log",
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

    number_of_stars = 10
    stellar_masses = new_salpeter_mass_distribution(number_of_stars)
    star_converter = nbody_system.nbody_to_si(
        stellar_masses.sum() + (50000 | units.MSun),
        10 | units.parsec,
    )
    stars = new_plummer_model(
        number_of_stars,
        convert_nbody=star_converter,
    )[:2]
    stars.mass = 1 | units.MSun  # stellar_masses

    gas_converter = nbody_system.nbody_to_si(
        50000 | units.MSun,
        10 | units.parsec,
    )
    gas = molecular_cloud(targetN=500000, convert_nbody=gas_converter).result

    model = ClusterInPotential(
        stars=stars,
        gas=gas,
        star_converter=star_converter,
        gas_converter=gas_converter,
    )

    timestep = 0.02 | units.Myr
    for step in range(500):
        time = step * timestep
        while model.model_time < time - timestep/2:
            model.evolve_model(time)
        print(
            "Evolved to %s" % model.model_time.in_(units.Myr)
        )
        print(
            "Most massive star: %s" % (
                model.star_particles.mass.max().in_(units.MSun)
            )
        )
        print(
            "Stars centre of mass: %s" % (
                model.star_particles.center_of_mass().in_(units.parsec)
            )
        )
        print(
            "Gas centre of mass: %s" % (
                model.gas_particles.center_of_mass().in_(units.parsec)
            )
        )
        plotname = "embedded-test4-%04i.png" % (step)
        logger.info("Creating plot")
        plot_hydro_and_stars(
            model.model_time,
            model.gas_code,
            model.star_particles,
            L=40,
            filename=plotname,
            title="time = %06.2f %s" % (
                model.gas_code.model_time.value_in(units.Myr),
                units.Myr,
            ),
            gasproperties=["density", "temperature"],
            colorbar=True,
            alpha_sfe=model.alpha_sfe,
        )
        if step%10 == 0:
            write_set_to_file(
                model.gas_particles,
                "gas-%04i.hdf5" % step,
                "amuse",
            )
            write_set_to_file(
                model.star_particles,
                "stars-%04i.hdf5" % step,
                "amuse",
            )

    return


if __name__ == "__main__":
    main()
