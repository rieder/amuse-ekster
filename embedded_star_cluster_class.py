"Class for a star cluster embedded in a tidal field and a gaseous region"
from __future__ import print_function, division
# from amuse.community.fastkick.interface import FastKick
from amuse.couple.bridge import Bridge
from amuse.datamodel import ParticlesSuperset
from amuse.units import units, nbody_system
from amuse.units.quantities import VectorQuantity

from gas_class import Gas
from star_cluster_class import StarCluster
from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)

Tide = TimeDependentSpiralArmsDiskModel


class ClusterInPotential(
        StarCluster,
        Gas,
):
    """Stellar cluster in an external potential"""

    def __init__(
            self,
            stars=None,
            gas=None,
            epsilon=0.1 | units.parsec,
            star_converter=None,
            gas_converter=None,
    ):
        self.objects = (StarCluster, Gas)
        mass_scale_stars = stars.mass.sum()
        length_scale_stars = 1 | units.parsec
        if star_converter is None:
            converter_for_stars = nbody_system.nbody_to_si(
                mass_scale_stars,
                length_scale_stars,
            )
        else:
            converter_for_stars = star_converter
        StarCluster.__init__(
            self, stars=stars, converter=converter_for_stars, epsilon=epsilon,
        )

        mass_scale_gas = gas.mass.sum()
        length_scale_gas = 10 | units.parsec
        if gas_converter is None:
            converter_for_gas = nbody_system.nbody_to_si(
                mass_scale_gas,
                length_scale_gas,
            )
        else:
            converter_for_gas = gas_converter
        Gas.__init__(
            self, gas=gas, converter=converter_for_gas, epsilon=epsilon,
        )
        self.tidal_field = Tide()

        # self.gravity_field_code = FastKick(
        #     self.star_converter, mode="cpu", number_of_workers=2,
        # )
        # self.gravity_field_code.parameters.epsilon_squared = epsilon**2

        self.system = Bridge()
        self.system.add_system(
            self.star_code,
            partners=[
                self.tidal_field,
                self.gas_code,
            ],
            do_sync=True,
        )
        self.system.add_system(
            self.gas_code,
            partners=[
                self.tidal_field,
                self.star_code,
            ],
            do_sync=True,
        )
        self.system.timestep = 0.005 | units.Myr

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

    def evolve_model(self, tend):
        "Evolve system to specified time"
        self.model_to_evo_code.copy()
        self.model_to_gas_code.copy()
        while self.model_time < tend:
            evo_time = self.evo_code.model_time
            self.model_to_star_code.copy()
            evo_timestep = self.evo_code.particles.time_step.min()
            print(
                "Smallest evo timestep: %s" % evo_timestep.in_(units.Myr)
            )
            time = min(
                evo_time+evo_timestep,
                tend,
            )
            print("Evolving to %s" % time.in_(units.Myr))
            self.evo_code.evolve_model(time)
            self.system.evolve_model(time)
            print(
                "Evo time is now %s" % self.evo_code.model_time.in_(units.Myr)
            )
            print(
                "Bridge time is now %s" % (
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

    numpy.random.seed(13)

    number_of_stars = 4000
    stellar_masses = new_salpeter_mass_distribution(number_of_stars)
    star_converter = nbody_system.nbody_to_si(
        stellar_masses.sum(),
        3 | units.parsec,
    )
    stars = new_plummer_model(number_of_stars, convert_nbody=star_converter)
    stars.mass = stellar_masses

    gas_converter = nbody_system.nbody_to_si(
        10000 | units.MSun,
        10 | units.parsec,
    )
    gas = molecular_cloud(targetN=10000, convert_nbody=gas_converter).result

    model = ClusterInPotential(
        stars=stars,
        gas=gas,
        star_converter=star_converter,
        gas_converter=gas_converter,
    )

    timestep = 0.2 | units.Myr
    for step in range(10):
        time = step * timestep
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

    return


if __name__ == "__main__":
    main()
