"Class for a star cluster embedded in a tidal field and a gaseous region"
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
    ):
        self.objects = (StarCluster, Gas)
        mass_scale_stars = stars.mass.sum()
        length_scale_stars = 1 | units.parsec
        converter_for_stars = nbody_system.nbody_to_si(
            mass_scale_stars,
            length_scale_stars,
        )
        StarCluster.__init__(
            self, stars=stars, converter=converter_for_stars, epsilon=epsilon,
        )

        mass_scale_gas = gas.mass.sum()
        length_scale_gas = 10 | units.parsec
        converter_for_gas = nbody_system.nbody_to_si(
            mass_scale_gas,
            length_scale_gas,
        )
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
            # self.star_code,
            StarCluster,
            [
                self.tidal_field,
                # self.gas_code,
                Gas,
            ],
            True
        )
        self.system.add_system(
            Gas,
            # self.gas_code,
            [
                self.tidal_field,
                # self.star_code,
                StarCluster,
            ],
            True
        )
        self.system.timestep = 0.005 | units.Myr

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
        while self.model_time < tend:
            self.system.evolve_model(tend)
            # Check for stopping conditions

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
