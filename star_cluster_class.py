"Class for a star cluster with dynamics and evolution"
from __future__ import print_function, division
import logging
from amuse.units import units
from stellar_evolution_class import (
    # StellarEvolution,
    StellarEvolutionCode,
)
from stellar_dynamics_class import (
    # StellarDynamics,
    StellarDynamicsCode,
)


# class StarClusterCode(
#         object,
# ):
#     """
#     Stellar cluster code object.
#     This wraps around a stellar dynamics and a stellar evolution code to handle
#     both aspects.
#     The stellar dynamics code is the "main" code here.
#     """
# 
#     def __init__(
#             self,
#             converter=None,
#             logger=None,
#             **keyword_arguments
#     ):
#         self.logger = logger or logging.getLogger(__name__)
# 
#         self.logger.debug("Initialising stellar dynamics")
#         self.star_code = StellarDynamicsCode(
#             converter=converter,
#             **keyword_arguments
#         )
#         self.logger.debug("Initialised stellar dynamics")
#         self.logger.debug("Initialising stellar evolution")
#         self.evo_code = StellarEvolutionCode(
#             **keyword_arguments
#         )
#         self.logger.debug("Initialised stellar evolution")
#         self.evo_code_to_star_code_attributes = ["mass", "radius"]
#         self.star_code_to_evo_code_attributes = ["mass", "radius"]
# 
#     @property
#     def model_time(self):
#         "Return the minimum time of the star and evo code times"
#         if self.star_code.model_time < self.evo_code.model_time:
#             return self.star_code.model_time
#         return self.evo_code.model_time
# 
#     @property
#     def particles(self):
#         return self.star_code.particles
# 
#     def sync_to_evolution(self):
#         channel = self.particles.new_channel_to(
#             self.evo_code.particles,
#         )
#         channel.copy_attributes(self.evo_code_attributes)
# 
#     def sync_from_evolution(self):
#         channel = self.evo_code.particles.new_channel_to(
#             self.particles,
#         )
#         channel.copy_attributes(self.star_code_attributes)
# 
#     def evolve_model(self, tend):
#         """
#         Evolve gravity and stellar evolution to specified time, making sure
#         that the changes to the stars by stellar evolution are included over
#         the proper timescales.
#         """
#         time = self.model_time
#         self.model_to_evo_code.copy()
#         while self.model_time < tend:
#             self.model_to_star_code.copy()
#             timestep = self.evo_code.particles.time_step.min()
#             time = min(
#                 time+timestep,
#                 tend,
#             )
#             self.evo_code.evolve_model(time)
#             self.star_code.evolve_model(time)
#             self.evo_code_to_model.copy()
#             # Check stopping conditions
#         self.star_code_to_model.copy()
# 
#     def stop(self):
#         "Stop star_code and evo_code"
#         self.star_code.stop()
#         self.evo_code.stop()


class StarCluster(
        object,
        # StellarDynamics,
        # StellarEvolution,
):
    "Stellar cluster object"

    def __init__(
            self,
            stars=None,
            converter=None,
            epsilon=0.1 | units.parsec,
            logger=None,
            start_time=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initialising StellarDynamics")
        self.star_code = StellarDynamicsCode(
            converter=converter,
        )
        self.star_code.particles.add_particles(stars)
        self.logger.info("Initialised StellarDynamics")
        self.logger.info("Initialising StellarEvolution")
        self.evo_code = StellarEvolutionCode(redirection="none")
        self.evo_code.particles.add_particles(stars)
        self.logger.info("Initialised StellarEvolution")
        self.setup_channels()

    def setup_channels(self):
        self.model_to_evo_code = self.star_particles.new_channel_to(
            self.evo_code.particles,
        )
        self.model_from_evo_code = self.evo_code.particles.new_channel_to(
            self.star_particles,
        )

    @property
    def model_time(self):
        "Return the minimum time of the star and evo code times"
        if self.star_code.model_time < self.evo_code.model_time:
            return self.star_code.model_time
        return self.evo_code.model_time

    # @property
    # def star_particles(self):
    #     return self.star_code.particles

    # @property
    # def particles(self):
    #     return self.star_code.particles

    def evolve_model(self, tend):
        """
        Evolve gravity and stellar evolution to specified time, making sure
        that the changes to the stars by stellar evolution are included over
        the proper timescales.
        """
        time = self.model_time
        # self.model_to_evo_code.copy()
        while self.model_time < tend:
            timestep = self.evo_code.particles.time_step.min()
            time = min(
                time+timestep,
                tend,
            )
            self.evo_code.evolve_model(time)
            self.star_code.evolve_model(time)
            self.model_from_evo_code.copy_attributes(
                ["radius", "mass"],
            )
            # Check stopping conditions

    def get_gravity_at_point(self, *list_arguments, **keyword_arguments):
        """Return gravity at specified point"""
        return self.star_code.get_gravity_at_point(
            *list_arguments, **keyword_arguments
        )

    def get_potential_at_point(self, *list_arguments, **keyword_arguments):
        """Return potential at specified point"""
        return self.star_code.get_potential_at_point(
            *list_arguments, **keyword_arguments
        )

    def stop(self, *list_arguments, **keyword_arguments):
        """Stop codes"""
        self.star_code.stop(*list_arguments, **keyword_arguments)
        self.evo_code.stop(*list_arguments, **keyword_arguments)
        return


def main():
    "Simulate a star cluster (dynamics + evolution)"
    import sys
    from amuse.units import nbody_system

    if len(sys.argv) > 1:
        from amuse.io import read_set_from_file
        stars = read_set_from_file(sys.argv[1], "amuse")
        converter = nbody_system.nbody_to_si(
            stars.mass.sum(),
            3 | units.parsec,
        )
    else:
        import numpy
        from amuse.ic.plummer import new_plummer_model
        from amuse.ic.salpeter import new_salpeter_mass_distribution

        numpy.random.seed(12)
        converter = nbody_system.nbody_to_si(
            1000 | units.MSun,
            3 | units.parsec,
        )
        stars = new_plummer_model(1000, convert_nbody=converter)
        stars.mass = new_salpeter_mass_distribution(len(stars))

    model = StarCluster(stars=stars, converter=converter)

    timestep = 0.1 | units.Myr
    for step in range(10):
        time = step * timestep
        model.evolve_model(time)
        print(
            "Evolved to %s." % model.model_time.in_(units.Myr),
            "Shortest stellar timestep is %s." % (
                model.evo_code.particles.time_step.min(),
            )
        )
        print(
            "Most massive star is %s" % (
                model.particles.mass.max().in_(units.MSun)
            )
        )
        print(
            "Centre-of-mass is at %s" % (
                model.particles.center_of_mass()
            )
        )


if __name__ == "__main__":
    main()
