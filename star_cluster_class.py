"Class for a star cluster with dynamics and evolution"
from __future__ import print_function, division
import logging
from amuse.units import units
from stellar_evolution_class import StellarEvolution
from stellar_dynamics_class import StellarDynamics

logger = logging.getLogger(__name__)


class StarCluster(
        StellarDynamics,
        StellarEvolution,
):
    "Stellar cluster object"

    def __init__(
            self,
            stars=None,
            converter=None,
            epsilon=0.1 | units.parsec,
    ):
        logger.info("Initialising StellarDynamics")
        StellarDynamics.__init__(
            self, stars=stars, converter=converter, epsilon=epsilon,
        )
        logger.info("Initialised StellarDynamics")
        logger.info("Initialising StellarEvolution")
        StellarEvolution.__init__(
            self, stars=stars,
        )
        logger.info("Initialised StellarEvolution")

    @property
    def model_time(self):
        "Return the minimum time of the star and evo code times"
        if self.star_code.model_time < self.evo_code.model_time:
            return self.star_code.model_time
        return self.evo_code.model_time

    @property
    def particles(self):
        return self.star_particles

    def evolve_model(self, tend):
        """
        Evolve gravity and stellar evolution to specified time, making sure
        that the changes to the stars by stellar evolution are included over
        the proper timescales.
        """
        time = self.model_time
        self.model_to_evo_code.copy()
        while self.model_time < tend:
            self.model_to_star_code.copy()
            timestep = self.evo_code.particles.time_step.min()
            time = min(
                time+timestep,
                tend,
            )
            self.evo_code.evolve_model(time)
            self.star_code.evolve_model(time)
            self.evo_code_to_model.copy()
            # Check stopping conditions
        self.star_code_to_model.copy()

    def stop(self):
        "Stop star_code and evo_code"
        self.star_code.stop()
        self.evo_code.stop()


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
                model.star_particles.mass.max().in_(units.MSun)
            )
        )
        print(
            "Centre-of-mass is at %s" % (
                model.star_particles.center_of_mass()
            )
        )


if __name__ == "__main__":
    main()
