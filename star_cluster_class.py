"Class for a star cluster with dynamics and evolution"
from amuse.units import units
from stellar_evolution_class import StellarEvolution
from stellar_dynamics_class import StellarDynamics


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
        StellarDynamics.__init__(
            self, stars=stars, converter=converter, epsilon=epsilon,
        )
        StellarEvolution.__init__(
            self, stars=stars,
        )

    @property
    def model_time(self):
        "Return the minimum time of the star and evo code times"
        return min(
            StellarDynamics.model_time,
            StellarEvolution.model_time,
        )

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
