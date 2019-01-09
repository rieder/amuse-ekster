"Class for a stellar evolution object"
from amuse.datamodel import Particles


class StellarEvolution(
        object,
):
    "Stellar evolution object"

    def __init__(
            self,
            stars=None,
            evo_code=None,
    ):
        if stars is None:
            self.star_particles = Particles()
        else:
            self.star_particles = stars
        if evo_code is None:
            from amuse.community.seba.interface import SeBa
            self.evo_code = SeBa()
        else:
            self.evo_code = evo_code
        self.evo_code.particles.add_particles(stars)
        self.model_to_evo_code = self.star_particles.new_channel_to(
            self.evo_code.particles,
        )
        self.evo_code_to_model = self.evo_code.particles.new_channel_to(
            self.star_particles,
        )

    @property
    def particles(self):
        "return particles in the evolution code"
        return self.evo_code.particles

    @property
    def model_time(self):
        "return the time of the evolution code"
        return self.evo_code.model_time

    def evolve_model(self, tend):
        "Evolve stellar evolution to time and sync"
        self.model_to_evo_code.copy()
        self.evo_code.evolve_model(tend)
        self.evo_code_to_model.copy()

    def stop(self):
        "Stop stellar evolution code"
        self.evo_code.stop()
