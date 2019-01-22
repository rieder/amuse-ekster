"Class for a stellar evolution object"
from __future__ import print_function, division
import logging
from amuse.datamodel import Particles

logger = logging.getLogger(__name__)


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
            from amuse.community.sse.interface import SSE
            self.evo_code = SSE(
            #     channel_type="sockets"
            )
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
    def __name__(self):
        return "StellarEvolution"

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


def main():
    "Test class with an IMF"
    import numpy
    from amuse.ic.salpeter import new_salpeter_mass_distribution
    from amuse.units import units

    numpy.random.seed(11)

    stars = Particles(10000)
    stars.mass = new_salpeter_mass_distribution(len(stars))
    print("Number of stars: %i" % (len(stars)))

    model = StellarEvolution(stars=stars)
    print(model.evo_code.parameters)
    timestep = 0.2 | units.Myr
    for step in range(10):
        time = step * timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time.in_(units.Myr))
        print("Most massive star: %s" % model.star_particles.mass.max())


if __name__ == "__main__":
    main()
