"Class for a stellar evolution object"
from __future__ import print_function, division
import logging
from amuse.datamodel import Particles
from amuse.units import units

try:
    from amuse.community.sse.interface import SSE
except ImportError:
    SSE = None
try:
    from amuse.community.seba.interface import SeBa
except ImportError:
    SeBa = None


class StellarEvolutionCode(
        object,
):
    "Stellar evolution object"

    def __init__(
            self,
            evo_code=SeBa,
            logger=None,
            begin_time=0 | units.Myr,
            **keyword_arguments
    ):
        self.typestr = "Evolution"
        self.namestr = evo_code.__name__
        self.logger = logger or logging.getLogger(__name__)
        self.__evo_code = evo_code

        if evo_code is SSE:
            self.code = SSE(
                # channel_type="sockets",
                **keyword_arguments
            )
        elif evo_code is SeBa:
            self.code = SeBa(
                **keyword_arguments
            )
        else:
            self.code = evo_code(
                **keyword_arguments
            )
        self.parameters = self.code.parameters
        if begin_time is None:
            begin_time = 0 | units.Myr
        self.__begin_time = begin_time

    @property
    def particles(self):
        "return particles in the evolution code"
        return self.code.particles

    @property
    def model_time(self):
        "return the time of the evolution code"
        return self.code.model_time + self.__begin_time

    def evolve_model(self, tend):
        "Evolve stellar evolution to time and sync"
        return self.code.evolve_model(tend)

    def save_state(self):
        """
        Store current settings
        """
        self.__state["parameters"] = self.code.parameters.copy()
        self.__state["evo_code"] = self.__evo_code
        self.__state["model_time"] = self.code.model_time
        self.__begin_time = self.model_time

    def stop(
            self,
            save_state=False,
    ):
        "Stop stellar evolution code"
        return self.code.stop()


class StellarEvolution(
        object,
):
    "Stellar evolution object"

    def __init__(
            self,
            stars=None,
            evo_code=None,
            logger=None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        if stars is None:
            self.star_particles = Particles()
        else:
            self.star_particles = stars
        if evo_code is None:
            self.evo_code = SSE(
                # channel_type="sockets",
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
    import sys
    import numpy
    from amuse.units import units

    numpy.random.seed(11)

    try:
        from amuse_masc import make_a_star_cluster
        use_masc = True
    except ImportError:
        use_masc = False

    if len(sys.argv) > 1:
        from amuse.io import read_set_from_file
        stars = read_set_from_file(sys.argv[1], "amuse")
    elif use_masc:
        stars = make_a_star_cluster.new_cluster(number_of_stars=10001)
    else:
        from amuse.ic.salpeter import new_salpeter_mass_distribution
        stars = Particles(10000)
        stars.mass = new_salpeter_mass_distribution(len(stars))
    print("Number of stars: %i" % (len(stars)))

    code = StellarEvolutionCode(evo_code=SeBa)
    code.particles.add_particles(stars)
    print(code.parameters)
    timestep = 0.2 | units.Myr
    for step in range(10):
        time = step * timestep
        code.evolve_model(time)
        print("Evolved to %s" % code.model_time.in_(units.Myr))
        print("Most massive star: %s" % code.particles.mass.max())


def _main():
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
