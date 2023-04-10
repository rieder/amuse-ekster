#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for a stellar evolution object
"""
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


class StellarEvolutionCode:
    "Stellar evolution object"

    def __init__(
            self,
            evo_code=SeBa,
            logger=None,
            time_offset=0 | units.Myr,
            settings=None,
            **keyword_arguments
    ):
        self.typestr = "Evolution"
        self.namestr = evo_code.__name__
        self.logger = logger or logging.getLogger(__name__)
        self.__evo_code = evo_code
        self.settings = settings

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
        if time_offset is None:
            time_offset = 0 | units.Myr
        self.__time_offset = time_offset

    @property
    def particles(self):
        "return particles in the evolution code"
        return self.code.particles

    @property
    def model_time(self):
        "return the time of the evolution code"
        return self.code.model_time + self.__time_offset

    def evolve_model(self, tend):
        "Evolve stellar evolution to time and sync"
        print("Starting stellar evolution evolve_model to %s" % tend)
        result = self.code.evolve_model(tend-self.__time_offset)
        print("Finished stellar evolution evolve_model to %s" % tend)
        return result

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


if __name__ == "__main__":
    main()
