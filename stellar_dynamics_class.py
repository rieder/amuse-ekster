"Class for stellar dynamics"
from __future__ import print_function, division
from amuse.units import units, nbody_system
from amuse.datamodel import Particles


class StellarDynamics(object):
    """Stellar cluster object"""

    def __init__(
            self,
            stars=None,
            converter=None,
            star_code=None,
            epsilon=0.1 | units.parsec,
    ):
        if converter is not None:
            self.star_converter = converter
        else:
            mass_scale = stars.mass.sum()
            # Change length scale to something related to 'stars'
            length_scale = 1 | units.parsec
            self.star_converter = nbody_system.nbody_to_si(
                mass_scale,
                length_scale,
            )
        if stars is None:
            self.star_particles = Particles()
        else:
            self.star_particles = stars
        number_of_workers = 4  # Relate this to number of processors available?
        if star_code is None:
            from amuse.community.ph4.interface import ph4
            self.star_code = ph4(
                self.star_converter,
                number_of_workers=number_of_workers,
                # channel_type="sockets",
                # redirection="none",
            )
        else:
            self.star_code = star_code
        self.star_code.parameters.epsilon_squared = (epsilon)**2
        # self.star_code.parameters.timestep_parameter = 0.14
        # self.star_code.parameters.block_steps = 1
        print(self.star_code.parameters)
        self.star_code.particles.add_particles(self.star_particles)
        self.model_to_star_code = self.star_particles.new_channel_to(
            self.star_code.particles,
        )
        self.star_code_to_model = self.star_code.particles.new_channel_to(
            self.star_particles,
        )

    @property
    def model_time(self):
        "Return the time of the star code"
        return self.star_code.model_time

    # This is not always supported by the stellar gravity code
    # If it is not, we should do something about that...
    def get_gravity_at_point(self, *args, **kwargs):
        "Return gravity at specified location"
        return self.star_code.get_gravity_at_point(*args, **kwargs)

    @property
    def particles(self):
        "Return particles in star_code"
        return self.star_code.particles

    def evolve_model(self, tend):
        "Evolve gravity to specified time"
        self.model_to_star_code.copy()
        while self.model_time < tend:
            self.star_code.evolve_model(tend)
            # Check stopping conditions
        self.star_code_to_model.copy()

    def stop(self):
        "Stop star_code"
        self.star_code.stop()


def main():
    "Test class with a Plummer sphere"
    import sys
    if len(sys.argv) > 1:
        from amuse.io import read_set_from_file
        stars = read_set_from_file(sys.argv[1], "amuse")
        converter = nbody_system.nbody_to_si(
            stars.mass.sum(),
            3 | units.parsec,
        )
    else:
        from amuse.ic.plummer import new_plummer_model
        converter = nbody_system.nbody_to_si(
            1000 | units.MSun,
            3 | units.parsec,
        )
        stars = new_plummer_model(1000, convert_nbody=converter)

    model = StellarDynamics(stars=stars, converter=converter)
    print(model.star_code.parameters)
    timestep = 0.1 | units.Myr
    for step in range(10):
        time = step * timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time.in_(units.Myr))


if __name__ == "__main__":
    main()
