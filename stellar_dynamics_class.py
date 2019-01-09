"Class for stellar dynamics"
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
        number_of_workers = 2  # Relate this to number of processors available?
        if star_code is None:
            from amuse.community.ph4.interface import ph4
            self.star_code = ph4(
                self.star_converter, number_of_workers=number_of_workers,
            )
        else:
            self.star_code = star_code
        self.star_code.parameters.epsilon_squared = (epsilon)**2
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
