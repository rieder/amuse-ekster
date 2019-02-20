"Class for stellar dynamics"
from __future__ import print_function, division
import logging
from amuse.community.ph4.interface import ph4
from amuse.datamodel import Particles, Particle
from amuse.units import units, nbody_system


class StellarDynamicsCode(object):
    """Wraps around stellar dynamics code, supports collisions"""
    def __init__(
            self,
            converter=None,
            code=ph4,
            logger=None,
            handle_stopping_conditions=False,
    ):
        self.typestr = "Nbody"
        self.namestr = code.__name__
        self.logger = logger or logging.getLogger(__name__)
        self.handle_stopping_conditions = handle_stopping_conditions
        if converter is not None:
            self.converter = converter
        else:
            self.converter = nbody_system.nbody_to_si(
                1 | units.MSun,
                1 | units.parsec,
            )

        if code is ph4:
            self.code = code(
                self.converter,
                number_of_workers=1,
                mode="cpu",
            )
            param = self.parameters
            # Set the parameters explicitly to some default
            param.begin_time = 0.0 | units.Myr
            # self.parameters.block_steps = False
            param.epsilon_squared = 0 | units.AU**2
            # param.force_sync = False
            # param.gpu_id = something
            # param.initial_timestep_fac = 0.0625
            # param.initial_timestep_limit = 0.03125
            # param.initial_timestep_median = 8.0
            # param.manage_encounters = 4
            # # We won't use these stopping conditions anyway
            # param.stopping_condition_maximum_density = some HUGE number
            # param.stopping_condition_maximum_internal_energy = inf
            # param.stopping_condition_minimum_density = - huge
            # param.stopping_condition_minimum_internal_energy = - big number
            # param.stopping_conditions_number_of_steps = 1
            # param.stopping_conditions_out_of_box_size = 0 | units.m
            # param.stopping_conditions_out_of_box_use_center_of_mass = True
            # param.stopping_conditions_timeout = 4.0 | units.s
            # param.sync_time = 0.0 | units.s
            # param.timestep_parameter = 0.0
            # param.total_steps = False
            # param.use_gpu = False
            # param.zero_step_mode = False

    def evolve_model(self, end_time):
        """
        Evolve model, handle collisions when they occur
        """
        collision_detection = self.code.stopping_conditions.collision_detection
        collision_detection.enable()
        # ph4 has a dynamical timestep, so it will stop on or slightly after
        # 'end_time'
        result = 0
        while self.model_time < end_time:
            result = self.code.evolve_model(end_time)
            while collision_detection.is_set():
                # If we don't handle stopping conditions, return instead
                if self.handle_stopping_conditions:
                    self.resolve_collision(collision_detection)
                    result = self.code.evolve_model(end_time)
                else:
                    return result
        return result

    def resolve_collision(self, collision_detection):
        "Determine how to solve a collision and resolve it"
        coll = collision_detection
        for i, primary in enumerate(coll.particles(0)):
            secondary = coll.particles(1)[i]
            # Optionally, we could do something else.
            # For now, we are just going to merge.
            self.merge_two_stars(primary, secondary)

    def merge_two_stars(self, primary, secondary):
        "Merge two colliding stars into one new one"
        massunit = units.MSun
        colliders = Particles()
        colliders.add_particle(primary)
        colliders.add_particle(secondary)
        new_particle = Particle()
        new_particle.mass = colliders.mass.sum()
        new_particle.position = colliders.center_of_mass()
        new_particle.velocity = colliders.center_of_mass_velocity()
        # This should/will be calculated by stellar evolution
        new_particle.radius = colliders.radius.max()
        # new_particle.age = max(colliders.age)
        new_particle.parents = colliders.key
        # This should not just be the oldest or youngest.
        # But youngest seems slightly better.
        new_particle.age = colliders.age.min()
        new_particle.birth_age = colliders.birth_age.min()
        self.particles.add_particle(new_particle)
        self.logger.info(
            "Two stars (M1=%s, M2=%s %s) collided at t=%s",
            colliders[0].mass.value_in(massunit),
            colliders[1].mass.value_in(massunit),
            massunit,
            self.model_time,
        )
        self.particles.remove_particles(colliders)

        return

    @property
    def model_time(self):
        """Return code model_time"""
        return self.code.model_time

    @property
    def particles(self):
        """Return code particles"""
        return self.code.particles

    @property
    def parameters(self):
        """Return code parameters"""
        return self.code.parameters

    @property
    def get_gravity_at_point(self, *list_arguments, **keyword_arguments):
        """Return gravity at specified point"""
        return self.code.get_gravity_at_point(
            *list_arguments, **keyword_arguments
        )

    @property
    def get_potential_at_point(self, *list_arguments, **keyword_arguments):
        """Return potential at specified point"""
        return self.code.get_potential_at_point(
            *list_arguments, **keyword_arguments
        )

    @property
    def stop(self, *list_arguments, **keyword_arguments):
        """Stop code"""
        return self.code.stop(*list_arguments, **keyword_arguments)


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
    def __name__(self):
        return "StellarDynamics"

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
    try:
        from amuse_masc import make_a_star_cluster
        use_masc = True
    except ImportError:
        use_masc = False
    if len(sys.argv) > 1:
        from amuse.io import read_set_from_file
        stars = read_set_from_file(sys.argv[1], "amuse")
        converter = nbody_system.nbody_to_si(
            stars.mass.sum(),
            3 | units.parsec,
        )
    elif use_masc:
        stars = make_a_star_cluster.new_cluster(number_of_stars=1000)
        rmax = (stars.position - stars.center_of_mass()).lengths().max()
        converter = nbody_system.nbody_to_si(
            stars.mass.sum(),
            rmax,
        )
    else:
        from amuse.ic.plummer import new_plummer_model
        converter = nbody_system.nbody_to_si(
            1000 | units.MSun,
            3 | units.parsec,
        )
        stars = new_plummer_model(1000, convert_nbody=converter)

    code = StellarDynamicsCode(code=ph4, converter=converter)
    code.particles.add_particles(stars)
    print(code.parameters)
    timestep = 0.1 | units.Myr
    for step in range(10):
        time = step * timestep
        code.evolve_model(time)
        print("Evolved to %s" % code.model_time.in_(units.Myr))


def _main():
    "Test class with a Plummer sphere"
    import sys
    try:
        from amuse_masc import make_a_star_cluster
        use_masc = True
    except ImportError:
        use_masc = False
    if len(sys.argv) > 1:
        from amuse.io import read_set_from_file
        stars = read_set_from_file(sys.argv[1], "amuse")
        converter = nbody_system.nbody_to_si(
            stars.mass.sum(),
            3 | units.parsec,
        )
    elif use_masc:
        stars = make_a_star_cluster.new_cluster(number_of_stars=1000)
        converter = nbody_system.nbody_to_si(
            stars.mass.sum(),
            stars.lagrangian_radii.max()
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
