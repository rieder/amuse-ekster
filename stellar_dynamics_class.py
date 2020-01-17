#!/usr/bin/env python3
"Class for stellar dynamics"
import sys
import logging

try:
    from amuse.community.bhtree.interface import BHTree
except ImportError:
    BHTree = None
try:
    from amuse.community.ph4.interface import ph4
except ImportError:
    ph4 = None
try:
    from amuse.community.phigrape.interface import PhiGRAPE
except ImportError:
    PhiGRAPE = None
try:
    from amuse.community.hermite.interface import Hermite
except ImportError:
    Hermite = None
try:
    from amuse.community.pentacle.interface import Pentacle
except ImportError:
    Pentacle = None

from amuse.datamodel import Particles  # , Particle
from amuse.units import units, nbody_system

import default_settings


class StellarDynamicsCode(object):
    """Wraps around stellar dynamics code, supports collisions"""
    def __init__(
            self,
            converter=None,
            star_code=Pentacle,
            # star_code=Hermite,
            logger=None,
            handle_stopping_conditions=False,
            epsilon_squared=(default_settings.epsilon_stars)**2,
            # mode="cpu",
            begin_time=0 | nbody_system.time,
            stop_after_each_step=False,
            **kwargs
    ):
        self.typestr = "Nbody"
        self.star_code = star_code
        self.namestr = self.star_code.__name__
        self.__name__ = "StellarDynamics"
        self.logger = logger or logging.getLogger(__name__)
        self.handle_stopping_conditions = \
            handle_stopping_conditions
        self.__current_state = "stopped"
        self.__state = {}
        self.__particles = Particles()
        self.__stop_after_each_step = stop_after_each_step
        if converter is not None:
            self.unit_converter = converter
        else:
            self.unit_converter = nbody_system.nbody_to_si(
                default_settings.star_mscale,
                default_settings.star_rscale,
            )
            # TODO: modify to allow N-body units

        if begin_time is None:
            begin_time = 0. | units.Myr
        self.__begin_time = self.unit_converter.to_si(begin_time)

        self.code = self.new_code(
            converter=self.unit_converter,
            star_code=star_code,
            epsilon_squared=epsilon_squared,
            **kwargs)
        if self.__stop_after_each_step:
            # self.code.commit_particles()
            self.stop(save_state=True)
            # print("Stopped/saved")
        else:
            self.save_state()

    def new_code(
            self,
            converter=None,
            star_code=Hermite,
            epsilon_squared=(default_settings.epsilon_stars)**2,
            redirection="null",
            mode="cpu",
            # handle_stopping_conditions=False,
            **kwargs
    ):
        number_of_workers = 1
        if star_code is ph4:
            code = star_code(
                converter,
                mode=mode,
                redirection=redirection,
                number_of_workers=number_of_workers,
                **kwargs
            )
            param = code.parameters
            # Set the parameters explicitly to some default
            # param.block_steps = False

            # Force ph4 to synchronise to the exact time requested - important
            # for Bridge!
            param.force_sync = True

            # param.gpu_id = something
            param.initial_timestep_fac = 0.0625
            param.initial_timestep_limit = 0.03125
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
        elif star_code is Hermite:
            code = star_code(
                converter,
                number_of_workers=number_of_workers,
                redirection=redirection,
            )
            param = code.parameters

            # Force Hermite to sync to the exact time requested - see force_sync for ph4
            param.end_time_accuracy_factor = 0
        elif star_code is PhiGRAPE:
            code = star_code(
                converter,
                number_of_workers=number_of_workers,
                redirection=redirection,
            )
            param = code.parameters
        elif star_code is BHTree:
            code = star_code(
                converter,
                redirection=redirection,
            )
            param = code.parameters
        elif star_code is Pentacle:
            code = star_code(
                converter,
                redirection=redirection,
            )
            param = code.parameters
            param.time_step = 0.5 * default_settings.timestep
        param.epsilon_squared = epsilon_squared
        self.__current_state = "started"
        return code

    def evolve_model(self, end_time):
        """
        Evolve model, handle collisions when they occur
        """
        # print("Evo step")
        if self.__stop_after_each_step:
            # print("Code will be stopped after each step")
            if self.__current_state == "stopped":
                # print("Code is currently stopped - restarting")
                self.restart()
        # collision_detection = \
        #     self.code.stopping_conditions.collision_detection
        # collision_detection.enable()
        # ph4 has a dynamical timestep, so it will stop on or slightly after
        # 'end_time'
        result = 0
        time_unit = end_time.unit
        time_fraction = 1 | units.s
        while self.model_time < (end_time-self.__begin_time):
            print("%s < (%s-%s), continuing" % (self.model_time.in_(time_unit), end_time.in_(time_unit), self.__begin_time.in_(time_unit)))
            if self.model_time >= (end_time - self.__begin_time - time_fraction):
                print("but %s >= (%s-%s-%s), not continuing" % (self.model_time.in_(time_unit), end_time.in_(time_unit), self.__begin_time.in_(time_unit), time_fraction.in_(time_unit)))
                break
            # print("step", end_time, self.__begin_time)
            print("Starting evolve_model of stellar_dynamics")
            result = self.code.evolve_model(
                end_time-self.__begin_time
            )
            print("Finished evolve_model of stellar_dynamics")
            # print("step done")
            # while collision_detection.is_set():
            #     # If we don't handle stopping conditions, return instead
            #     if self.handle_stopping_conditions:
            #         self.resolve_collision(collision_detection)
            #         result = self.code.evolve_model(code_dt)
            #     else:
            #         return result
        # print(
        #     "Reached BT=%s MT=%s CT=%s Nc=%s Nm=%s" % (
        #         self.__begin_time,
        #         self.model_time,
        #         self.code.model_time,
        #         len(self.code.particles),
        #         len(self.particles),
        #     )
        # )

        if self.__stop_after_each_step:
            # print("Now stopping code")
            self.stop(save_state=True)
        return result

    # def resolve_collision(self, collision_detection):
    #     "Determine how to solve a collision and resolve it"
    #     coll = collision_detection
    #     for i, primary in enumerate(coll.particles(0)):
    #         secondary = coll.particles(1)[i]
    #         # Optionally, we could do something else.
    #         # For now, we are just going to merge.
    #         self.merge_two_stars(primary, secondary)

    # def merge_two_stars(self, primary, secondary):
    #     "Merge two colliding stars into one new one"
    #     massunit = units.MSun
    #     colliders = Particles()
    #     colliders.add_particle(primary)
    #     colliders.add_particle(secondary)
    #     new_particle = Particle()
    #     new_particle.mass = colliders.mass.sum()
    #     new_particle.position = colliders.center_of_mass()
    #     new_particle.velocity = colliders.center_of_mass_velocity()
    #     # This should/will be calculated by stellar evolution
    #     new_particle.radius = colliders.radius.max()
    #     # new_particle.age = max(colliders.age)
    #     new_particle.parents = colliders.key
    #     # This should not just be the oldest or youngest.
    #     # But youngest seems slightly better.
    #     new_particle.age = colliders.age.min()
    #     new_particle.birth_age = colliders.birth_age.min()
    #     self.particles.add_particle(new_particle)
    #     self.logger.info(
    #         "Two stars (M1=%s, M2=%s %s) collided at t=%s",
    #         colliders[0].mass.value_in(massunit),
    #         colliders[1].mass.value_in(massunit),
    #         massunit,
    #         self.model_time,
    #     )
    #     self.particles.remove_particles(colliders)

    #     return

    @property
    def begin_time(self):
        """Return begin_time"""
        begin_time = self.__begin_time
        return begin_time

    @property
    def model_time(self):
        """Return code model_time"""
        if self.__current_state != "stopped":
            time = self.__begin_time + self.code.model_time
            return time
        time = self.__begin_time
        return time

    @property
    def particles(self):
        """Return particles"""
        if self.__stop_after_each_step:
            return self.__particles
        # if self.__current_state is not "stopped":
        #     return self.code.particles
        else:
            return self.code.particles

    @property
    def parameters(self):
        """Return code parameters"""
        if self.__current_state != "stopped":
            parameters = self.code.parameters
        else:
            parameters = self.__state["parameters"]
        return parameters

    # TODO: make sure this parameter set is synchronised with code.parameters
    # def parameters(self):
    #     """Return code parameters"""
    #     self.__parameters = self.code.parameters.copy()
    #     return self.__parameters

    @property
    def stopping_conditions(self):
        """Return stopping conditions for dynamics code"""
        return self.code.stopping_conditions

    def get_gravity_at_point(self, *list_arguments, **keyword_arguments):
        """Return gravity at specified point"""
        return self.code.get_gravity_at_point(
            *list_arguments, **keyword_arguments
        )

    def get_potential_at_point(self, *list_arguments, **keyword_arguments):
        """Return potential at specified point"""
        return self.code.get_potential_at_point(
            *list_arguments, **keyword_arguments
        )

    def save_state(self):
        """
        Store current settings
        """
        self.__state["parameters"] = self.code.parameters.copy()
        self.__state["converter"] = self.unit_converter
        self.__state["star_code"] = self.star_code
        self.__state["model_time"] = self.code.model_time
        self.__state["redirection"] = "null"  # FIXME
        self.__state["mode"] = "cpu"  # FIXME
        self.__state["handle_stopping_conditions"] = \
            self.handle_stopping_conditions
        self.__begin_time = self.model_time

    def save_particles(self):
        """
        Store the current particleset, but keep the same particleset!
        """
        self.__particles.remove_particles(self.__particles)
        self.__particles.add_particles(self.code.particles)

    def stop_and_restart(self):
        """
        Store current settings and restart gravity code from saved state
        """
        self.stop(save_state=True)
        self.restart()

    def restart(self):
        """
        Restart gravity code from saved state
        """
        # print("Restarting")
        self.code = self.new_code(
            converter=self.__state["converter"],
            star_code=self.__state["star_code"],
            redirection=self.__state["redirection"],
            mode=self.__state["mode"],
            handle_stopping_conditions=self.__state[
                "handle_stopping_conditions"],
        )
        self.code.particles.add_particles(
            self.__particles
        )
        self.code.parameters.reset_from_memento(
            self.__state["parameters"]
        )
        self.__current_state = "restarted"

    def stop(
            self,
            save_state=True,
            **keyword_arguments
    ):
        """Stop code"""
        # print(
        #     "Stopping at BT=%s MT=%s CT=%s" % (
        #         self.__begin_time,
        #         self.model_time,
        #         self.code.model_time,
        #     )
        # )
        if save_state:
            self.save_state(**keyword_arguments)
            self.save_particles(**keyword_arguments)
        stopcode = self.code.stop(**keyword_arguments)
        self.__current_state = "stopped"
        return stopcode


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
        # print(self.star_code.parameters)
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
    import numpy
    numpy.random.seed(52)
    try:
        from amuse.ext.masc import new_star_cluster
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
        stars = new_star_cluster(number_of_stars=1000)
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

    for stop in [True, False]:
        code = StellarDynamicsCode(
            code=Pentacle, converter=converter,
            stop_after_each_step=stop,
        )
        code.particles.add_particles(stars)
        # print(code.parameters)
        timestep = default_settings.timestep
        cumulative_time = 0. * timestep
        for step in range(10):
            time = step * timestep
            cumulative_time += time
            code.evolve_model(time)
            # print("Evolved to %s" % code.model_time.in_(units.Myr))
            print(
                code.model_time.in_(units.Myr),
                code.particles[0].x.in_(units.parsec),
                code.particles[0].vx.in_(units.kms),
                cumulative_time.in_(units.Myr),
                # code.code.model_time.in_(units.Myr),
                code.begin_time.in_(units.Myr),
            )
        # print(code.particles[0])
        print("\n\n")


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
    # print(model.star_code.parameters)
    timestep = 0.1 | units.Myr
    for step in range(10):
        time = step * timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time.in_(units.Myr))


if __name__ == "__main__":
    main()
