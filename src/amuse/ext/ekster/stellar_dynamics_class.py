#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for stellar dynamics
"""
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
try:
    from amuse.community.petar.interface import Petar
except ImportError:
    Petar = None

from amuse.datamodel import Particles  # , Particle
from amuse.units import units, nbody_system

from amuse.ext.ekster import available_codes
from amuse.ext.ekster import ekster_settings


class StellarDynamicsCode:
    """Wraps around stellar dynamics code, supports collisions"""
    def __init__(
            self,
            converter=None,
            # star_code=ph4,
            star_code=Petar,
            # star_code=Hermite,
            logger=None,
            handle_stopping_conditions=False,
            # mode="cpu",
            time_offset=0 | nbody_system.time,
            stop_after_each_step=False,
            number_of_workers=8,
            settings=None,
            **kwargs
    ):
        self.__name__ = "StellarDynamics"
        self.logger = logger or logging.getLogger(__name__)
        if settings is None:
            from ekster.ekster_settings import Settings
            settings = Settings()
            print("WARNING: using default settings!")
            logger.info("WARNING: using default settings!")
        self.settings = settings
        epsilon_squared = settings.epsilon_stars**2
        self.typestr = "Nbody"
        self.star_code = star_code
        try:
            self.namestr = self.star_code.__name__
        except AttributeError:
            self.namestr = "unknown name"
        self.handle_stopping_conditions = \
            handle_stopping_conditions
        self.__current_state = "stopped"
        self.__state = {}
        self.__particles = Particles()
        self.__stop_after_each_step = (
            stop_after_each_step if self.star_code is not Petar else False
        )
        if converter is not None:
            self.unit_converter = converter
        else:
            self.unit_converter = nbody_system.nbody_to_si(
                settings.star_mscale,
                settings.star_rscale,
            )
            # TODO: modify to allow N-body units

        if time_offset is None:
            time_offset = 0. | units.Myr
        self.__time_offset = self.unit_converter.to_si(time_offset)

        self.code = self.new_code(
            converter=self.unit_converter,
            star_code=star_code,
            epsilon_squared=epsilon_squared,
            number_of_workers=number_of_workers,
            **kwargs)
        self.parameters_to_default(
            star_code=star_code,
        )
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
            redirection=None,
            mode="cpu",
            number_of_workers=8,
            # handle_stopping_conditions=False,
            **kwargs
    ):
        if redirection is None:
            redirection = self.settings.code_redirection
        if hasattr(available_codes, star_code):
            star_code = getattr(available_codes, star_code)
        if star_code is ph4:
            code = star_code(
                converter,
                mode=mode,
                redirection=redirection,
                number_of_workers=number_of_workers,
                **kwargs
            )
        elif star_code is Hermite:
            code = star_code(
                converter,
                number_of_workers=number_of_workers,
                redirection=redirection,
            )
        elif star_code is PhiGRAPE:
            code = star_code(
                converter,
                number_of_workers=number_of_workers,
                redirection=redirection,
            )
        elif star_code is BHTree:
            code = star_code(
                converter,
                redirection=redirection,
            )
        elif star_code is Pentacle:
            code = star_code(
                converter,
                redirection=redirection,
            )
        elif star_code is Petar:
            code = star_code(
                converter,
                mode=mode,
                redirection=redirection,
                # number_of_workers=number_of_workers,
                **kwargs
            )
        else:
            raise Exception(
                "Code not found: %s" % star_code
            )
        self.__current_state = "started"
        return code

    def parameters_to_default(
            self,
            star_code=Hermite,
    ):
        "Set default parameters"
        settings = self.settings
        epsilon_squared = settings.epsilon_stars**2
        logger = self.logger
        param = self.code.parameters
        param.epsilon_squared = epsilon_squared
        if star_code is ph4:
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
            # Force Hermite to sync to the exact time requested - see
            # force_sync for ph4
            param.end_time_accuracy_factor = 0
        elif star_code is Petar:
            # Set the parameters explicitly to some default
            param.theta = settings.stellar_dynamics_theta
            logger.info("Old r_out value: %s", param.r_out.in_(units.pc))
            param.r_out = settings.stellar_dynamics_r_out
            param.ratio_r_cut = settings.stellar_dynamics_ratio_r_cut
            logger.info("Old r_bin value: %s", param.r_bin.in_(units.pc))
            param.r_bin = settings.stellar_dynamics_r_bin
            # param.r_search_min = 0 | units.pc

            # very small = technically disabled
            param.r_search_min = settings.stellar_dynamics_r_search_min

            param.dt_soft = settings.stellar_dynamics_dt_soft
            # param.dt_soft = self.unit_converter.to_si(
            #     2**-8 | nbody_system.time
            # )
            # settings.timestep_bridge / 4  # 0 | units.Myr
            # param.r_out = 10 * settings.epsilon_stars

            # dt_soft: 9.765625e-06 Myr default: 0.0 Myr
            # epsilon_squared: 0.0001 parsec**2 default: 0.0 parsec**2
            # r_bin: 0.000137167681417 parsec default: 0.0 parsec
            # r_out: 0.00171459601771 parsec default: 0.0 parsec
            # r_search_min: 0.00206443388608 parsec default: 0.0 parsec
            # ratio_r_cut: 0.1 default: 0.1
            #   r_in         = 0.00043686
            #   r_out        = 0.0043686
            #   r_bin        = 0.00034949
            #   r_search_min = 0.0056792
            #   vel_disp     = 0.89469
            #   dt_soft      = 0.00048828

    def evolve_model(self, end_time):
        """
        Evolve model, handle collisions when they occur
        """
        if self.__stop_after_each_step:
            # print("Code will be stopped after each step")
            if self.__current_state == "stopped":
                # print("Code is currently stopped - restarting")
                self.restart()
        result = 0
        time_unit = end_time.unit
        time_fraction = 1 | units.s
        print(
            "START model time: %s -> end_time: %s" % (
                self.model_time.in_(units.Myr),
                end_time.in_(units.Myr),
            )
        )
        self.logger.info(
            "Starting evolve of %s, model time is %s, end time is %s",
            self.__name__,
            self.model_time.in_(time_unit),
            end_time.in_(time_unit),
        )
        while self.model_time < end_time:
            print(
                "%s < %s, continuing" % (
                    self.model_time.in_(time_unit), end_time.in_(time_unit),
                )
            )
            if self.model_time >= (
                    end_time - time_fraction
            ):
                print(
                    "but %s >= (%s-%s), not continuing" % (
                        self.model_time.in_(time_unit),
                        end_time.in_(time_unit),
                        time_fraction.in_(time_unit)
                    )
                )
                break
            if not self.code.particles.is_empty():
                print("Starting evolve_model of stellar_dynamics")
                result = self.code.evolve_model(
                    end_time-self.__time_offset
                )
                print("Finished evolve_model of stellar_dynamics")
            else:
                self.logger.info(
                    "No particles, skipping evolve and readjusting time offset"
                )
                print(
                    "Skipping evolve_model of stellar_dynamics, no particles!"
                )
                self.__time_offset = end_time
                result = 0

        if self.__stop_after_each_step:
            # print("Now stopping code")
            self.stop(save_state=True)
        print(
            "FINISH model time: %s > end_time: %s" % (
                self.model_time.in_(units.Myr),
                end_time.in_(units.Myr),
            )
        )
        self.logger.info(
            "Finishing evolve of %s, model time is %s, end time is %s",
            self.__name__,
            self.model_time.in_(time_unit),
            end_time.in_(time_unit),
        )
        return result

    @property
    def model_time(self):
        """Return code model_time"""
        if self.__current_state != "stopped":
            time = self.code.model_time + self.__time_offset
            return time
        time = self.__last_time
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

    @property
    def commit_particles(self):
        return self.code.commit_particles

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
        self.__last_time = self.model_time

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
        print(self.__state["parameters"])
        if self.star_code is Petar:
            for name in self.__state["parameters"].names():
                if name != "timestep":
                    setattr(
                        self.code.parameters,
                        name,
                        getattr(self.__state["parameters"], name)
                    )
        else:
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
        if save_state:
            self.save_state(**keyword_arguments)
            self.save_particles(**keyword_arguments)
        stopcode = self.code.stop(**keyword_arguments)
        self.__current_state = "stopped"
        return stopcode


def main():
    "Test class with a Plummer sphere"
    import sys
    import numpy
    numpy.random.seed(52)
    settings = ekster_settings.Settings()
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
            star_code=Petar, converter=converter,
            stop_after_each_step=stop,
        )
        code.particles.add_particles(stars)
        # print(code.parameters)
        timestep = settings.timestep
        cumulative_time = 0. * timestep
        for step in range(10):
            time = step * timestep
            cumulative_time += time
            code.evolve_model(time)
            # print("Evolved to %s" % code.model_time.in_(units.Myr))
            print(
                "Outer loop: ",
                code.model_time.in_(units.Myr),
                code.particles[0].x.in_(units.parsec),
                code.particles[0].vx.in_(units.kms),
                cumulative_time.in_(units.Myr),
                # code.code.model_time.in_(units.Myr),
            )
        # print(code.particles[0])
        print("\n\n")


if __name__ == "__main__":
    main()
