"Class for a star cluster embedded in a tidal field and a gaseous region"
from __future__ import print_function, division
import logging
import numpy
try:
    from amuse.community.fi.interface import Fi
except ImportError:
    Fi = None
# from amuse.community.bhtree.interface import BHTree
try:
    from amuse.community.fastkick.interface import FastKick
except ImportError:
    FastKick = None
try:
    from amuse.community.hermite.interface import Hermite
except ImportError:
    Hermite = None
try:
    from amuse.community.ph4.interface import ph4
except ImportError:
    ph4 = None
try:
    from amuse.community.pentacle.interface import Pentacle
except ImportError:
    Pentacle = None

from amuse.datamodel import ParticlesSuperset, Particles, Particle
from amuse.units import units, nbody_system  # , constants
from amuse.units.quantities import VectorQuantity

from amuse.io import write_set_to_file

from amuse.ext.masc import new_star_cluster

from gas_class import GasCode
from sinks_class import accrete_gas, should_a_sink_form  # , sfe_to_density
from star_cluster_class import StarCluster
from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)
from plotting_class import plot_hydro_and_stars  # , plot_stars
from merge_recipes import form_new_star
from star_forming_region_class import form_stars  # StarFormingRegion
from bridge import (
    Bridge, CalculateFieldForCodes,
)
# from setup_codes import new_field_code

Tide = TimeDependentSpiralArmsDiskModel


def new_argument_parser():
    "Parse command line arguments"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        dest='starsfilename',
        default=None,
        help='file containing stars (optional) []',
    )
    parser.add_argument(
        '-i',
        dest='sinksfilename',
        default=None,
        help='file containing sinks (optional) []',
    )
    parser.add_argument(
        '-g',
        dest='gasfilename',
        default=None,
        help='file containing gas (optional) []',
    )
    parser.add_argument(
        '-r',
        dest='randomfilename',
        default=None,
        help='file containing random state (optional) []',
    )
    parser.add_argument(
        '-d',
        dest='rundir',
        default="./",
        help='directory to store run in (optional) [./]',
    )
    return parser.parse_args()


class ClusterInPotential(
        StarCluster,
):
    """
    Stellar cluster in an external potential

    This class builds on StarCluster, and extends it with support for gas, an
    external tidal field, and star formation.
    """

    def __init__(
            self,
            stars=Particles(),
            gas=Particles(),
            sinks=Particles(),
            epsilon=0.1 | units.parsec,
            star_converter=None,
            gas_converter=None,
            logger=None,
            logger_level=None,
            begin_time=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        if logger_level is None:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logger_level)

        self.__begin_time = (
            # begin_time if begin_time is not None else 0.0 | units.Myr
            begin_time
            if begin_time is not None
            else (5.0802 * 1.4874E+15 | units.s)
        )
        self.new_stars_added = False

        # FIXME because this is no longer true
        # This class stores particle data by itself, in contrast to StarCluster
        self.gas_particles = Particles()
        self.dm_particles = Particles()
        self.sink_particles = Particles()
        self.star_particles = Particles()

        mass_scale_stars = 200 | units.MSun  # or stars.mass.sum()
        length_scale_stars = 1 | units.parsec
        if star_converter is None:
            converter_for_stars = nbody_system.nbody_to_si(
                mass_scale_stars,
                length_scale_stars,
            )
        else:
            converter_for_stars = star_converter

        self.logger.info("Initialising StarCluster")
        StarCluster.__init__(
            self,
            # stars=stars,
            converter=converter_for_stars,
            epsilon=epsilon,
            # star_code=Hermite,
            # star_code=Pentacle,
            star_code=ph4,
            # begin_time=self.__begin_time,
        )
        self.add_stars(stars)
        self.logger.info("Initialised StarCluster")

        if not gas.is_empty():
            mass_scale_gas = gas.mass.sum()
            length_scale_gas = 100 | units.parsec
            if gas_converter is None:
                converter_for_gas = nbody_system.nbody_to_si(
                    mass_scale_gas,
                    length_scale_gas,
                )
            else:
                converter_for_gas = gas_converter
            self.logger.info("Initialising Gas")
            new_gas_converter = nbody_system.nbody_to_si(
                gas.total_mass(),
                100 | units.parsec,
            )
            self.gas_code = GasCode(
                converter=new_gas_converter,
                # begin_time=self.__begin_time,
            )

            self.add_gas(gas)
            self.add_sinks(sinks)
            print(self.gas_code.parameters)
            self.timestep = 0.01 | units.Myr
            self.logger.info("Initialised Gas")

        self.logger.info("Creating Tide object")
        self.tidal_field = Tide(t_start=self.__begin_time)
        self.logger.info("Created Tide object")

        self.epsilon = epsilon
        self.converter = converter_for_gas

        def new_field_gravity_code(
                code=FastKick,
                # code=Fi,
        ):
            "Create a new field code"
            print("Creating field code")
            result = code(
                self.converter,
                redirection="null",
                mode="openmp",
            )
            result.parameters.epsilon_squared = self.epsilon**2
            return result

        def new_field_code(
                code,
        ):
            " something"
            result = CalculateFieldForCodes(
                new_field_gravity_code,
                [code],
            )
            return result

        to_gas_codes = []
        if self.tidal_field:
            to_gas_codes.append(self.tidal_field)
        to_gas_codes.append(
            # self.star_code,
            new_field_code(
                self.star_code,
            )
        )
        to_stars_codes = []
        if self.tidal_field:
            to_stars_codes.append(self.tidal_field)
        to_stars_codes.append(
            new_field_code(
                self.gas_code,
            )
        )

        self.system = Bridge(
            timestep=(
                self.timestep
            ),
        )
        self.system.add_system(
            self.star_code,
            partners=to_stars_codes,
            do_sync=True,
        )
        self.system.add_system(
            self.gas_code,
            partners=to_gas_codes,
            do_sync=True,
            h_smooth_is_eps=True,
        )

    def initialise_gas(
            self, gas, converter,
            begin_time=0 | units.yr,
    ):
        self.logger.info("Initialising Gas")
        self.gas_code = GasCode(
            converter=converter,
        )

    def initialise_starcluster(
            self, stars, converter,
            epsilon=0 | units.parsec,
            begin_time=0 | units.yr,
    ):
        self.logger.info("Initialising StarCluster")
        StarCluster.__init__(
            self,
            stars=stars,
            converter=converter,
            epsilon=epsilon,
            begin_time=begin_time,
        )
        self.logger.info("Initialised StarCluster")

    def sync_from_gas_code(self):
        """
        copy relevant attributes changed by gas/sink/dm dynamics
        """
        from_gas_attributes = [
            "x", "y", "z", "vx", "vy", "vz",
            "density", "h_smooth", "u",
        ]
        from_sink_attributes = [
            "x", "y", "z", "vx", "vy", "vz", "mass",
        ]
        from_dm_attributes = [
            "x", "y", "z", "vx", "vy", "vz",
        ]
        channel_from_gas = \
            self.gas_code.gas_particles.new_channel_to(self.gas_particles)
        channel_from_sinks = \
            self.gas_code.sink_particles.new_channel_to(
                self.sink_particles)
        channel_from_dm = \
            self.gas_code.dm_particles.new_channel_to(self.dm_particles)
        channel_from_gas.copy_attributes(
            from_gas_attributes,
        )
        if not self.sink_particles.is_empty():
            channel_from_sinks.copy_attributes(
                from_sink_attributes,
            )
        if not self.dm_particles.is_empty():
            channel_from_dm.copy_attributes(
                from_dm_attributes,
            )

    def sync_to_gas_code(self):
        """
        copy potentially modified attributes that matter to gas/sink/dm
        dynamics
        """
        channel_to_gas = \
            self.gas_particles.new_channel_to(self.gas_code.gas_particles)
        channel_to_sinks = \
            self.sink_particles.new_channel_to(
                self.gas_code.sink_particles)
        channel_to_dm = \
            self.dm_particles.new_channel_to(self.gas_code.dm_particles)
        channel_to_gas.copy_attributes(
            ["x", "y", "z", "vx", "vy", "vz", ]
        )
        if not self.sink_particles.is_empty():
            channel_to_sinks.copy_attributes(
                ["x", "y", "z", "vx", "vy", "vz", "mass", "radius", ]
            )
        if not self.dm_particles.is_empty():
            channel_to_dm.copy_attributes(
                ["x", "y", "z", "vx", "vy", "vz", ]
            )

    def sync_from_evo_code(self):
        """
        copy relevant attributes changed by stellar evolution
        """
        from_stellar_evolution_attributes = [
            "mass", "radius", "luminosity", "temperature", "age",
            "stellar_type"
        ]
        channel_from_star_evo = \
            self.evo_code.particles.new_channel_to(self.star_particles)
        channel_from_star_evo.copy_attributes(
            from_stellar_evolution_attributes
        )

    def sync_to_evo_code(self):
        """
        copy potentially modified attributes that matter to stellar evolution
        """
        to_stellar_evolution_attributes = [
            "mass", "age",
        ]
        channel_to_star_evo = \
            self.star_particles.new_channel_to(self.evo_code.particles)
        channel_to_star_evo.copy_attributes(
            to_stellar_evolution_attributes
        )

    def sync_from_star_code(self):
        """
        copy relevant attributes changed by stellar gravity
        """
        from_stellar_gravity_attributes = [
            "x", "y", "z", "vx", "vy", "vz"
        ]
        channel_from_star_dyn = \
            self.star_code.particles.new_channel_to(self.star_particles)
        channel_from_star_dyn.copy_attributes(
            from_stellar_gravity_attributes
        )

    def sync_to_star_code(self):
        """
        copy potentially modified attributes that matter to stellar gravity
        """
        to_stellar_gravity_attributes = [
            "x", "y", "z", "vx", "vy", "vz", "mass", "radius",
        ]
        channel_to_star_dyn = \
            self.star_particles.new_channel_to(self.star_code.particles)
        channel_to_star_dyn.copy_attributes(
            to_stellar_gravity_attributes
        )

    @property
    def particles(self):
        "Return all particles"
        return ParticlesSuperset(
            self.star_particles,
            self.dm_particles,
            self.sink_particles,
            self.gas_particles,
        )

    @property
    def code(self):
        "Return the main code - the Bridge system in this case"
        return self.system

    def resolve_sink_formation(self):
        "Identify high-density gas, and form sink(s) when needed"
        dump_saved = False
        removed_gas = Particles()
        maximum_density = (
            self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select_array(
            lambda density:
            density > maximum_density,
            ["density"],
        ).copy().sorted_by_attribute("density").reversed()
        print(
            "Number of gas particles above maximum density (%s): %i" % (
                maximum_density.in_(units.g * units.cm**-3),
                len(high_density_gas),
            )
        )
        # for i, origin_gas in enumerate(high_density_gas):
        while not high_density_gas.is_empty():
            i = 0
            origin_gas = high_density_gas[i]
            # print(
            #     "Processing gas %i, %i remain" % (
            #         origin_gas.key, len(high_density_gas)
            #     )
            # )
            # if i >= 1:
            #     # For now, do one sink at a time
            #     break
            if origin_gas in removed_gas:
                # print(
                #     "Gas particle %s removed by accretion, skipping" %
                #     origin_gas.key
                # )
                high_density_gas.remove_particle(origin_gas)
            else:
                form_sink = False
                if origin_gas.density/maximum_density > 10:
                    print(
                        "Gas density is %s (> %s), forming sink" % (
                            origin_gas.density.in_(units.g * units.cm**-3),
                            (10 * maximum_density).in_(units.g * units.cm**-3),
                        )
                    )
                    form_sink = True
                else:
                    try:
                        form_sink, not_forming_message = should_a_sink_form(
                            origin_gas, self.gas_particles)
                    except TypeError:
                        form_sink = False
                        not_forming_message = \
                            "Something went wrong (TypeError)"
                        dump_file_id = numpy.random.randint(100000000)
                        key = origin_gas.key
                        dumpfile = "dump-%08i-stars-key%i.hdf5" % (
                            dump_file_id,
                            key,
                        )
                        if not dump_saved:
                            print("saving gas dump to %s" % dumpfile)
                            write_set_to_file(
                                self.gas_particles, dumpfile, "amuse",
                            )
                            dump_saved = True
                if not form_sink:
                    # print(
                    #     "Gas particle %s not forming a sink: %s"
                    #     % (origin_gas.key, not_forming_message)
                    # )
                    self.logger.info(
                        "Not forming a sink - not meeting the"
                        " requirements: %s"
                        % (not_forming_message)
                    )
                    high_density_gas.remove_particle(origin_gas)
                else:  # try:
                    print("Forming a sink from particle %s" % origin_gas.key)
                    new_sink = Particle()
                    new_sink.initialised = False

                    # Should do accretion here, and calculate the radius from
                    # the average density, stopping when the jeans radius
                    # becomes larger than the accretion radius!

                    # Setting the radius to something that will lead to
                    # >~150MSun per sink would be ideal.  So this radius is
                    # related to the density.
                    minimum_sink_radius = 0.25 | units.parsec
                    desired_sink_mass = 200 | units.MSun
                    desired_sink_radius = max(
                        (desired_sink_mass / origin_gas.density)**(1/3),
                        minimum_sink_radius,
                    )

                    new_sink.radius = desired_sink_radius
                    new_sink.initial_density = origin_gas.density
                    # Average of accreted gas is better but for isothermal this
                    # is fine
                    # new_sink.u = origin_gas.u
                    new_sink.u = 0 | units.kms**2

                    new_sink.accreted_mass = 0 | units.MSun
                    o_x, o_y, o_z = origin_gas.position

                    new_sink.position = origin_gas.position
                    accreted_gas = accrete_gas(new_sink, self.gas_particles)
                    if accreted_gas.is_empty():
                        self.logger.info("Empty gas so no sink %i", i)
                        high_density_gas.remove_particle(origin_gas)
                    else:
                        self.logger.info(
                            "Number of accreted gas particles: %i",
                            len(accreted_gas)
                        )
                        new_sink.position = accreted_gas.center_of_mass()
                        new_sink.velocity = \
                            accreted_gas.center_of_mass_velocity()
                        new_sink.mass = accreted_gas.total_mass()
                        new_sink.accreted_mass = (
                            accreted_gas.total_mass() - origin_gas.mass
                        )
                        # Sink's "internal energy" is the velocity dispersion
                        # of the infalling gas
                        new_sink.u = (
                            accreted_gas.velocity
                            - accreted_gas.center_of_mass_velocity()
                        ).lengths_squared().mean()
                        # TODO: think about preferential directions for this
                        # (radial?)
                        # Alternatively, it could be based on the gas sound
                        # speed
                        # new_sink.u = accreted_gas.u.mean()

                        removed_gas.add_particles(accreted_gas.copy())
                        self.remove_gas(accreted_gas)
                        # Which high-density gas particles are accreted and
                        # should no longer be considered?
                        accreted_high_density_gas = \
                            high_density_gas.get_intersecting_subset_in(
                                accreted_gas
                            )
                        high_density_gas.remove_particles(
                            accreted_high_density_gas
                        )
                        self.add_sink(new_sink)
                        self.logger.info(
                            "Added sink %i with mass %s and radius %s", i,
                            new_sink.mass.in_(units.MSun),
                            new_sink.radius.in_(units.parsec),
                        )
                        # except:
                        #     print("Could not add another sink")

    def add_sink(self, sink):
        self.gas_code.sink_particles.add_particle(sink)
        self.sink_particles.add_particle(sink)

    def add_sinks(self, sinks):
        self.gas_code.sink_particles.add_particles(sinks)
        self.sink_particles.add_particles(sinks)

    def remove_sinks(self, sinks):
        self.gas_code.sink_particles.remove_particles(sinks)
        self.sink_particles.remove_particles(sinks)

    def add_gas(self, gas):
        self.gas_code.gas_particles.add_particles(gas)
        self.gas_particles.add_particles(gas)

    def remove_gas(self, gas_):
        # Workaround: to prevent either from not being removed.
        gas = gas_.copy()
        self.gas_code.gas_particles.remove_particles(gas)
        self.gas_particles.remove_particles(gas)

    def add_stars(self, stars):
        self.star_particles.add_particles(stars)
        self.evo_code.particles.add_particles(stars)
        self.sync_from_evo_code()
        self.star_code.particles.add_particles(stars)

    def remove_stars(self, stars):
        self.star_code.particles.remove_particles(stars)
        self.evo_code.particles.remove_particles(stars)
        self.star_particles.remove_particles(stars)

    def resolve_star_formation(
            self, stop_star_forming_time=10. | units.Myr,
            shrink_sinks=True,
    ):
        if self.model_time >= stop_star_forming_time:
            self.logger.info(
                "No star formation since time > %s" % stop_star_forming_time
            )
            return None
        self.logger.info("Resolving star formation")
        mass_before = (
            (
                self.sink_particles.total_mass()
                if not self.sink_particles.is_empty()
                else (0 | units.MSun)
            )
            + (
                self.star_particles.total_mass()
                if not self.star_particles.is_empty()
                else (0 | units.MSun)
            )
        )
        # max_new_stars_per_timestep = 500
        formed_stars = False
        stellar_mass_formed = 0 | units.MSun
        for i, sink in enumerate(self.sink_particles):
            # TODO: this loop needs debugging/checking...
            self.logger.debug("Processing sink %i", i)
            new_stars = form_stars(
                sink,
                local_sound_speed=self.gas_code.parameters.polyk.sqrt(),
                logger=self.logger,
            )
            if new_stars is not None:
                formed_stars = True
                self.add_stars(new_stars)
                stellar_mass_formed += new_stars.total_mass()

            # formed_stars = False
            # while new_stars is not None:
            #     formed_stars = True
            #     j = 0
            #     self.add_stars(new_stars)
            #     stellar_mass_formed += new_stars.total_mass()
            #     j += 1
            #     if j < max_new_stars_per_timestep:
            #         polyk = self.gas_code.parameters.polyk
            #         new_stars = form_stars(
            #             sink,
            #             local_sound_speed=polyk.sqrt(),
            #             logger=self.logger,
            #         )
            #     else:
            #         new_stars = None
            if shrink_sinks:
                # After forming stars, shrink the sink's (accretion) radius to
                # prevent it from accreting relatively far away gas and moving
                # a lot
                sink.radius = (
                    (sink.mass / sink.initial_density)
                    / (4/3 * numpy.pi)
                )**(1/3)

        mass_after = (
            (
                self.sink_particles.total_mass()
                if not self.sink_particles.is_empty()
                else (0 | units.MSun)
            )
            + (
                self.star_particles.total_mass()
                if not self.star_particles.is_empty()
                else (0 | units.MSun)
            )
        )
        self.logger.debug(
            "dM = %s, mFormed = %s ",
            (mass_after - mass_before).in_(units.MSun),
            stellar_mass_formed.in_(units.MSun),
        )
        self.sync_to_gas_code()
        return formed_stars

    def _resolve_starformation(self):
        self.logger.info("Resolving star formation")

        maximum_density = (
            self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select(
            lambda density:
            density >= maximum_density,
            ["density"],
        )

        stars_formed = 0
        while not high_density_gas.is_empty():
            new_star, absorbed_gas = form_new_star(
                high_density_gas[0],
                self.gas_particles
            )
            print(
                "Removing %i former gas particles to form one star" % (
                    len(absorbed_gas)
                )
            )

            self.logger.debug(
                "Removing %i former gas particles", len(absorbed_gas)
            )
            self.gas_particles.remove_particles(absorbed_gas)
            self.logger.debug(
                "Removed %i former gas particles", len(absorbed_gas)
            )

            new_star.birth_age = self.gas_code.model_time + self.__begin_time
            self.logger.debug("Adding new star to evolution code")
            evo_stars = self.evo_code.particles.add_particle(new_star)
            self.logger.debug("Added new star to evolution code")

            new_star.radius = evo_stars.radius

            self.logger.debug("Adding new star to model")
            self.star_particles.add_particle(new_star)
            self.logger.debug("Added new star to model")
            self.logger.info("Resolved star formation")
            stars_formed += 1
        print("Number of gas particles is now %i" % len(self.gas_particles))
        print("Number of star particles is now %i" % len(self.star_particles))
        print("Number of dm particles is now %i" % len(self.dm_particles))
        print("Formed %i new stars" % stars_formed)

    def resolve_collision(self, collision_detection):
        "Determine how to solve a collision and resolve it"

        coll = collision_detection
        if not coll.particles(0):
            print("No collision found? Disabling collision detection for now.")
            coll.disable()
            return
        collisions_counted = 0
        for i, primary in enumerate(coll.particles(0)):
            secondary = coll.particles(1)[i]
            # Optionally, we could do something else.
            # For now, we are just going to merge.
            self.merge_two_stars(primary, secondary)
            collisions_counted += 1
        print("Resolved %i collisions" % collisions_counted)

    def merge_two_stars(self, primary, secondary):
        "Merge two colliding stars into one new one"
        massunit = units.MSun
        lengthunit = units.RSun
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
        # new_particle.parents = colliders.key

        # Since stellar dynamics code doesn't know about age, add the particles
        # there before we set these. This is a bit of a hack. We should do our
        # own bookkeeping here instead.
        dyn_particle = self.star_particles.add_particle(new_particle)

        # This should not just be the oldest or youngest.
        # But youngest seems slightly better.
        # new_particle.age = colliders.age.min()
        # new_particle.birth_age = colliders.birth_age.min()
        evo_particle = self.evo_code.particles.add_particle(new_particle)
        dyn_particle.radius = evo_particle.radius

        # self.logger.info(
        print(
            "Two stars ("
            "M1=%s, M2=%s, M=%s %s; "
            "R1=%s, R2=%s, R=%s %s"
            ") collided at t=%s" % (
                colliders[0].mass.value_in(massunit),
                colliders[1].mass.value_in(massunit),
                dyn_particle.mass.value_in(massunit),
                massunit,
                colliders[0].radius.value_in(lengthunit),
                colliders[1].radius.value_in(lengthunit),
                dyn_particle.radius.value_in(lengthunit),
                lengthunit,
                self.model_time,
            )
        )
        self.evo_code.particles.remove_particles(colliders)
        self.star_particles.remove_particles(colliders)

        return

    def stellar_feedback(self):
        # Deliberately not doing anything here yet
        return

    def evolve_model(self, real_tend):
        "Evolve system to specified time"
        relative_tend = real_tend - self.__begin_time

        dmax = self.gas_code.parameters.stopping_condition_maximum_density
        Myr = units.Myr

        self.logger.info(
            "Evolving to time %s",
            real_tend,
        )
        # self.model_to_evo_code.copy()
        # self.model_to_gas_code.copy()

        density_limit_detection = \
            self.gas_code.stopping_conditions.density_limit_detection
        # density_limit_detection.enable()
        density_limit_detection.disable()

        # TODO: re-enable at some point?
        # collision_detection = \
        #     self.star_code.stopping_conditions.collision_detection
        # collision_detection.enable()

        # maximum_density = (
        #     self.gas_code.parameters.stopping_condition_maximum_density
        # )

        print("Starting loop")
        time_unit = units.Myr
        print(self.model_time.in_(time_unit), real_tend.in_(time_unit), self.system.timestep.in_(time_unit))
        step = 0
        minimum_steps = 1
        # print(
        #     ((self.model_time + self.begin_time)
        #      < (tend - self.system.timestep))
        #     or (step < minimum_steps)
        # )
        print("Saving initial backup")
        randomstate = numpy.random.get_state()
        numpy.save("randomstate-backup.npy", randomstate)
        if not self.gas_particles.is_empty():
            write_set_to_file(
                self.gas_particles.savepoint(
                    self.gas_code.model_time + self.__begin_time),
                "gas-backup.hdf5",
                "amuse",
                append_to_file=False,
                version='2.0',
                return_working_copy=False,
            )
        if not self.sink_particles.is_empty():
            write_set_to_file(
                self.sink_particles.savepoint(
                    self.gas_code.model_time + self.__begin_time),
                "sinks-backup.hdf5",
                "amuse",
                append_to_file=False,
                version='2.0',
                return_working_copy=False,
            )
        if not self.star_particles.is_empty():
            write_set_to_file(
                self.star_particles.savepoint(
                    self.star_code.model_time + self.__begin_time),
                "stars-backup.hdf5",
                "amuse",
                append_to_file=False,
                version='2.0',
                return_working_copy=False,
            )
        while (
                (self.model_time
                 < (real_tend - self.system.timestep))
                or (step < minimum_steps)
        ):
            step += 1
            # evo_time = self.evo_code.model_time
            # self.model_to_star_code.copy()
            if not self.star_particles.is_empty():
                evo_timestep = self.evo_code.particles.time_step.min()
                self.logger.info(
                    "Smallest evo timestep: %s", evo_timestep.in_(Myr)
                )
            # time = min(
            #     evo_time+evo_timestep,
            #     relative_tend,
            # )
            self.logger.info("Evolving to %s", real_tend.in_(Myr))
            if not self.star_particles.is_empty():
                self.logger.info("Stellar evolution...")
                self.evo_code.evolve_model(real_tend)
                self.sync_from_evo_code()
            # dt_cooling = time - self.gas_code.model_time
            # if self.cooling:
            #     self.logger.info("Cooling gas...")
            #     self.cooling.evolve_for(dt_cooling/2)
            self.logger.info("System...")

            print("Evolving system")
            self.system.evolve_model(relative_tend)
            self.sync_from_gas_code()
            self.sync_from_star_code()
            if (
                    self.gas_code.model_time
                    < (relative_tend - self.system.timestep)
            ):
                print("Evolving gas code a bit more")
                self.gas_code.evolve_model(relative_tend)
                self.sync_from_gas_code()
            print("Evolved system")
            print("Saving backup")
            randomstate = numpy.random.get_state()
            numpy.save("randomstate-backup.npy", randomstate)
            if not self.gas_particles.is_empty():
                write_set_to_file(
                    self.gas_particles.savepoint(
                        self.gas_code.model_time + self.__begin_time),
                    "gas-backup.hdf5",
                    "amuse",
                    append_to_file=False,
                    version='2.0',
                    return_working_copy=False,
                )
            if not self.sink_particles.is_empty():
                write_set_to_file(
                    self.sink_particles.savepoint(
                        self.gas_code.model_time + self.__begin_time),
                    "sinks-backup.hdf5",
                    "amuse",
                    append_to_file=False,
                    version='2.0',
                    return_working_copy=False,
                )
            if not self.star_particles.is_empty():
                write_set_to_file(
                    self.star_particles.savepoint(
                        self.star_code.model_time + self.__begin_time),
                    "stars-backup.hdf5",
                    "amuse",
                    append_to_file=False,
                    version='2.0',
                    return_working_copy=False,
                )

            check_for_new_sinks = True
            while check_for_new_sinks:
                n_sink = len(self.sink_particles)
                self.resolve_sink_formation()
                if len(self.sink_particles) == n_sink:
                    self.logger.info("No new sinks")
                    check_for_new_sinks = False
                else:
                    self.logger.info("New sink formed")
                self.logger.info(
                    "Now we have %i stars; %i sinks and %i gas, %i particles"
                    " in total.",
                    len(self.star_particles),
                    len(self.sink_particles),
                    len(self.gas_particles),
                    (
                        len(self.gas_code.particles)
                        + len(self.star_code.particles)
                    ),
                )


            # stopping_iteration = 0
            # max_number_of_iterations = 2
            # high_density_remaining = True
            # while (
            #         high_density_remaining
            #         and stopping_iteration < max_number_of_iterations
            #         and density_limit_detection.is_set()
            # ):
            #     # Add this check since density detection isn't working
            #     # perfectly yet
            #     dens_frac = (
            #         self.gas_particles.density.max()
            #         / dmax
            #     )
            #     if dens_frac < 1:
            #         self.logger.info(
            #             "max density < stopping density but stopping condition"
            #             " still set?"
            #         )
            #         high_density_remaining = False
            #         # stopping_iteration = max_number_of_iterations
            #         break

            #     self.logger.debug("Forming new stars - %i", stopping_iteration)
            #     self.logger.info("Gas code stopped - max density reached")
            #     self.logger.debug(
            #         "Highest density / max density: %s",
            #         (
            #             self.gas_particles.density.max()
            #             / dmax
            #         )
            #     )
            #     self.sync_from_gas_code()
            #     self.resolve_sink_formation()
            #     self.logger.info(
            #         "Now we have %i stars; %i sinks and %i gas, %i particles"
            #         " in total.",
            #         len(self.star_particles),
            #         len(self.sink_particles),
            #         len(self.gas_particles),
            #         (
            #             len(self.gas_code.particles)
            #             + len(self.star_code.particles)
            #         ),
            #     )
            #     # Make sure we break the loop if the gas code is not going to
            #     # evolve further
            #     # if (
            #     #         self.gas_code.model_time
            #     #         < (time - self.gas_code.parameters.timestep)
            #     # ):
            #     #     break
            #     self.gas_code.evolve_model(
            #         relative_tend
            #         # self.gas_code.model_time
            #         # + self.gas_code.parameters.timestep
            #     )
            #     self.sync_from_gas_code()
            #     stopping_iteration += 1
            # if high_density_remaining is False:
            #     self.logger.info(
            #         "Remaining gas below critical density, disabling sink"
            #         " formation for now"
            #     )
            #     density_limit_detection.disable()
            #     self.gas_code.evolve_model(relative_tend)
            #     self.sync_from_gas_code()
            if not self.sink_particles.is_empty():
                for i, sink in enumerate(self.sink_particles):
                    self.logger.info(
                        "sink %i's radius = %s, mass = %s",
                        i,
                        sink.radius.in_(units.parsec),
                        sink.mass.in_(units.MSun),
                    )
                print("Forming stars")
                formed_stars = self.resolve_star_formation()
                if formed_stars and self.star_code is ph4:
                    self.star_code.zero_step_mode = True
                    self.star_code.evolve_model(relative_tend)
                    self.star_code.zero_step_mode = False
                print("Formed stars")
                if not self.star_particles.is_empty:
                    self.logger.info(
                        "Average mass of stars: %s. "
                        "Average mass of sinks: %s.",
                        self.star_particles.mass.mean().in_(units.MSun),
                        self.sink_particles.mass.mean().in_(units.MSun),
                    )

            self.logger.info(
                "Evo time is now %s",
                self.evo_code.model_time.in_(Myr)
            )
            self.logger.info(
                "Bridge time is now %s", (
                    (self.__begin_time + self.system.model_time).in_(Myr)
                )
            )
            # self.evo_code_to_model.copy()
            # Check for stopping conditions

            # Clean up accreted gas
            dead_gas = self.gas_particles.select_array(
                lambda x: x <= (0. | units.parsec),
                ["h_smooth"]
            )
            self.logger.info(
                "Number of dead/accreted gas particles: %i", len(dead_gas)
            )
            self.remove_gas(dead_gas)

            self.feedback_timestep = self.timestep
            if not self.star_particles.is_empty():
                self.stellar_feedback()

            self.logger.info(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s",
                real_tend.in_(units.Myr),
                (self.model_time).in_(Myr),
                (self.__begin_time + self.gas_code.model_time).in_(Myr),
                (self.__begin_time + self.star_code.model_time).in_(Myr),
                (self.evo_code.model_time).in_(Myr),
            )
            print(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s" % (
                    real_tend.in_(units.Myr),
                    (self.model_time).in_(Myr),
                    (self.__begin_time + self.gas_code.model_time).in_(Myr),
                    (self.__begin_time + self.star_code.model_time).in_(Myr),
                    (self.evo_code.model_time).in_(Myr),
                )
            )

        # self.gas_code_to_model.copy()
        # self.star_code_to_model.copy()

    def get_gravity_at_point(self, *args, **kwargs):
        force = VectorQuantity(
            [0, 0, 0],
            unit=units.m * units.s**-2,
        )
        for parent in [self.star_code, self.gas_code, self.tidal_field]:
            force += parent.get_gravity_at_point(*args, **kwargs)
        return force

    @property
    def model_time(self):
        "Return time of the system"
        return self.system.model_time + self.__begin_time


def main(
        seed=22, have_stars=False, have_gas=False, have_sinks=False,
):
    "Simulate an embedded star cluster (sph + dynamics + evolution)"
    from amuse.io import read_set_from_file
    from plotting_class import temperature_to_u
    from version import version

    logger = logging.getLogger(__name__)

    args = new_argument_parser()
    gasfilename = args.gasfilename
    starsfilename = args.starsfilename
    sinksfilename = args.sinksfilename
    randomfilename = args.randomfilename
    rundir = args.rundir
    # TODO: get time stamp from gas, stars, or sinks
    # Default for the initial spiral gas is 1.4874E+15 seconds

    if randomfilename is None:
        numpy.random.seed(seed)
    else:
        numpy.random.set_state(
            numpy.load(
                randomfilename,
                allow_pickle=True,
            )
        )
    run_prefix = rundir + "/"

    logging_level = logging.INFO
    logging.basicConfig(
        filename="%sembedded_star_cluster_info.log" % run_prefix,
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
    logger.info("git revision: %s", version())

    star_converter = nbody_system.nbody_to_si(
        150 | units.MSun,
        1 | units.parsec,
    )

    gas_converter = nbody_system.nbody_to_si(
        100000 | units.MSun,
        1 | units.parsec,
    )

    if starsfilename is not None:
        stars = read_set_from_file(starsfilename, "amuse", close_file=True,)
        have_stars = True
    if gasfilename is not None:
        print("reading gas")
        gas_ = read_set_from_file(gasfilename, "amuse", close_file=True,)
        begin_time = gas_.get_timestamp()
        if begin_time is None:
            begin_time = 0.0 | units.Myr
            try:
                del gas_.u
            except KeyError:
                pass
            try:
                del gas_.pressure
            except KeyError:
                pass
            u = temperature_to_u(100 | units.K)
            gas_.u = u

        # gas = gas_.select(
        #     lambda x, y:
        #     (x + (1810 | units.parsec))**2
        #     + (y + (1820 | units.parsec))**2
        #     < (50 | units.parsec)**2,
        #     ["x", "y"]
        # )
        gas = gas_
        print("Using %i particles" % len(gas))
        have_gas = True
    else:
        from amuse.ext.molecular_cloud import molecular_cloud
        mtot = 100000 | units.MSun
        rtot = 10 | units.parsec
        converter = nbody_system.nbody_to_si(mtot, rtot)
        gas = molecular_cloud(convert_nbody=converter, targetN=50000).result
        gas.u = temperature_to_u(10 | units.K)
        have_gas = True
        begin_time = 0.0 | units.Myr

    if sinksfilename is not None:
        sinks = read_set_from_file(sinksfilename, "amuse", close_file=True,)
        have_sinks = True

    if not (have_stars or have_gas or have_sinks):
        print("No particles!")
        exit()
    # if not have_stars:
    #     # This is a ph4 workaround. Don't like it but hard to run with zero stars.
    #     stars = new_star_cluster(
    #         number_of_stars=2
    #     )
    #     stars.mass *= 0.0001
    #     stars.position += gas.center_of_mass()
    #     stars.velocity += gas.center_of_mass_velocity()
    #     have_stars = True
    model = ClusterInPotential(
        stars=stars if have_stars else Particles(),
        gas=gas if have_gas else Particles(),
        sinks=sinks if have_sinks else Particles(),
        star_converter=star_converter,
        gas_converter=gas_converter,
        begin_time=begin_time,
    )

    timestep = 0.01 | units.Myr
    starting_step = int(begin_time / timestep)
    print("Forming sinks")
    # print(
    #     "Maximum density before forming sinks: %s"
    #     % model.gas_particles.density.max().in_(units.g * units.cm**-3)
    # )
    model.resolve_sink_formation()
    # print(
    #     "Maximum density after forming sinks: %s"
    #     % model.gas_particles.density.max().in_(units.g * units.cm**-3)
    # )
    # model.resolve_sink_formation()
    # print(
    #     "Maximum density after second round of forming sinks: %s"
    #     % model.gas_particles.density.max().in_(units.g * units.cm**-3)
    # )
    # exit()
    print("Formed sinks, now forming stars")
    print("Forming stars")
    model.resolve_star_formation()
    print("Formed stars")
    for step in range(starting_step, 500):
        time_unit = units.Myr
        print(
            "MT: %s HT: %s GT: %s ET: %s" % (
                model.model_time.in_(time_unit),
                model.gas_code.model_time.in_(time_unit),
                model.star_code.model_time.in_(time_unit),
                model.evo_code.model_time.in_(time_unit),
            )
        )
        time = (1+step) * timestep
        while model.model_time < time - timestep/2:
            model.evolve_model(time)
        print(
            "Evolved to %s" % model.model_time.in_(time_unit)
        )
        print(
            "Number of particles - gas: %i sinks: %i stars: %i" % (
                len(model.gas_particles),
                len(model.sink_particles),
                len(model.star_particles),
            )
        )
        try:
            print(
                "Most massive - sink: %s star: %s" % (
                    model.sink_particles.mass.max().in_(units.MSun),
                    model.star_particles.mass.max().in_(units.MSun),
                )
            )
            print(
                "Sinks centre of mass: %s" % (
                    model.sink_particles.center_of_mass().in_(units.parsec),
                )
            )
        except AttributeError:
            pass
        print(
            "Gas centre of mass: %s" % (
                model.gas_particles.center_of_mass().in_(units.parsec),
            )
        )
        dmax_now = model.gas_particles.density.max()
        dmax_stop = model.gas_code.parameters.stopping_condition_maximum_density
        print(
            "Maximum density / stopping density = %s" % (
                dmax_now
                / dmax_stop,
            )
        )
        logger.info("Max density: %s", dmax_now.in_(units.g * units.cm**-3))
        logger.info("Max density / sink formation density: %s", dmax_now/dmax_stop)

        logger.info("Making plot")
        print("Creating plot")
        plotname = "%sembedded-phantom-%04i.png" % (run_prefix, step)
        mtot = model.gas_particles.total_mass()
        com = mtot * model.gas_particles.center_of_mass()
        if not model.sink_particles.is_empty():
            mtot += model.sink_particles.total_mass()
            com += (
                model.sink_particles.total_mass()
                * model.sink_particles.center_of_mass()
            )
        if not model.star_particles.is_empty():
            mtot += model.star_particles.total_mass()
            com += (
                model.star_particles.total_mass()
                * model.star_particles.center_of_mass()
            )
        com = com / mtot
        print("Centre of mass: %s" % com)
        plot_hydro_and_stars(
            model.model_time,
            model.gas_code,
            stars=model.star_particles,
            sinks=model.sink_particles,
            L=500,  # 2*plot_radius.value_in(units.parsec),
            N=600,
            image_size_scale=2.,
            filename=plotname,
            title="time = %06.2f %s" % (
                model.model_time.value_in(units.Myr),
                units.Myr,
            ),
            offset_x=com[0].value_in(units.parsec),
            offset_y=com[1].value_in(units.parsec),
            gasproperties=["density", ],
            colorbar=True,
            starscale=0.2,
            # stars_are_sinks=True,
            # stars_are_sinks=False,
            # alpha_sfe=model.alpha_sfe,
        )

        logger.info("Writing snapshots")
        print("Writing snapshots")
        if step % 1 == 0:
            randomstate = numpy.random.get_state()
            randomstatefile = (
                "%srandomstate-%04i.npy" % (run_prefix, step)
            )
            numpy.save(randomstatefile, randomstate)
            if not model.gas_particles.is_empty():
                write_set_to_file(
                    model.gas_particles.savepoint(
                        model.gas_code.model_time + begin_time),
                    "%sgas-%04i.hdf5" % (run_prefix, step),
                    "amuse",
                    append_to_file=False,
                    version='2.0',
                    return_working_copy=False,
                )
            if not model.sink_particles.is_empty():
                write_set_to_file(
                    model.sink_particles.savepoint(
                        model.gas_code.model_time + begin_time),
                    "%ssinks-%04i.hdf5" % (run_prefix, step),
                    "amuse",
                    append_to_file=False,
                    version='2.0',
                    return_working_copy=False,
                )
            if not model.star_particles.is_empty():
                write_set_to_file(
                    model.star_particles.savepoint(
                        model.star_code.model_time + begin_time),
                    "%sstars-%04i.hdf5" % (run_prefix, step),
                    "amuse",
                    append_to_file=False,
                    version='2.0',
                    return_working_copy=False,
                )
    return


if __name__ == "__main__":
    main()
