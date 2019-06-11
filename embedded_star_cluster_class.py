"Class for a star cluster embedded in a tidal field and a gaseous region"
from __future__ import print_function, division
import logging
import numpy
from amuse.community.fastkick.interface import FastKick
from amuse.datamodel import ParticlesSuperset, Particles, Particle
from amuse.units import units, nbody_system
from amuse.units.quantities import VectorQuantity
from amuse.io import write_set_to_file

from bridge import (
    Bridge, CalculateFieldForCodes,
)
from gas_class import GasCode  # , sfe_to_density
from star_cluster_class import StarCluster
from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)
from plotting_class import plot_hydro_and_stars
from merge_recipes import form_new_star
from star_forming_region_class import form_stars  # StarFormingRegion
from amuse.units.trigo import sin, cos, arccos, arctan

Tide = TimeDependentSpiralArmsDiskModel


def accrete_gas(sink, gas):
    accreted_gas = Particles()
    distance_to_sink_squared = (
        (gas.x - sink.x)**2
        + (gas.y - sink.y)**2
        + (gas.z - sink.z)**2
    )
    accreted_gas.add_particles(gas[
        numpy.where(distance_to_sink_squared < sink.radius**2)
    ])

    return accreted_gas


class ClusterInPotential(
        StarCluster,
        # Gas,
):
    """Stellar cluster in an external potential"""

    def __init__(
            self,
            stars=None,
            gas=None,
            # epsilon=200 | units.AU,
            epsilon=0.1 | units.parsec,
            star_converter=None,
            gas_converter=None,
            logger=None,
            logger_level=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        if logger_level is None:
            self.logger.setLevel(logging.DEBUG)
        # else:
        #     self.logger.setLevel(logger_level)

        # self.objects = (StarCluster, Gas)
        mass_scale_stars = stars.mass.sum()
        length_scale_stars = 3 | units.parsec
        if star_converter is None:
            converter_for_stars = nbody_system.nbody_to_si(
                mass_scale_stars,
                length_scale_stars,
            )
        else:
            converter_for_stars = star_converter
        self.logger.info("Initialising StarCluster")
        StarCluster.__init__(
            self, stars=stars, converter=converter_for_stars, epsilon=epsilon,
        )
        self.logger.info("Initialised StarCluster")

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
        self.gas_code = GasCode(converter=new_gas_converter)
        self.gas_code.gas_particles.add_particles(gas)
        print(self.gas_code.parameters)
        # Gas.__init__(
        #     self, gas=gas, converter=converter_for_gas, epsilon=epsilon,
        #     internal_cooling=False,
        # )
        # self.gas_code.parameters.timestep = 0.005 | units.Myr
        self.timestep = 0.01 | units.Myr
        self.logger.info("Initialised Gas")
        self.logger.info("Creating Tide object")
        self.tidal_field = Tide()
        self.logger.info("Created Tide object")

        self.epsilon = epsilon
        self.converter = converter_for_gas

        def new_field_gravity_code():
            "Create a new field code"
            print("Creating field code")
            result = FastKick(
                self.converter,
                redirection="none",
                # redirect_file=(
                #     p.dir_codelogs + "/field_gravity_code.log"
                #     ),
                mode="cpu",
                number_of_workers=8,
            )
            result.parameters.epsilon_squared = (self.epsilon)**2
            print(result.parameters)
            return result

        def new_field_code(
                code,
        ):
            " something"
            # result = CalculateFieldForCodesUsingReinitialize(
            result = CalculateFieldForCodes(
                new_field_gravity_code,
                [code],
                # required_attributes=[
                #     'mass', 'radius',
                #     'x', 'y', 'z',
                #     'vx', 'vy', 'vz',
                # ]
                # required_attributes=[
                #     'mass', 'u',
                #     'x', 'y', 'z',
                #     'vx', 'vy', 'vz',
                # ]
            )
            return result

        to_gas_codes = [
            new_field_code(
                self.star_code,
            ),
            self.tidal_field,
        ]
        to_stars_codes = [
            new_field_code(
                self.gas_code,
            ),
            self.tidal_field,
        ]

        self.system = Bridge(
            timestep=(
                self.timestep
            ),
            # use_threading=False,
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
            # do_sync=False,
        )
        # self.system.timestep = 2 * self.gas_code.parameters.timestep
        # 0.05 | units.Myr

    @property
    def gas_particles(self):
        "Return gas particles"
        return self.gas_code.gas_particles

    @property
    def dm_particles(self):
        "Return dark matter particles"
        return self.gas_code.dm_particles

    @property
    def sink_particles(self):
        "Return sink particles"
        return self.gas_code.sink_particles

    @property
    def star_particles(self):
        "Return star particles"
        return self.star_code.particles

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
        removed_gas = Particles()
        maximum_density = (
            self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select_array(
            lambda rho:
            rho > maximum_density,
            ["rho"],
        )
        for i, origin_gas in enumerate(high_density_gas):
            if origin_gas not in removed_gas:
                # try:
                new_sink = Particle()
                new_sink.initialised = False

                # Setting the radius to something that will lead to >~150MSun
                # per sink would be ideal.
                # So this radius is related to the density.
                minimum_sink_radius = 1.0 | units.parsec
                desired_sink_mass = 200 | units.MSun
                desired_sink_radius = min(
                    (desired_sink_mass / origin_gas.density)**(1/3),
                    minimum_sink_radius,
                )

                new_sink.radius = desired_sink_radius
                # new_sink.radius = 1 | units.parsec # 100 | units.AU  # or something related to mass?

                new_sink.accreted_mass = 0 | units.MSun
                o_x, o_y, o_z = origin_gas.position

                new_sink.position = origin_gas.position
                accreted_gas = accrete_gas(new_sink, self.gas_particles)
                if accreted_gas.is_empty():
                    self.logger.info("Empty gas so no sink %i", i)
                else:
                    self.logger.info(
                        "Number of accreted gas particles: %i",
                        len(accreted_gas)
                    )
                    new_sink.position = accreted_gas.center_of_mass()
                    new_sink.velocity = accreted_gas.center_of_mass_velocity()
                    new_sink.mass = accreted_gas.total_mass()
                    new_sink.accreted_mass = (
                        accreted_gas.total_mass() - origin_gas.mass
                    )
                    removed_gas.add_particles(accreted_gas)
                    self.remove_gas(accreted_gas)
                    self.add_sink(new_sink)
                    self.logger.info(
                        "Added sink %i with mass %s", i,
                        new_sink.mass.in_(units.MSun)
                    )
                    # except:
                    #     print("Could not add another sink")

    def add_sink(self, sink):
        # self.gas_code.gas_code.sink_particles.add_particle(sink)
        self.sink_particles.add_particle(sink)
        # sfr = StarFormingRegion(
        #     key=sink.key,
        #     position=sink.position,
        #     velocity=sink.velocity,
        #     mass=sink.mass,
        #     radius=sink.radius,
        #     formation_time=self.model_time,
        # )
        # self.star_forming_particles.append(sfr)

    def remove_sinks(self, sinks):
        # self.gas_code.gas_code.sink_particles.remove_particles(sinks)
        self.sink_particles.remove_particles(sinks)

    def add_gas(self, gas):
        # self.gas_code.gas_code.gas_particles.add_particles(gas)
        self.gas_particles.add_particles(gas)

    def remove_gas(self, gas):
        # self.gas_code.gas_code.gas_particles.remove_particles(gas)
        self.gas_particles.remove_particles(gas)

    def add_stars(self, stars):
        self.star_particles.add_particles(stars)
        self.evo_code.particles.add_particles(stars)

    def resolve_star_formation(self):
        self.logger.info("Resolving star formation")
        mass_before = (
            self.sink_particles.total_mass() + self.star_particles.total_mass()
        )
        max_new_stars_per_timestep = 100
        stellar_mass_formed = 0 | units.MSun
        for i, sink in enumerate(self.sink_particles):
            self.logger.debug("Processing sink %i", i)
            # if i%100 == 99:
            #     print(i)
            new_stars = form_stars(sink)
            if new_stars is not None:
                j = 0
                while j <= max_new_stars_per_timestep:
                    if new_stars is not None:
                        self.add_stars(new_stars)
                        stellar_mass_formed += new_stars.total_mass()
                        new_stars = form_stars(sink)
                        j += 1
                    else:
                        break

        mass_after = (
            self.sink_particles.total_mass() + self.star_particles.total_mass()
        )
        self.logger.debug(
            "dM = %s, mFormed = %s ",
            (mass_after - mass_before).in_(units.MSun),
            stellar_mass_formed.in_(units.MSun),
        )

    def _resolve_starformation(self):
        self.logger.info("Resolving star formation")

        maximum_density = (
            self.gas_code.parameters.stopping_condition_maximum_density
        )
        high_density_gas = self.gas_particles.select(
            lambda rho:
            rho >= maximum_density,
            # sfe_to_density(
            #     1, alpha=self.alpha_sfe,
            # ),
            ["rho"],
        )  # .sorted_by_attribute("rho")

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

            new_star.birth_age = self.gas_code.model_time
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

    def _stellar_feedback(self):
        # Extremely simplified stellar feedback onto gas
        gas = self.gas_particles.copy()
        mass_cutoff = 5 | units.MSun
        rmax = 1000 | units.AU
        massive_stars = self.star_particles.select(
            lambda m:
            m >= mass_cutoff,
            ["mass"]
        )
        for i, star in enumerate(massive_stars):
            gas.position -= star.position
            gas.dist = gas.position.lengths()
            gas = gas.sorted_by_attribute("dist")
            gas_near_star = gas.select(
                lambda r:
                r < rmax,
                ["dist"]
            )
            theta = arccos(gas_near_star.z/gas_near_star.dist)
            phi = arctan(gas_near_star.y / gas_near_star.x)
            # Something related to energy
            # (Slow) solar wind is about 400 km/s
            base_velocity = (400 | units.kms)  # (1 | units.AU) / self.feedback_timestep
            gas_near_star.vx += sin(theta) * cos(phi) * base_velocity
            gas_near_star.vy += sin(theta) * sin(phi) * base_velocity
            gas_near_star.vz += cos(theta) * base_velocity

            gas.position += star.position

    def evolve_model(self, tend):
        "Evolve system to specified time"
        self.logger.info(
            "Evolving to time %s",
            tend,
        )
        # self.model_to_evo_code.copy()
        # self.model_to_gas_code.copy()

        density_limit_detection = \
            self.gas_code.stopping_conditions.density_limit_detection
        density_limit_detection.enable()
        # density_limit_detection.disable()

        # TODO: re-enable at some point?
        # collision_detection = \
        #     self.star_code.stopping_conditions.collision_detection
        # collision_detection.enable()

        # maximum_density = (
        #     self.gas_code.parameters.stopping_condition_maximum_density
        # )

        print("Starting loop")
        print(self.model_time, tend, self.system.timestep)
        step = 0
        minimum_steps = 1
        print(
            (
                (self.model_time < (tend - self.system.timestep))
                or (step < minimum_steps)
            )
        )
        while (
                (self.model_time < (tend - self.system.timestep))
                or (step < minimum_steps)
        ):
            step += 1
            # evo_time = self.evo_code.model_time
            # self.model_to_star_code.copy()
            evo_timestep = self.evo_code.particles.time_step.min()
            self.logger.info(
                "Smallest evo timestep: %s", evo_timestep.in_(units.Myr)
            )
            time = min(
                # evo_time+evo_timestep,
                tend,
                10 * tend,
            )
            self.logger.info("Evolving to %s", time.in_(units.Myr))
            self.logger.info("Stellar evolution...")
            self.evo_code.evolve_model(time)
            # dt_cooling = time - self.gas_code.model_time
            # if self.cooling:
            #     self.logger.info("Cooling gas...")
            #     self.cooling.evolve_for(dt_cooling/2)
            self.logger.info("System...")

            print("Evolving system")
            self.system.evolve_model(time)
            if self.gas_code.model_time < (time - self.system.timestep):
                self.gas_code.evolve_model(time)
            print("Evolved system")

            # stopping_iteration = 0
            # while (
            #         collision_detection.is_set()
            #         and stopping_iteration < (len(self.star_particles) / 2)
            # ):
            #     print("Merging colliding stars - %i" % stopping_iteration)
            #     self.resolve_collision(collision_detection)
            #     self.star_code.evolve_model(time)
            #     stopping_iteration += 1
            # if stopping_iteration >= (len(self.star_particles) / 2):
            #     print(
            #         "Stopping too often - disabling collision detection"
            #     )
            #     collision_detection.disable()
            #     self.star_code.evolve_model(time)

            stopping_iteration = 0
            max_number_of_iterations = 10
            while (
                    stopping_iteration < max_number_of_iterations
                    and density_limit_detection.is_set()
            ):
                # Add this check since density detection isn't working perfectly yet
                dens_frac = (
                    self.gas_particles.density.max()
                    / self.gas_code.parameters.stopping_condition_maximum_density
                )
                if dens_frac < 1:
                    self.logger.info("max density < stopping density but stopping condition still set?")
                    stopping_iteration = max_number_of_iterations
                    break

                self.logger.debug("Forming new stars - %i", stopping_iteration)
                # print(
                #     self.gas_particles.density.max()
                #     / maximum_density
                # )
                self.logger.info("Gas code stopped - max density reached")
                self.logger.debug(
                    "Highest density / max density: %s",
                    (
                        self.gas_particles.density.max()
                        / self.gas_code.parameters.stopping_condition_maximum_density
                    )
                )
                # self.gas_code_to_model.copy()
                self.resolve_sink_formation()
                self.logger.info(
                    "Now we have %i stars; %i sinks and %i gas, %i particles in total.",
                    len(self.star_particles),
                    len(self.sink_particles),
                    len(self.gas_particles),
                    (len(self.gas_code.particles) + len(self.star_code.particles)),
                )
                # Make sure we break the loop if the gas code is not going to
                # evolve further
                # if (
                #         self.gas_code.model_time
                #         < (time - self.gas_code.parameters.timestep)
                # ):
                #     break
                self.gas_code.evolve_model(
                    time
                    # self.gas_code.model_time
                    # + self.gas_code.parameters.timestep
                )
                stopping_iteration += 1
            if stopping_iteration >= max_number_of_iterations:
                self.logger.info("Stopping too often - disabling sink formation for now")
                density_limit_detection.disable()
                self.gas_code.evolve_model(time)
            if not self.sink_particles.is_empty():
                print("Forming stars")
                self.resolve_star_formation()
                print("Formed stars")
                self.logger.info(
                    "Average mass of stars: %s. Average mass of sinks: %s.",
                    self.star_particles.mass.mean().in_(units.MSun),
                    self.sink_particles.mass.mean().in_(units.MSun),
                )
                for i, sink in enumerate(self.sink_particles):
                    self.logger.info(
                        "sink %i's radius = %s, mass = %s",
                        i,
                        sink.radius.in_(units.parsec),
                        sink.mass.in_(units.MSun),
                    )

            # self.system.evolve_model(time)

            # if self.cooling:
            #     self.logger.info("Second cooling gas...")
            #     self.cooling.evolve_for(dt_cooling/2)

            self.logger.info(
                "Evo time is now %s", self.evo_code.model_time.in_(units.Myr)
            )
            self.logger.info(
                "Bridge time is now %s", (
                    self.system.model_time.in_(units.Myr)
                )
            )
            # self.evo_code_to_model.copy()
            # Check for stopping conditions

            # Clean up accreted gas
            dead_gas = self.gas_code.gas_particles.select(
                lambda x: x <= 0.,
                ["h_smooth"]
            )
            self.logger.info("Number of dead/accreted gas particles: %i", len(dead_gas))
            self.remove_gas(dead_gas)

            self.feedback_timestep = self.timestep
            if not self.star_particles.is_empty():
                self.stellar_feedback()

            self.logger.info(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s",
                tend.in_(units.Myr),
                self.model_time.in_(units.Myr),
                self.gas_code.model_time.in_(units.Myr),
                self.star_code.model_time.in_(units.Myr),
                self.evo_code.model_time.in_(units.Myr),
            )
            print(
                "Time: end= %s bridge= %s gas= %s stars=%s evo=%s" % (
                    tend.in_(units.Myr),
                    self.model_time.in_(units.Myr),
                    self.gas_code.model_time.in_(units.Myr),
                    self.star_code.model_time.in_(units.Myr),
                    self.evo_code.model_time.in_(units.Myr),
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

    # @property
    # def particles(self):
    #     return self.code.particles

    @property
    def model_time(self):
        "Return time of the system"
        return self.system.model_time


def main():
    "Simulate an embedded star cluster (sph + dynamics + evolution)"
    import sys
    import numpy
    from amuse.io import read_set_from_file
    from amuse.ic.plummer import new_plummer_model
    from amuse.ic.salpeter import new_salpeter_mass_distribution
    from amuse.ext.molecular_cloud import molecular_cloud
    from plotting_class import temperature_to_u

    # numpy.random.seed(14)
    numpy.random.seed(15)
    # run_prefix = "Test_small_"
    run_prefix = ""

    logging_level = logging.INFO
    logging.basicConfig(
        filename="%sembedded_star_cluster_info.log" % run_prefix,
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

    number_of_stars = 10
    stellar_masses = new_salpeter_mass_distribution(number_of_stars)
    star_converter = nbody_system.nbody_to_si(
        stellar_masses.sum() + (50000 | units.MSun),
        0.5 | units.parsec,
    )
    # stars = read_set_from_file(sys.argv[1], "amuse")
    stars = new_plummer_model(
        number_of_stars,
        convert_nbody=star_converter,
    )[:2]
    stars.mass = 0.1 | units.MSun  # stellar_masses

    gas_converter = nbody_system.nbody_to_si(
        10000000 | units.MSun,
        10 | units.parsec,
    )
    # gas = read_set_from_file(sys.argv[2], "amuse")
    # gas = molecular_cloud(targetN=50000, convert_nbody=gas_converter).result

    # print(sys.argv)
    if len(sys.argv) > 1:
        print("reading gas")
        gas_ = read_set_from_file(sys.argv[1], "amuse")
        del(gas_.u)
        print("%i particles read" % len(gas_))
        if run_prefix == "Test_small_":
            gas = gas_[:1000000]
        else:
            gas = gas_
    else:
        gas = molecular_cloud(targetN=100000, convert_nbody=gas_converter).result

        gastwo = molecular_cloud(
            targetN=100000, convert_nbody=gas_converter
        ).result

        gas.x -= 5 | units.parsec
        gastwo.x += 5 | units.parsec
        gastwo.y += 0 | units.parsec
        gastwo.z -= 0 | units.parsec
        gas.vx += 2.5 | units.kms
        gastwo.vx -= 2.5 | units.kms
        # gas.vx += ((3 | units.parsec) / (2 | units.Myr))
        # gastwo.vx -= ((3 | units.parsec) / (2 | units.Myr))

        gas.add_particles(gastwo)
    stars.position += gas.center_of_mass()
    stars.velocity += gas.center_of_mass_velocity()
    # if len(sys.argv) > 2:
    #     print("reading sinks")

    u = temperature_to_u(10 | units.K)
    gas.u = u

    model = ClusterInPotential(
        stars=stars,
        gas=gas,
        star_converter=star_converter,
        gas_converter=gas_converter,
    )

    timestep = 0.01 | units.Myr
    for step in range(500):
        time = (1+step) * timestep
        while model.model_time < time - timestep/2:
            model.evolve_model(time)
        print(
            "Evolved to %s" % model.model_time.in_(units.Myr)
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
        print(
            "Maximum density / stopping density = %s" % (
                model.gas_particles.density.max()
                / model.gas_code.parameters.stopping_condition_maximum_density,
            )
        )

        plotname = "%sembedded-phantom-%04i.png" % (run_prefix, step)
        print("Creating plot")
        mtot = model.gas_particles.total_mass()
        com = mtot * model.gas_particles.center_of_mass()
        if not model.sink_particles.is_empty():
            mtot += model.sink_particles.total_mass()
            com += model.sink_particles.total_mass() * model.sink_particles.center_of_mass()
        if not model.star_particles.is_empty():
            mtot += model.star_particles.total_mass()
            com + model.star_particles.total_mass() * model.star_particles.center_of_mass()
        com = com / mtot
        print("Centre of mass: %s" % com)
        plot_hydro_and_stars(
            model.model_time,
            model.gas_code,
            stars=model.star_particles,
            sinks=model.sink_particles,
            L=500,
            N=500,
            filename=plotname,
            title="time = %06.2f %s" % (
                model.gas_code.model_time.value_in(units.Myr),
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
        # for i, sink in enumerate(model.sink_particles):
        #     plotname = "%sembedded-phantom-sink%04i-%04i.png" % (run_prefix, i, step)

        #     # TODO: center on centre_of_mass for all the stars + sink
        #     # so create a particle group to keep track of this
        #     center_x = sink.x
        #     center_y = sink.y
        #     plot_hydro_and_stars(
        #         model.model_time,
        #         model.gas_code,
        #         stars=model.star_particles,
        #         sinks=None,
        #         L=5,
        #         N=10,
        #         filename=plotname,
        #         title="time = %06.2f %s - sink %i" % (
        #             model.gas_code.model_time.value_in(units.Myr),
        #             units.Myr,
        #             i,
        #         ),
        #         offset_x=center_x.value_in(units.parsec),
        #         offset_y=center_y.value_in(units.parsec),
        #         gasproperties=["density", ],
        #         colorbar=True,
        #     )
        if step % 10 == 0:
            if not model.gas_particles.is_empty():
                write_set_to_file(
                    model.gas_particles,
                    "%sgas-%04i.hdf5" % (run_prefix, step),
                    "amuse",
                )
            if not model.sink_particles.is_empty():
                write_set_to_file(
                    model.sink_particles,
                    "%ssinks-%04i.hdf5" % (run_prefix, step),
                    "amuse",
                )
            if not model.star_particles.is_empty():
                write_set_to_file(
                    model.star_particles,
                    "%sstars-%04i.hdf5" % (run_prefix, step),
                    "amuse",
                )

    return


if __name__ == "__main__":
    main()
