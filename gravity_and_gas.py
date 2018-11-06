#!/usr/bin/python
# enc: utf-8
"""
Simulate a molecular gas cloud
"""
from __future__ import division, print_function

import numpy

import matplotlib.pyplot as pyplot

from amuse.units import (nbody_system, units, constants)
from amuse.datamodel import (Particles, ParticlesSuperset)
from amuse.support.console import set_printing_strategy

from amuse.io import write_set_to_file, read_set_from_file
from amuse.io.base import IoException

from amuse.ic.salpeter import new_salpeter_mass_distribution
# from amuse.ic.gasplummer import new_plummer_gas_model
# from amuse.ic.plummer import new_plummer_model
from amuse.ext.molecular_cloud import molecular_cloud
# from amuse.ext.sink import new_sink_particles
from amuse.ext.evrard_test import body_centered_grid_unit_cube
from amuse.ext.LagrangianRadii import LagrangianRadii

from amuse.community.fastkick.interface import FastKick

# import matplotlib.pyplot as plt

from cooling_class import SimplifiedThermalModelEvolver
from prepare_figure import single_frame
# from diagnostics import identify_subgroups

# Initialise environment
# include:
# - Gravity for stars
# - SPH code for gas
# - Stellar evolution
# - Feedback
# - Binary stars


def make_map(sph, number_of_cells=100, length_scaling=1):
    """
    create a density map
    """
    x, y = numpy.indices((number_of_cells+1, number_of_cells+1))
    x = length_scaling*(x.flatten()-number_of_cells/2.)/number_of_cells
    y = length_scaling*(y.flatten()-number_of_cells/2.)/number_of_cells
    z = x*0.
    vx = 0.*x
    vy = 0.*x
    vz = 0.*x

    x = units.parsec(x)
    y = units.parsec(y)
    z = units.parsec(z)
    vx = units.kms(vx)
    vy = units.kms(vy)
    vz = units.kms(vz)

    rho, rhovx, rhovy, rhovz, rhoe = sph.get_hydro_state_at_point(
        x, y, z, vx, vy, vz,
    )
    del(rhovx, rhovy, rhovz, rhoe)
    rho = rho.reshape((number_of_cells+1, number_of_cells+1))
    return rho


class DefaultUnits(object):
    """
    Set units to be used
    """
    def __init__(self):
        self.mass = units.MSun
        self.length = units.parsec
        self.time = units.Myr
        self.speed = units.kms

    def use_physical(self):
        """Use physical units"""
        self.mass = units.MSun
        self.length = units.parsec
        self.time = units.Myr
        self.speed = units.kms

    def use_nbody(self):
        """Use n-body (H'enon) units"""
        self.mass = nbody_system.mass
        self.length = nbody_system.length
        self.time = nbody_system.time
        self.speed = nbody_system.speed


def plot_hydro(
        time, i, sph, length_scaling=10,):
    """
    make a plot of the current gas densities in a hydro code
    """
    # pylint: disable=too-many-locals
    unitsystem = DefaultUnits()
    x_label = "x [%s]" % unitsystem.length
    y_label = "y [%s]" % unitsystem.length
    fig = single_frame(
        x_label, y_label,
        logx=False, logy=False,
        xsize=15, ysize=15)

    # gas = sph.gas_particles
    dm_particles = sph.dm_particles
    rho = make_map(sph, number_of_cells=400, length_scaling=length_scaling)
    cax = pyplot.imshow(
        numpy.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
        extent=[
            -length_scaling/2, length_scaling/2,
            -length_scaling/2, length_scaling/2],
        vmin=1, vmax=5, origin="lower")

    cbar = fig.colorbar(cax, orientation='vertical', fraction=0.045)
    cbar.set_label('projected density [$amu/cm^3$]', rotation=270)

    title_label = "time: %08.2f %s" % (
        time.value_in(unitsystem.time), unitsystem.time
    )

    fig.suptitle(title_label)

    color_map = pyplot.cm.get_cmap('RdBu')
    # cm = pyplot.cm.jet #gist_ncar
    if not dm_particles.is_empty():
        # m = 10.0*dm_particles.mass/dm_particles.mass.max()
        mass_scaling = 30*numpy.log10(
            dm_particles.mass/dm_particles.mass.min()
        )
        color_scaling = numpy.sqrt(dm_particles.mass/dm_particles.mass.max())
        pyplot.scatter(
            dm_particles.y.value_in(units.parsec),
            dm_particles.x.value_in(units.parsec),
            c=color_scaling, s=mass_scaling, lw=0, cmap=color_map)

    pyplot.savefig('gas-fig-%04i.png' % i)
    pyplot.close(fig)


def plot_stars(
        time, i, gravity, length_scaling=10,):
    """
    make a plot of the current gas densities in a hydro code
    """
    # pylint: disable=too-many-locals
    unitsystem = DefaultUnits()
    x_label = "x [%s]" % unitsystem.length
    y_label = "y [%s]" % unitsystem.length
    fig = single_frame(
        x_label, y_label,
        logx=False, logy=False,
        xsize=15, ysize=15)

    axes = fig.add_subplot(111)
    xmin = (-length_scaling/2)
    xmax = (length_scaling/2)
    ymin = (-length_scaling/2)
    ymax = (length_scaling/2)
    axes.set_xlim((xmin, xmax))
    axes.set_ylim((ymin, ymax))

    stars = gravity.particles

    title_label = "time: %08.2f %s" % (
        time.value_in(unitsystem.time), unitsystem.time
    )

    fig.suptitle(title_label)

    color_map = pyplot.cm.get_cmap('RdBu')

    if not stars.is_empty():
        # m = 10.0*dm_particles.mass/dm_particles.mass.max()
        mass_scaling = 30*numpy.log10(
            stars.mass/stars.mass.min()
        )
        color_scaling = numpy.sqrt(stars.mass/stars.mass.max())
        pyplot.scatter(
            stars.y.value_in(units.parsec),
            stars.x.value_in(units.parsec),
            c=color_scaling, s=mass_scaling, lw=0, cmap=color_map)

    pyplot.savefig('stars-fig-%04i.png' % i)
    pyplot.close(fig)


def merge_two_stars(bodies, particles_in_encounter):
    """
    Merge two stars into one
    """
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    star_0 = particles_in_encounter[0]
    star_1 = particles_in_encounter[1]

    new_particle = Particles(1)
    new_particle.birth_age = particles_in_encounter.birth_age.min()
    new_particle.mass = particles_in_encounter.total_mass()
    new_particle.age = min(particles_in_encounter.age) \
        * max(particles_in_encounter.mass)/new_particle.mass
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.name = "Star"
    new_particle.radius = particles_in_encounter.radius.max()
    print("# old radius:", particles_in_encounter.radius.in_(units.AU))
    print("# new radius:", new_particle.radius.in_(units.AU))
    bodies.add_particles(new_particle)
    print(
        "# Two stars (M=",
        particles_in_encounter.mass.in_(units.MSun),
        ") collided at d=",
        (star_0.position - star_1.position).length().in_(units.AU)
    )
    bodies.remove_particles(particles_in_encounter)


def new_evolution_code(code_name="SeBa"):
    """Initialises stellar evolution code"""
    if code_name == "SeBa":
        from amuse.community.seba.interface import SeBa
        code = SeBa()
    elif code_name == "SSE":
        from amuse.community.sse.interface import SSE
        code = SSE()
    else:
        raise "No such stellar evolution code"
    return code


class Cluster(object):
    """
    Create a simulator for an embedded star cluster
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    # pylint: disable=no-member
    # pylint: disable=missing-docstring
    # Important methods are described, others are self-explanatory

    def __init__(
            self,
            particles,
            convert_nbody=None,
            mode="gas",
            time_step_diag=0.1 | units.Myr,
    ):
        """
        Simulate a star cluster, with either gas, stars, or both.
        'particles' must be a shaped as (gas_particles, star_particles). These
        are passed to different codes, which are combined through Bridge.
        Mode is either "gas", "stars", or "gas+stars", and determines which
        codes are instantiated.
        """
        self.mode = mode
        self.converter = convert_nbody

        self.time_step_diag = time_step_diag

        self.gas_particles = particles[0]
        self.star_particles = particles[1]
        self.particles = ParticlesSuperset(
            [self.gas_particles, self.star_particles],
        )

        self.gas_sync_attributes = (
            "mass", "x", "y", "z", "vx", "vy", "vz", "u", "rho",
        )
        self.stars_sync_attributes = (
            "mass", "x", "y", "z", "vx", "vy", "vz", "radius",
        )
        self.evolution_sync_attributes = (
            "mass", "radius", "age", "temperature", "luminosity"
        )

        # Should set this to something automatically, not fixed
        self.box_size = 500 | units.parsec

        self.cooling_enabled = False
        self.cooling_flag = "thermal_model"

        if "gas" in self.mode:
            self.gas_code = self.new_gas_code()
            self.gas_code.gas_particles.add_particles(self.gas_particles)
            self.density_limit_detection = \
                self.gas_code.stopping_conditions.density_limit_detection
            # We're not yet doing star formation on the fly
            self.density_limit_detection.disable()

            self.gas_code_to_model = \
                self.gas_code.gas_particles.new_channel_to(
                    self.gas_particles,
                )
            self.gas_model_to_code = self.gas_particles.new_channel_to(
                self.gas_code.gas_particles,
            )
            self.sync_gas_to_model()

        if "stars" in self.mode:
            self.star_code = self.new_gravity_code()
            self.star_code.particles.add_particles(self.star_particles)
            self.collision_detection = \
                self.star_code.stopping_conditions.collision_detection
            # We are handling stellar collisions
            self.collision_detection.enable()

            self.evolution_code = new_evolution_code()
            self.evolution_code.particles.add_particles(self.star_particles)
            self.evolution_code.parameters.metallicity = 0.02
            self.evolution_code.evolve_model(0.0 | units.Myr)
            self.supernova_detection = \
                self.evolution_code.stopping_conditions.supernova_detection
            # Don't handle supernovae (yet)
            self.supernova_detection.disable()

            self.stars_code_to_model = \
                self.star_code.particles.new_channel_to(
                    self.star_particles,
                )
            self.stars_model_to_code = \
                self.star_particles.new_channel_to(
                    self.star_code.particles,
                )
            self.evolution_to_model = \
                self.evolution_code.particles.new_channel_to(
                    self.star_particles)

        if self.mode == "gas+stars":
            self.code = self.new_bridge()
        elif self.mode == "gas":
            self.code = self.gas_code
        elif self.mode == "stars":
            self.code = self.star_code

        # Calculate initial energy of the system
        self.kinetic_energy = self.code.kinetic_energy
        self.potential_energy = self.code.potential_energy
        if "gas" in mode:
            self.thermal_energy = self.code.thermal_energy
        else:
            self.thermal_energy = 0 * self.kinetic_energy
        self.tzero_thermal_energy = self.thermal_energy
        self.total_energy = (
            self.kinetic_energy
            + self.potential_energy
            + self.thermal_energy
        )
        self.tzero_total_energy = self.total_energy
        self.tlast_total_energy = self.tzero_total_energy
        self.energy_error_cumulative = 0.0
        self.tzero_mass = self.code.particles.mass.sum()

        print(
            "Virial ratio |Ek|/|Ep| = %s" % (
                abs(self.code.kinetic_energy)
                / abs(self.code.potential_energy)
            )
        )

        self.i = 0
        self.fig_num = 0

    def save_gas(self, filename="gas.hdf5"):
        write_set_to_file(
            self.gas_particles.savepoint(
                self.model_time),
            filename,
            "hdf5",
        )

    def save_stars(self, filename="stars.hdf5"):
        write_set_to_file(
            self.star_particles.savepoint(
                self.model_time),
            filename,
            "hdf5",
        )

    def sync_to_model(self):
        if "gas" in self.mode:
            self.sync_gas_to_model()
        if "stars" in self.mode:
            self.sync_stars_to_model()
        self.sync_energy_to_model()
        if self.i == 0:
            self.tzero_thermal_energy = self.thermal_energy
            self.tzero_total_energy = (
                self.kinetic_energy
                + self.potential_energy
                + self.thermal_energy
            )
            self.tlast_total_energy = self.tzero_total_energy
            self.energy_error_cumulative = 0.0
            self.tzero_mass = (
                self.code.particles.mass.sum()
            )

    # def print_diagnostics(self):
    #     hydrogen_atomic_mass = 1./6.02214179e+23 | units.g
    #     weight = 2.33
    #     temperature = (
    #         self.thermal_energy
    #         * 2.
    #         / (
    #             3.*self.gas_particles.mass.sum()
    #             / (weight*hydrogen_atomic_mass)*constants.kB
    #         )
    #     )
    #     print("temperature: %s" % (temperature))

    def new_gas_code(self, codename="Fi"):
        """Initialises the SPH code we will use"""
        # total_gas_mass = self.gas_particles.mass.sum()
        # number_of_gas_particles = len(self.gas_particles)
        # time_step = (
        #     0.004*numpy.pi*numpy.power(self.epsilon, 1.5)
        #     / numpy.sqrt(constants.G*total_gas_mass/number_of_gas_particles)
        # )
        # time_step = self.converter.to_si(0.01 | nbody_system.time)
        time_step = 5.e-2 | units.Myr
        print(
            "# Gas code time step: %s = %s Nbody time" % (
                time_step.in_(units.Myr),
                self.converter.to_nbody(time_step)
            )
        )
        if codename == "Fi":
            from amuse.community.fi.interface import Fi
            code = Fi(self.converter, mode="openmp")
            code.parameters.use_hydro_flag = True
            code.parameters.radiation_flag = False
            # code.parameters.begin_time = 0.0 | units.Myr
            # code.parameters.self_gravity_flag = True
            """
            isothermal_flag:
            When True then we have to do our own adiabatic cooling
            (and Gamma has to be 1.0)
            When False then we dont do the adiabatic cooling and Fi
            is changing u
            """
            code.parameters.gamma = 1
            code.parameters.isothermal_flag = True
            code.parameters.integrate_entropy_flag = False
            code.parameters.timestep = time_step
            code.parameters.verbosity = 0
            code.parameters.eps_is_h_flag = False
            code.parameters.gas_epsilon = 0.1 | units.parsec
            code.parameters.sph_h_const = 0.1 | units.parsec
            code.parameters.periodic_box_size = self.box_size
            # code.parameters.verbosity = 99
            code.parameters.integrate_entropy_flag = False
            # code.parameters.stopping_condition_maximum_density = \
            #     self.density_threshold

            self.gamma = code.parameters.gamma

            self.cooling = SimplifiedThermalModelEvolver(
                code.gas_particles)
            self.cooling.model_time = code.model_time
            return code
        # elif codename == "Gadget2":
        #     from amuse.community.gadget2.interface import Gadget2
        #     code = Gadget2(self.converter)
        else:
            raise "No such code implemented"

    def new_gravity_code(self, code_name="ph4"):
        """Initialises the gravity code we will use"""
        if code_name == "MI6":
            from amuse.community.mi6.interface import MI6
            code = MI6(self.converter)
            code.parameters.timestep_parameter = 0.01
            code.parameters.epsilon_squared = (0.0 | units.AU)**2
            code.parameters.calculate_postnewtonian = True
            code.parameters.calculate_postnewtonian_only_first_order = False
            self.drink = code.parameters.drink
            return code
        elif code_name == "ph4":
            from amuse.community.ph4.interface import ph4
            code = ph4(self.converter, number_of_processes=4)
            code.parameters.timestep_parameter = 0.01
            # code.parameters.epsilon_squared = (100.0 | units.AU)**2
            code.parameters.epsilon_squared = (
                self.converter.to_si(
                    (10 | nbody_system.length) / len(self.star_particles)
                )**2
            )
            print(
                "softening length: ",
                (code.parameters.epsilon_squared**0.5).value_in(units.AU),
                "AU")
            code = ph4(self.converter)
            return code
        elif code_name == "BHTree":
            from amuse.community.bhtree.interface import BHTree
            code = BHTree(self.converter)
            code.parameters.timestep = self.converter.to_si(
                0.01 | nbody_system.time
            )
            code.parameters.epsilon_squared = (100 | units.AU)**2
            # code.parameters.epsilon_squared = converter.to_si(
            #     0.01 | nbody_system.length
            # )
            return code
        elif code_name == "Bonsai":
            from amuse.community.bonsai.interface import Bonsai
            code = Bonsai(self.converter)
            code.parameters.timestep = self.converter.to_si(
                0.01 | nbody_system.time
            )
            code.parameters.epsilon_squared = (100 | units.AU)**2
            # code.parameters.epsilon_squared = converter.to_si(
            #     0.01 | nbody_system.length
            # )
            return code
        else:
            raise "No such code implemented"

    def new_field_code(self):
        return FastKick(self.converter)

    def new_bridge(self):
        from amuse.couple.bridge import Bridge
        code = Bridge()
        code.add_system(
            self.gas_code,
            (self.gravity_code),
        )
        code.add_system(
            self.gravity_code,
            (self.gas_code),
        )
        return code
        # self.gravity_to_gas = FastKick(
        #         self.converter,
        #         redirection="none",
        #         #redirect_file=(self.__log_dir+"/gravity_to_gas.log"),
        #         mode="normal"  # "gpu" if self.__use_gpu else "normal",
        #         )
        # self.gas_to_gravity = FastKick(
        #         self.converter,
        #         redirection="none",
        #         #redirect_file=(self.__log_dir+"/gas_to_gravity.log"),
        #         mode="normal"  # "gpu" if self.__use_gpu else "normal",
        #         )

        # self.system = Bridge(
        #         # timestep = timestep_interaction,
        #         use_threading=False,
        #         )
        # self.system.add_system(
        #         self.gas_code,
        #         # (self.__gravity_to_gas),
        #         (self.gravity_code,),
        #         )
        # self.system.add_system(
        #         self.gravity_code,
        #         # (self.__gas_to_gravity),
        #         (self.gas_code,),
        #         )
        # # self.evolve_model(self.model_time)

    @property
    def model_time(self):
        return self.code.model_time

    def sync_energy_to_model(self):
        self.kinetic_energy = self.code.kinetic_energy
        self.potential_energy = self.code.potential_energy
        if "gas" in self.mode:
            self.thermal_energy = self.code.thermal_energy
        else:
            self.thermal_energy = 0 * self.kinetic_energy
        self.total_energy = (
            self.kinetic_energy
            + self.potential_energy
            + self.thermal_energy
        )
        # print(
        #     self.code.particles[0],
        #     self.kinetic_energy,
        #     self.potential_energy,
        #     self.thermal_energy,
        #     self.total_energy,
        # )

    def sync_gas_to_model(self):
        self.gas_code_to_model.copy()

    def sync_stars_to_model(self):
        self.stars_code_to_model.copy()
        self.evolution_to_model.copy_attributes(
            self.evolution_sync_attributes
        )

    def sync_to_code(self):
        if "gas" in self.mode:
            self.sync_model_to_gas()

        if "stars" in self.mode:
            self.sync_model_to_stars()

    def sync_model_to_gas(self):
        self.gas_model_to_code.copy()
        # self.gas_model_to_code.copy_attributes(
        #     self.gas_sync_attributes)

    def sync_model_to_stars(self):
        # self.stars_model_to_code.copy()
        self.gravity_model_to_code.copy_attributes(
            self.gravity_sync_attributes)

    def resolve_sinks(self):
        """
        Find any gas particles above density threshold and turn them into sink
        particles.
        """
        high_density_gas = self.gas_particles.select_array(
            lambda rho: rho > self.density_threshold, ["rho"])
        candidate_stars = high_density_gas.copy()
        self.gas_particles.remove_particles(high_density_gas)
        self.gas_particles.synchronize_to(self.code.gas_particles)
        if candidate_stars.is_empty() > 0:
            print("# Adding %i new stars" % len(candidate_stars))
            new_stars_in_code = self.code.dm_particles.add_particles(
                candidate_stars)
            new_stars = Particles()
            for new_star in new_stars_in_code:
                if new_star not in self.star_particles:
                    new_stars.add_particle(new_star)
                else:
                    print("# This star should not exist")
            new_stars.name = "Star"
            new_stars.birth_age = self.code.model_time
            new_stars.Lx = 0 | (units.g * units.m**2)/units.s
            new_stars.Ly = 0 | (units.g * units.m**2)/units.s
            new_stars.Lz = 0 | (units.g * units.m**2)/units.s

            self.star_particles.add_particles(new_stars)

    def merge_stars(self):
        """
        Find stars within each others' merge radius and merge them
        """
        if len(self.star_particles) <= 0:
            return
        if self.star_particles.radius.max() <= (0 | units.AU):
            return
        connected_components = self.star_particles.copy().connected_components(
            threshold=self.merge_radius
        )
        n_merge = 0
        for colliders in connected_components:
            if len(colliders) > 1:
                n_merge += 1
                merge_two_stars(self.star_particles, colliders.copy())
                self.sync_model_to_stars()
                print("# merged stars")

    def resolve_collision(self):
        gravity = self.star_code
        stellar = self.evolution_code
        collision_detection = self.collision_detection
        bodies = self.star_particles
        # From AMUSE book
        if collision_detection.is_set():
            pre_collision_energy = (
                gravity.kinetic_energy + gravity.potential_energy
            )
            print("# Collision at time=", gravity.model_time.in_(units.Myr))
            number_of_collisions = len(collision_detection.particles(0))
            for i in range(number_of_collisions):
                particles_in_encounter = Particles(
                    particles=[
                        collision_detection.particles(0)[i],
                        collision_detection.particles(1)[i]
                    ]
                )
                particles_in_encounter = \
                    particles_in_encounter.get_intersecting_subset_in(bodies)
                merge_two_stars(bodies, particles_in_encounter)
                bodies.synchronize_to(gravity.particles)
                bodies.synchronize_to(stellar.particles)
                print("Resolved encounter nr %i" % (i+1))
            collision_energy_error = (
                pre_collision_energy
                - (
                    gravity.kinetic_energy + gravity.potential_energy
                )
            )
            print(
                "Energy error in the collisions: dE =",
                collision_energy_error
            )

    def evolve_model(self, t_end):
        time_diag = self.model_time
        while self.model_time < t_end:
            self.i += 1
            time = self.model_time + self.time_step_diag
            # print(
            #     time.in_(units.Myr),
            #     "diag at: ", time_diag.in_(units.Myr)
            # )

            # density_limit_detection.enable()
            # collision_detection.enable()

            if "gas" in self.mode:
                if self.cooling_enabled:
                    self.cooling.evolve_for(0.5 * self.time_step_diag)

            self.code.evolve_model(time)
            time = self.model_time

            if "stars" in self.mode:
                self.evolution_code.evolve_model(time)
                self.sync_stars_to_model()

                self.resolve_collision()
                self.sync_stars_to_model()

            if "gas" in self.mode:
                while self.density_limit_detection.is_set():
                    self.sync_gas_to_model()
                    self.sync_stars_to_model()
                    self.resolve_sinks()

                    # print("..done")
                    self.code.evolve_model(time)
                    self.sync_model_to_stars()
                    # print(
                    #         "end N=",
                    #         len(self.star_particles),
                    #         len(self.code.dm_particles)
                    #         )

                if self.cooling_enabled:
                    self.cooling.evolve_for(0.5 * self.time_step_diag)

            # self.merge_stars()

            # if not self.star_particles.is_empty():
            #     sinks = new_sink_particles(self.star_particles)
            #     sinks.accrete(self.gas_particles)
            #     for sink_i in range(len(self.star_particles)):
            #         self.star_particles[sink_i].Lx += \
            #             sinks[sink_i].angular_momentum[0]
            #         self.star_particles[sink_i].Ly += \
            #             sinks[sink_i].angular_momentum[1]
            #         self.star_particles[sink_i].Lz += \
            #             sinks[sink_i].angular_momentum[2]
            #         self.sync_to_code()

            self.sync_to_model()

            if self.model_time > time_diag:
                time_diag += self.time_step_diag
                if self.model_time > time_diag:
                    time_diag = self.model_time
                self.print_diagnostics()
                if "gas" in self.mode:
                    self.save_gas()
                if "stars" in self.mode:
                    self.save_stars()

    def print_diagnostics(self):
        density_threshold = 1000 | units.MSun * units.parsec**-3
        energy_error = (
            self.total_energy
            - self.tzero_total_energy
        ) / self.tzero_total_energy
        energy_error_last = (
            self.total_energy
            - self.tlast_total_energy
        ) / self.tzero_total_energy
        self.tlast_total_energy = self.total_energy
        self.energy_error_cumulative += abs(energy_error_last)
        try:
            rho_max = self.gas_particles.rho.max()
        except AttributeError:
            rho_max = 0 * density_threshold

        mass_error = (
            self.code.particles.mass.sum()
            - self.tzero_mass
        ) / self.tzero_mass

        print(
            "# Time: ",
            self.model_time.in_(units.Myr),
            "Energy: ",
            self.total_energy.in_(units.erg),
            "Energy conservation error: ",
            energy_error,
            "Cumulative energy conservation error: ",
            self.energy_error_cumulative,
            "Thermal energy: ",
            self.thermal_energy.in_(units.erg),
            "Max rho/rho_crit: ",
            rho_max/density_threshold,
            "Mass error: ",
            mass_error,
        )

        if "gas" in self.mode:
            length_scaling = 30
            # length_scaling = self.converter.to_si(
            #     3.1 | nbody_system.length).value_in(units.parsec)
            plot_hydro(
                self.model_time, self.fig_num, self.code,
                length_scaling=length_scaling,)
        if "stars" in self.mode:
            # subgroups = identify_subgroups(
            #     self.converter,
            #     self.star_particles)
            lagrangian_radii = LagrangianRadii(self.star_particles)
            length_scaling = 30
            plot_stars(
                self.model_time, self.fig_num, self.code,
                length_scaling=length_scaling,)
            print(
                "# ",
                "LR01: ", lagrangian_radii[1],
                "LR05: ", lagrangian_radii[3],
                "LR25: ", lagrangian_radii[5],
                "LR50: ", lagrangian_radii[6],
                "LR75: ", lagrangian_radii[7],
            )
        self.fig_num += 1

    def reset(self):
        # This will currently not work, as it needs new particle sets...
        self.stop()
        self.__init__()

    def stop(self):
        self.code.stop()

    # def select_star_forming_regions(self, alpha_sfe=0.02):
    def generate_stars(
            self,
            alpha_sfe=0.02,
            mass_min=0.3 | units.MSun,
            mass_max=100. | units.MSun):
        """
        Calculate star forming efficiency and select particles above specified
        threshold to become stars.
        """
        gas = self.gas_particles.copy()
        gas.e_loc = alpha_sfe * (
            gas.rho
            / (100 | units.MSun * units.parsec**-3)
        )**0.5
        gas[numpy.where(gas.e_loc > 1)].e_loc = 1
        dense_region_gas = gas.select_array(
            lambda x: x > 1000 | units.MSun * units.parsec**-3,
            ["rho"])
        print(
            "Mean sfe: ", gas.e_loc.mean(),
            "Mean sfe (dense): ", dense_region_gas.e_loc.mean(),
        )

        selection_chance = numpy.random.random(len(gas))
        selected_particles = gas.select_array(
            lambda x: x > selection_chance, ["e_loc"]
        )

        stellar_masses = new_salpeter_mass_distribution(
            len(selected_particles),
            mass_min=mass_min,
            mass_max=mass_max,
        )
        stars = Particles(len(selected_particles))
        stars.position = selected_particles.position
        stars.velocity = selected_particles.velocity
        stars.mass = stellar_masses
        stars.age = 0 | units.Myr
        stars.birth_age = 0 | units.Myr

        evo = new_evolution_code()
        evo.particles.add_particles(stars)
        stars.radius = evo.particles.radius
        evo.stop()

        return stars


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option(
        "-i", dest="filename", default="none",
        help="Input filename [%default]")
    result.add_option(
        "--mode", dest="mode", default="both",
        help="simulation mode (gas, stars, both) [%default]")
    result.add_option(
        "-N", dest="number_of_gas_particles", type="int", default=10000,
        help="number of gas particles [%default]")
    result.add_option(
        "--dt", unit=units.Myr, dest="time_step_diag", type="float",
        default=0.05, help="output timesteps [%default]")
    result.add_option(
        "-M", unit=units.MSun, dest="max_stellar_mass", type="float",
        default=100,
        help="maximal stellar mass [%default] (MSun)")
    result.add_option(
        "-m", unit=units.MSun, dest="min_stellar_mass", type="float",
        default=0.3,
        help="minimal stellar mass [%default] (MSun)")
    result.add_option(
        "-T", unit=units.K, dest="temperature", type="float", default=30,
        help="gas temperature")
    result.add_option(
        "-t", unit=units.Myr, dest="t_end", type="float", default=10.0,
        help="end time of the simulation [%default] (Myr)")
    result.add_option(
        "-z", dest="metallicity", type="float", default=0.02,
        help="metalicity [%default]")
    result.add_option(
        "-s", dest="seed", type="int", default=11,
        help="initial random seed")
    result.add_option(
        "-d", unit=units.cm**-3, dest="cloud_density", default=170,
        help="initial gas cloud density")
    result.add_option(
        "--cloud-weight", unit=units.amu, dest="cloud_weight", default=2.33,
        help="average molecular weight [%default] (amu)")
    return result


def main():
    """
    Run a simulation of a molecular cloud, similar to those in Fujii and
    Portegies Zwart (2016)
    """
    # pylint: disable=too-many-locals
    o, arguments = new_option_parser().parse_args()
    mode = o.mode
    filename = o.filename
    seed = o.seed
    cloud_weight = o.cloud_weight
    cloud_density = o.cloud_density
    temperature = o.temperature
    min_stellar_mass = o.min_stellar_mass
    max_stellar_mass = o.max_stellar_mass
    number_of_gas_particles = o.number_of_gas_particles

    numpy.random.seed(seed)
    # temperature = 30 | units.K

    if mode == "gas" or mode == "both":
        # mass_scale = 400000 | units.MSun
        mass_scale = number_of_gas_particles * (1 | units.MSun)
        # mass_scale = 40000 | units.MSun

        # length_scale = 9.8 | units.parsec
        # length_scale = 13.3 | units.parsec
        # length_scale = 21. | units.parsec
        # length_scale = 6.2 | units.parsec

        rho_cloud = cloud_weight * cloud_density
        length_scale = (3.*mass_scale / (4.*numpy.pi*rho_cloud))**(1/3)
        # rho_cloud = 3.*mass_scale/(4.*numpy.pi*length_scale**3)

        print("# Mass scale=", mass_scale.in_(units.MSun))
        print("# Length scale=", length_scale.in_(units.parsec))
        print(
            "# Cloud density=",
            (rho_cloud).in_(units.MSun * units.parsec**-3)
        )
        print(
            "# Cloud density= %s [cm**-3]" % (
                (rho_cloud/cloud_weight).value_in(units.cm**-3),
            )
        )
        number_of_gas_particles = int(mass_scale / (1 | units.MSun))
        print(
            "# Mass per particle=",
            (mass_scale/number_of_gas_particles).in_(units.MSun)
        )

        converter = nbody_system.nbody_to_si(
            length_scale,
            mass_scale,
        )

        # print(
        #     "# Cloud density=",
        #     (rho_cloud).in_(units.MSun * units.parsec**-3)
        # )
        if o.filename is "none":
            gas_particles = molecular_cloud(
                nf=32,  # default
                power=-4.,  # spectral index as in FPZ2016
                targetN=number_of_gas_particles,
                # ethep_ratio=0.01*0.828,  # 0.01*1.53 for 30K?
                ekep_ratio=1.,  # as in FPZ2016
                seed=seed,  # change this for other realisations
                convert_nbody=converter,  # set scale
                base_grid=body_centered_grid_unit_cube,  # default
            ).result
        else:
            gas_particles = read_set_from_file(filename, "amuse").history[0]

        t_ff = gas_particles.dynamical_timescale()
        # t_ff = 0.5427/numpy.sqrt(constants.G*rho_cloud)
        print("# Freefall timescale=", t_ff.in_(units.Myr))

        internal_energy = (
            3.0 * constants.kB * temperature
            / (2.0 * cloud_weight)
        )
        gas_particles.u = internal_energy

        print(
            "# Velocity dispersion: %s [kms]" % (
                gas_particles.velocity.lengths().mean().value_in(units.kms),
            )
        )
        gas_particles.name = "gas"

        cluster = Cluster(
            (gas_particles, Particles()),
            convert_nbody=converter,
            mode="gas"
        )
        cluster.save_gas()

        print(
            "# Hydro timesteps:",
            cluster.gas_code.parameters.timestep.in_(units.yr),
            "number of gas particles=", number_of_gas_particles)

        print("# gas center-of-mass: ", cluster.gas_particles.center_of_mass())
        print("# time: ", cluster.model_time.in_(units.yr))
        t_end = 0.9 * t_ff
        cluster.evolve_model(t_end)

        stars = cluster.generate_stars(
            mass_min=min_stellar_mass, mass_max=max_stellar_mass,)

        cluster.stop()
        del cluster

    if mode == "stars" or mode == "both":
        if mode == "stars":
            try:
                stars = read_set_from_file(
                    filename,
                    "amuse",
                ).history[0]
            except IoException:
                print("Could not read file %s, exiting" % filename)
                exit()

        print("Number of stars: ", len(stars))
        if stars.is_empty():
            exit()

        scale_length = stars.virial_radius().in_(units.parsec)
        scale_mass = stars.mass.sum()
        print("Scale length: %s" % scale_length.in_(units.parsec))
        print("Scale mass: %s" % scale_mass.in_(units.MSun))
        converter = nbody_system.nbody_to_si(
            scale_length,
            scale_mass,
        )

        cluster = Cluster(
            (Particles(), stars),
            convert_nbody=converter,
            mode="stars"
        )
        cluster.save_stars()

        cluster.evolve_model(10 | units.Myr)


if __name__ == "__main__":
    set_printing_strategy(
        "custom",
        preferred_units=[
            units.MSun, units.parsec, units.Myr, units.erg,
        ],
        precision=6, prefix="", separator=" [", suffix="]"
    )
    main()
