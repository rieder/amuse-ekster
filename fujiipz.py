#!/usr/bin/python
# enc: utf-8
"""
Simulate a molecular gas cloud
"""
from __future__ import division, print_function

import sys
import numpy

import matplotlib.pyplot as pyplot

from amuse.units import (nbody_system, units, constants)
from amuse.datamodel import (Particles, ParticlesSuperset)

from amuse.io import write_set_to_file

from amuse.ext.molecular_cloud import molecular_cloud
from amuse.ext.sink import new_sink_particles
from amuse.ext.evrard_test import body_centered_grid_unit_cube
# from amuse.ic.gasplummer import new_plummer_gas_model
# from amuse.ic.plummer import new_plummer_model
# from amuse.community.fastkick.interface import FastKick
# from amuse.couple.bridge import Bridge
# import matplotlib.pyplot as plt

from cooling_class import SimplifiedThermalModelEvolver
from prepare_figure import single_frame
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
    rho = make_map(sph, number_of_cells=200, length_scaling=length_scaling)
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

    pyplot.savefig('fig-%04i.png' % i)
    pyplot.close(fig)


def merge_two_stars(bodies, particles_in_encounter):
    """
    Merge two stars into one
    """
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    new_particle = Particles(1)
    # mu = particles_in_encounter[0].mass/particles_in_encounter.mass.sum()
    new_particle.birth_age = particles_in_encounter.birth_age.min()
    new_particle.mass = particles_in_encounter.total_mass()
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
        com_pos.length().in_(units.AU)
    )
    bodies.remove_particles(particles_in_encounter)


class Cluster(object):
    """
    Create a simulator for an embedded star cluster
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=missing-docstring
    # Important methods are described, others are self-explanatory

    def __init__(
            self,
            gas_particles,
            star_particles,
            length_scale=1 | units.parsec,
            mass_scale=1000 | units.MSun,
            # time_init=0 | units.yr,
            system_size=200 | units.parsec,
            epsilon=0.1 | units.parsec,):
        # pylint: disable=too-many-arguments
        # self.model_time = time_init
        self.converter = nbody_system.nbody_to_si(
            length_scale,
            mass_scale,
        )
        self.gas_particles = Particles()  # gas_particles
        self.star_particles = Particles()  # star_particles
        self.particles = ParticlesSuperset(
            [self.gas_particles, self.star_particles],
        )
        self.gas_sync_attributes = (
            "mass", "x", "y", "z", "vx", "vy", "vz", "u",
        )
        self.stars_sync_attributes = (
            "mass", "x", "y", "z", "vx", "vy", "vz",
        )
        self.evolution_sync_attributes = (
            "mass", "radius",
        )
        self.use_gpu = False
        self.log_dir = "./log/"
        self.system_size = system_size
        self.epsilon = epsilon
        self.density_threshold = (1 | units.MSun)/(self.epsilon)**3
        cloud_mass = gas_particles.mass.sum()
        number_of_gas_particles = len(gas_particles)
        self.time_step = (
            0.004*numpy.pi*numpy.power(self.epsilon, 1.5)
            / numpy.sqrt(constants.G*cloud_mass/number_of_gas_particles)
        )
        self.time_step = 5.e-2 | units.Myr
        print(
            "# Hydro timesteps:", (0.5 * self.time_step).in_(units.yr),
            "number of gas particles=", number_of_gas_particles)
        self.time_step_diag = self.time_step  # 0.01 | units.Myr
        self.cooling_enabled = False
        self.cooling_flag = "thermal_model"
        self.density_threshold = (1 | units.MSun)/(self.epsilon)**3
        self.merge_radius = 0.2*self.epsilon

        self.initialise_gas_code()
        # self.initialise_stars_code()
        # self.initialise_bridge()
        self.setup_channels()
        self.add_gas(gas_particles)
        self.add_stars(star_particles)
        self.sync_energy_to_model()
        self.tzero_thermal_energy = self.thermal_energy
        self.tzero_total_energy = (
            self.kinetic_energy
            + self.potential_energy
            + self.thermal_energy
        )
        self.tlast_total_energy = self.tzero_total_energy
        self.energy_error_cumulative = 0.0
        self.tzero_mass = (
            self.code.gas_particles.mass.sum()
        )
        print("# E_thermal =", self.thermal_energy)
        # mtot = N*(100.0|units.MSun)

        hydrogen_atomic_mass = 1./6.02214179e+23 | units.g
        weight = 2.33
        temperature = (
            self.thermal_energy
            * 2. / (3.*cloud_mass/(weight*hydrogen_atomic_mass)*constants.kB)
        )
        print("# Temperature = ", temperature.in_(units.K))

    def initialise_gas_code(self):
        """Initialises the SPH code we will use"""
        # from amuse.community.gadget2.interface import Gadget2
        from amuse.community.fi.interface import Fi
        # from hydrodynamics_class import Hydro
        self.code = Fi(self.converter, mode="openmp")

        # self.code.parameters.begin_time = 0.0 | units.Myr
        self.code.parameters.use_hydro_flag = True
        self.code.parameters.radiation_flag = False
        # self.code.parameters.self_gravity_flag = True
        """
        isothermal_flag:
        When True then we have to do our own adiabatic cooling
        (and Gamma has to be 1.0)
        When False then we dont do the adiabatic cooling and Fi
        is changing u
        """
        self.code.parameters.gamma = 1
        self.code.parameters.isothermal_flag = True
        self.code.parameters.integrate_entropy_flag = False
        self.code.parameters.timestep = self.time_step
        self.code.parameters.verbosity = 0
        self.code.parameters.eps_is_h_flag = False
        self.code.parameters.gas_epsilon = self.epsilon
        self.code.parameters.sph_h_const = self.epsilon
        # self.code.parameters.periodic_box_size = 20.*self.system_size
        self.code.parameters.periodic_box_size = 500 | units.parsec
        # self.code.parameters.verbosity = 99

        self.code.parameters.integrate_entropy_flag = False
        self.gamma = self.code.parameters.gamma

        self.code.parameters.stopping_condition_maximum_density = \
            self.density_threshold

        # self.code = Hydro(gas_code, self.gas_particles, Particles())
        self.code.parameters.stopping_condition_maximum_density = \
            self.density_threshold

        # self.code.commit_parameters()

        self.cooling = SimplifiedThermalModelEvolver(
            self.code.gas_particles)
        self.cooling.model_time = self.code.model_time

    # def initialise_stars_code(self):
    #     from amuse.community.bhtree.interface import BHTree
    #     self.stars_code = BHTree(self.converter)

    def setup_channels(self):
        """Set up channels to copy code data to and from model"""
        self.gas_code_to_model = self.code.gas_particles.new_channel_to(
            self.gas_particles,
        )
        self.gas_model_to_code = self.gas_particles.new_channel_to(
            self.code.gas_particles,
        )

        self.stars_code_to_model = \
            self.code.dm_particles.new_channel_to(
                self.star_particles,
            )
        self.stars_model_to_code = \
            self.star_particles.new_channel_to(
                self.code.dm_particles,
            )

    # def new_field_code(self):
    #     field_code = FastKick(self.converter)

    # def initialise_bridge(self):

    #     # self.gravity_to_gas = new_field_code(
    #     #         self.stars_gravity_code,
    #     #         field_gravity_code,
    #     #         )
    #     # self.gravity_to_gas = FastKick(
    #     #         self.converter,
    #     #         redirection="none",
    #     #         #redirect_file=(self.__log_dir+"/gravity_to_gas.log"),
    #     #         mode="normal"  # "gpu" if self.__use_gpu else "normal",
    #     #         )
    #     # self.gas_to_gravity = FastKick(
    #     #         self.converter,
    #     #         redirection="none",
    #     #         #redirect_file=(self.__log_dir+"/gas_to_gravity.log"),
    #     #         mode="normal"  # "gpu" if self.__use_gpu else "normal",
    #     #         )

    #     self.system = Bridge(
    #             # timestep = timestep_interaction,
    #             use_threading=False,
    #             )
    #     self.system.add_system(
    #             self.gas_code,
    #             # (self.__gravity_to_gas),
    #             (self.gravity_code,),
    #             )
    #     self.system.add_system(
    #             self.gravity_code,
    #             # (self.__gas_to_gravity),
    #             (self.gas_code,),
    #             )
    #     # self.evolve_model(self.model_time)

    @property
    def model_time(self):
        return self.code.model_time

    def sync_to_model(self):
        self.sync_gas_to_model()
        self.sync_stars_to_model()
        self.sync_energy_to_model()

    def sync_energy_to_model(self):
        self.kinetic_energy = self.code.kinetic_energy
        self.potential_energy = -self.code.potential_energy
        self.thermal_energy = self.code.thermal_energy
        self.total_energy = (
            self.kinetic_energy
            + self.potential_energy
            + self.thermal_energy
        )

    def sync_gas_to_model(self):
        self.gas_code_to_model.copy()

    def sync_stars_to_model(self):
        self.stars_code_to_model.copy()

    def sync_to_code(self):
        self.sync_model_to_gas()
        self.sync_model_to_stars()

    def sync_model_to_gas(self):
        self.gas_model_to_code.copy()
        # self.gas_model_to_code.copy_attributes(
        #         self.gas_sync_attributes)

    def sync_model_to_stars(self):
        self.stars_model_to_code.copy()
        # self.gravity_model_to_code.copy_attributes(
        #         self.gravity_sync_attributes)

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

    def evolve_model(self, t_end):
        time_diag = self.model_time
        fig_num = 0
        while self.model_time < t_end:
            time = self.model_time + self.time_step_diag
            density_limit_detection = \
                self.code.stopping_conditions.density_limit_detection
            # density_limit_detection.enable()

            if self.cooling_enabled:
                self.cooling.evolve_for(0.5 * self.time_step_diag)
            self.code.evolve_model(time)

            while density_limit_detection.is_set():
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

            self.merge_stars()

            if not self.star_particles.is_empty() > 0:
                sinks = new_sink_particles(self.star_particles)
                sinks.accrete(self.gas_particles)
                for sink_i in range(len(self.star_particles)):
                    self.star_particles[sink_i].Lx += \
                        sinks[sink_i].angular_momentum[0]
                    self.star_particles[sink_i].Ly += \
                        sinks[sink_i].angular_momentum[1]
                    self.star_particles[sink_i].Lz += \
                        sinks[sink_i].angular_momentum[2]

                    self.sync_to_code()

            self.sync_to_model()
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
                rho_max = 0 * self.density_threshold

            mass_error = (
                self.code.particles.mass.sum()
                - self.tzero_mass
            ) / self.tzero_mass
            print(
                "Time: ",
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
                rho_max/self.density_threshold,
                "Mass error: ",
                mass_error,
            )
            if time > time_diag:
                time_diag += self.time_step_diag
                plot_hydro(time_diag, fig_num, self.code, length_scaling=20,)
                fig_num += 1

    def add_gas(self, gas_particles):
        gas_particles.h_smooth = self.epsilon
        self.gas_particles.add_particles(gas_particles)
        self.code.gas_particles.add_particles(self.gas_particles)

    def add_stars(self, star_particles):
        self.star_particles.add_particles(star_particles)
        self.code.dm_particles.add_particles(star_particles)

    def reset(self):
        self.stop()
        self.__init__()

    def stop(self):
        self.code.stop()

    def remove_gas(self, gas_particles):
        self.gas_particles.remove_particles(gas_particles)
        self.code.gas_particles.remove_particles(gas_particles)

    def remove_stars(self, star_particles):
        self.star_particles.remove_particles(star_particles)
        self.code.dm_particles.remove_particles(star_particles)

    def select_star_forming_regions(self, alpha_sfe=0.02, sfe_cutoff=0.):
        """
        Calculate star forming efficiency and select particles above specified
        threshold.
        """
        gas = self.gas_particles.copy()
        gas.e_loc = alpha_sfe * (
            gas.rho
            / (100 | units.MSun * units.parsec**-3)
        )**0.5
        gas[numpy.where(gas.e_loc > 1)].e_loc = 1
        return gas.select_array(
            lambda x:
            x > sfe_cutoff,
            ["e_loc"]
        )


def main():
    """
    Run a simulation of a molecular cloud, similar to those in Fujii and
    Portegies Zwart (2016)
    """
    # pylint: disable=too-many-locals
    try:
        seed = int(sys.argv[1])
    except IndexError:
        print("No seed specified, using default")
        seed = 11
    except ValueError:
        print("Invalid seed, using default")
        seed = 11
    numpy.random.seed(seed)
    temperature = 30 | units.K

    mass_scale = 400000 | units.MSun
    # mass_scale = 10000 | units.MSun
    # mass_scale = 40000 | units.MSun

    # length_scale = 9.8 | units.parsec
    # length_scale = 13.3 | units.parsec
    # length_scale = 21. | units.parsec
    # length_scale = 6.2 | units.parsec

    density_scale = (2.33 | units.amu)

    rho_cloud = density_scale * (170 | units.cm**-3)
    length_scale = (3.*mass_scale / (4.*numpy.pi*rho_cloud))**(1/3)
    rho_cloud = 3.*mass_scale/(4.*numpy.pi*length_scale**3)

    print("# Mass scale=", mass_scale.in_(units.MSun))
    print("# Length scale=", length_scale.in_(units.parsec))
    print("# Cloud density=", (rho_cloud).in_(units.MSun * units.parsec**-3))
    print("# Cloud density=", (rho_cloud/density_scale).in_(units.cm**-3))
    number_of_gas_particles = int(mass_scale / (1 | units.MSun))
    print(
        "# Mass per particle=",
        (mass_scale/number_of_gas_particles).in_(units.MSun)
    )

    converter = nbody_system.nbody_to_si(
        length_scale,
        mass_scale,
    )

    t_ff = 0.5427/numpy.sqrt(constants.G*rho_cloud)
    print("# Freefall timescale=", t_ff.in_(units.Myr))
    # print("# Cloud density=", (rho_cloud).in_(units.MSun * units.parsec**-3))
    # exit()

    # from molecular_cloud import molecular_cloud

    # gas_particles = new_plummer_gas_model(
    #         number_of_particles=number_of_gas_particles,
    #         convert_nbody=converter,
    #         )
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
    hydrogen_atomic_mass = 1./6.02214179e+23 | units.g
    weight = 2.33
    gas_mean_molecular_weight = weight * hydrogen_atomic_mass

    internal_energy = (
        3.0 * constants.kB * temperature
        / (2.0 * gas_mean_molecular_weight)
    )
    gas_particles.u = internal_energy

    print(
        "# Velocity dispersion:",
        gas_particles.velocity.lengths().mean().in_(units.kms),
    )
    gas_particles.name = "gas"
    star_particles = Particles()

    cluster = Cluster(
        gas_particles,
        star_particles,
        length_scale=length_scale,
        mass_scale=mass_scale,
    )

    print("# number of gas particles: ", len(cluster.gas_particles))
    # print("# number of star particles: ", len(cluster.star_particles))
    print("# gas center-of-mass: ", cluster.gas_particles.center_of_mass())
    print("# time: ", cluster.model_time.in_(units.yr))
    t_end = 0.9 * t_ff
    cluster.evolve_model(t_end)

    star_forming_gas = cluster.select_star_forming_regions()
    print(
        "# Max sfe: ", star_forming_gas.e_loc.max(),
        " Mean sfe: ", star_forming_gas.e_loc.mean(),
    )
    write_set_to_file(star_forming_gas, "star-forming-gas.hdf5", "amuse")


if __name__ == "__main__":
    main()
