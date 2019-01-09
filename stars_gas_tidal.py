"""
Simulate a system of stars and gas, with an external tidal field
"""
from __future__ import print_function, division

import sys

import numpy

from amuse.units import units, nbody_system
from amuse.units.quantities import VectorQuantity
from amuse.io import write_set_to_file, read_set_from_file
from amuse.community.ph4.interface import ph4
from amuse.community.fastkick.interface import FastKick
from amuse.community.seba.interface import SeBa
from amuse.community.fi.interface import Fi
# from amuse.community.bhtree.interface import BHTree
from amuse.couple.bridge import Bridge
from amuse.datamodel import Particles, ParticlesSuperset

from spiral_potential import (
    TimeDependentSpiralArmsDiskModel,
)
from plot_models import plot_cluster


def sfe_to_density(e_loc, alpha=0.02):
    "Calculate density needed for specified star formation efficiency"
    rho = 100 * (e_loc / alpha)**2 | units.MSun * units.parsec**-3
    return rho


def density_to_sfe(rho, alpha=0.02):
    "Calculate star formation efficiency for specified density"
    sfe = alpha * (rho.value_in(100 * units.MSun * units.parsec**-3))**0.5
    return sfe


class Gas(object):
    """Gas object"""

    def __init__(
            self,
            gas=None,
            converter=None,
            # mass_scale=1000 | units.MSun,
            # length_scale=100 | units.parsec,
            epsilon=0.1 | units.parsec,
            alpha_sfe=0.02,
    ):
        self.alpha_sfe = alpha_sfe
        if converter is not None:
            self.gas_converter = converter
        else:
            mass_scale = gas.mass.sum()
            # should be something related to length spread?
            length_scale = 100 | units.parsec
            self.gas_converter = nbody_system.nbody_to_si(
                mass_scale,
                length_scale,
            )
        # We're simulating gas as gas, not just "dark matter".
        if gas is None:
            self.gas_particles = Particles()
        else:
            self.gas_particles = gas
        self.gas_code = Fi(self.gas_converter)
        self.gas_code.parameters.epsilon_squared = (epsilon)**2
        self.gas_code.parameters.timestep = self.gas_converter.to_si(
            0.01 | nbody_system.time,  # 0.05 | units.Myr
        )
        self.gas_code.gas_particles.add_particles(self.gas_particles)
        self.model_to_gas_code = self.gas_particles.new_channel_to(
            self.gas_code.gas_particles,
        )
        self.gas_code_to_model = self.gas_code.gas_particles.new_channel_to(
            self.gas_particles,
        )

        # We want to control cooling ourselves
        self.gas_code.parameters.gamma = 1
        self.gas_code.parameters.isothermal_flag = True

        # Sensible to set these, even though they are already default
        self.gas_code.parameters.use_hydro_flag = True
        self.gas_code.parameters.self_gravity_flag = True
        self.gas_code.parameters.verbosity = 0

        # We want to stop and create stars when a certain density is reached
        self.gas_code.parameters.stopping_condition_maximum_density = \
            sfe_to_density(1, alpha=self.alpha_sfe)

        """
        acc_timestep_crit_constant: 0.25
        acc_timestep_flag: True
        adaptive_smoothing_flag: False
        artificial_viscosity_alpha: 0.5
        balsara_flag: False
        begin_time: 0.0 time
        beta: 1.0
        code_length_unit: 1.0 kpc
        code_mass_unit: 1000000000.0 MSun
        conservative_sph_flag: True
        cool_par: 1.0
        courant: 0.3
        direct_sum_flag: False
        enforce_min_sph_grav_softening_flag: False
        eps_is_h_flag: True
        epsilon_squared: 0.0 length * length
        feedback: fuv
        fi_data_directory: /Users/rieder/Env/Amuse2/share/amuse/data/fi/input/
        first_snapshot: 0
        fixed_halo_flag: False
        free_timestep_crit_constant_a: 0.35
        free_timestep_crit_constant_aexp: -1.0
        free_timestep_crit_constant_v: 0.5
        free_timestep_crit_constant_vexp: 0.0
        freeform_timestep_flag: False
        gadget_cell_opening_constant: 0.01
        gadget_cell_opening_flag: True
        gas_epsilon: 0.005 length
        grain_heat_eff: 0.05
        h_update_method: mass
        halofile: none
        heat_par1: 0.0
        heat_par2: 0.0
        integrate_entropy_flag: True
        log_interval: 5
        max_density: 100.0 mass / (length**3)
        maximum_time_bin: 4096
        min_gas_part_mass: 0.25
        minimum_part_per_bin: 1
        n_smooth: 64
        n_smooth_tol: 0.1
        nn_tol: 0.1
        opening_angle: 0.5
        optical_depth: 0.0
        output_interval: 5
        periodic_boundaries_flag: False  (read only)
        periodic_box_size: 10000.0 length
        quadrupole_moments_flag: False
        radiation_flag: False
        smooth_input_flag: False
        sph_artificial_viscosity_eps: 0.01
        sph_dens_init_flag: True
        sph_h_const: 0.2 length
        sph_viscosity: sph
        sqrt_timestep_crit_constant: 1.0
        square_root_timestep_flag: False
        star_form_delay_fac: 1.0
        star_form_eff: 0.25
        star_form_mass_crit: 100000.0 MSun
        star_formation_flag: False
        star_formation_mode: gerritsen
        stopping_condition_maximum_internal_energy:\
            1.79769313486e+308 length**2 / (time**2)
        stopping_condition_minimum_density: -1.0 mass / (length**3)
        stopping_condition_minimum_internal_energy: -1.0 length**2 / (time**2)
        stopping_conditions_number_of_steps: 1
        stopping_conditions_out_of_box_size: 0.0 length
        stopping_conditions_out_of_box_use_center_of_mass: True
        stopping_conditions_timeout: 4.0 s
        supernova_duration: 30000000.0 Myr
        supernova_eff: 0.0
        t_supernova_start: 3000000.0 Myr
        targetnn: 32
        timestep: 1.0 time
        zeta_cr_ion_rate: 3.6 1.8e-17 * s**-1
        """
        # self.code = self.gas_code
        # self.particles = self.gas_particles

    @property
    def model_time(self):
        "Return the current time in the code"
        return self.gas_code.model_time

    def get_gravity_at_point(self, *args, **kwargs):
        "Return gravity at specified location"
        return self.gas_code.get_gravity_at_point(*args, **kwargs)

    # @property
    # def particles(self):
    #     return self.code.particles

    def evolve_model(self, time, sync=True):
        "Evolve model to specified time and synchronise model"
        if sync:
            self.model_to_gas_code.copy()
        self.gas_code.evolve_model(time)
        if sync:
            self.gas_code_to_model.copy()

    def stop(self):
        "Stop code"
        self.gas_code.stop()


class Cluster(object):
    """Stellar cluster object"""

    def __init__(
            self,
            stars=None,
            converter=None,
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
        self.star_code = ph4(
            self.star_converter, number_of_workers=number_of_workers,
        )
        # self.star_code = BHTree(self.converter)
        self.star_code.parameters.epsilon_squared = (epsilon)**2
        self.star_code.particles.add_particles(self.star_particles)
        self.evo_code = SeBa()
        self.evo_code.particles.add_particles(self.star_particles)
        # self.evo_code.parameters.metallicity = 0.01
        self.model_to_star_code = self.star_particles.new_channel_to(
            self.star_code.particles,
        )
        self.star_code_to_model = self.star_code.particles.new_channel_to(
            self.star_particles,
        )
        self.evo_code_to_model = self.evo_code.particles.new_channel_to(
            self.star_particles,
        )
        # self.particles = self.star_particles
        # self.code = self.star_code

    @property
    def model_time(self):
        "Return the minimum time of the star and evo code times"
        star_code_time = self.star_code.model_time
        evo_code_time = self.evo_code.model_time
        time = min(star_code_time, evo_code_time)
        return time

    # This is not always supported by the stellar gravity code
    # If it is not, we should do something about that...
    def get_gravity_at_point(self, *args, **kwargs):
        "Return gravity at specified location"
        return self.star_code.get_gravity_at_point(*args, **kwargs)

    # @property
    # def particles(self):
    #     return self.code.particles

    def evolve_model(self, time):
        "Evolve gravity and stellar evolution to specified time"
        self.model_to_star_code.copy()
        self.evo_code.evolve_model(time)
        self.star_code.evolve_model(time)
        self.evo_code_to_model.copy()
        self.star_code_to_model.copy()

    def stop(self):
        "Stop star_code and evo_code"
        self.star_code.stop()
        self.evo_code.stop()


Tide = TimeDependentSpiralArmsDiskModel


class ClusterInPotential(
        # object,
        Cluster,
        Gas,
        Tide,
):
    """Stellar cluster in an external potential"""

    def __init__(
            self,
            stars=None,
            gas=None,
            epsilon=0.1 | units.parsec,
    ):
        self.objects = (Cluster, Gas, Tide)
        mass_scale_stars = stars.mass.sum()
        length_scale_stars = 1 | units.parsec
        converter_for_stars = nbody_system.nbody_to_si(
            mass_scale_stars,
            length_scale_stars,
        )
        Cluster.__init__(
            self, stars=stars, converter=converter_for_stars, epsilon=epsilon,
        )
        # self.cluster_code = Cluster(
        #     stars, converter=self.converter_for_stars, epsilon=epsilon,
        # )

        mass_scale_gas = gas.mass.sum()
        length_scale_gas = 10 | units.parsec
        converter_for_gas = nbody_system.nbody_to_si(
            mass_scale_gas,
            length_scale_gas,
        )
        Gas.__init__(
            self, gas=gas, converter=converter_for_gas, epsilon=epsilon,
        )
        # self.gas_code = Gas(
        #     gas, converter=self.converter_for_gas, epsilon=epsilon,
        # )
        Tide.__init__(self)
        # self.potential = TimeDependentSpiralArmsDiskModel()

        self.gravity_field_code = FastKick(
            self.star_converter, mode="cpu", number_of_workers=2,
        )
        self.gravity_field_code.parameters.epsilon_squared = (
            epsilon,
        )**2

        self.system = Bridge()
        self.system.add_system(
            # self.star_code,
            Cluster,
            # [Tide, self.gas_code],
            [Tide, Gas],
            True
        )
        self.system.add_system(
            # self.gas_code,
            Gas,
            # [Tide, self.star_code],
            [Tide, Cluster],
            True
        )
        self.system.timestep = 0.05 | units.Myr
        # self.converter.to_si(
        #     0.01 | nbody_system.time,
        # )
        # self.particles = self.system.particles
        # self.stars = self.cluster_code.star_particles
        # self.gas = self.gas_code.gas_particles
        self.particles = ParticlesSuperset(
            self.star_particles,
            self.gas_particles,
        )
        self.code = self.system

    def evolve_model(self, time):
        "Evolve system to specified time"
        self.system.evolve_model(time)

    def get_gravity_at_point(self, *args, **kwargs):
        force = VectorQuantity(
            [0, 0, 0],
            unit=units.m * units.s**-2,
        )
        for parent in self.objects:
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
    "Load stars and gas, and evolve them"
    if len(sys.argv) > 1:
        stars = read_set_from_file(sys.argv[1], "amuse")
        if len(sys.argv) > 2:
            gas = read_set_from_file(sys.argv[2], "amuse")
        else:
            gas = None
        # print(gas.dynamical_timescale())
        # exit()
        model = ClusterInPotential(
            stars=stars,
            gas=gas,
        )
    else:
        return

    length = units.parsec
    timestep = 0.1 | units.Myr
    time = 0 | units.Myr
    for i in range(100):
        time += timestep
        print("Evolving to %s" % time.in_(units.Myr))
        model.evolve_model(time)
        print("Model time: %s" % (model.model_time))
        write_set_to_file(
            model.star_particles,
            "model-stars-%04i.hdf5" % i, "amuse")
        write_set_to_file(
            model.gas_particles,
            "model-gas-%04i.hdf5" % i, "amuse")
        x, y = model.particles.center_of_mass()[0:2]
        plot_cluster(
            model.star_particles,
            xmin=(-400 + x.value_in(length)),
            xmax=(400 + x.value_in(length)),
            ymin=(-400 + y.value_in(length)),
            ymax=(400 + y.value_in(length)),
            i=i,
            name="model-ph4",
        )


if __name__ == "__main__":
    numpy.random.seed(1)
    main()
