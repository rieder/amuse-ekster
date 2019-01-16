"Class for a gas system"
from __future__ import print_function, division
from amuse.units import units, nbody_system
from amuse.datamodel import Particles


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
        from amuse.community.fi.interface import Fi
        self.gas_code = Fi(
            self.gas_converter,
            mode="openmp",
            # redirection="none",
        )
        # from amuse.community.gadget2.interface import Gadget2
        # self.gas_code = Gadget2(
        #     self.gas_converter,
        #     # redirection="none",
        # )
        self.gas_code.parameters.epsilon_squared = (epsilon)**2
        self.gas_code.parameters.timestep = min(
            self.gas_converter.to_si(
                0.01 | nbody_system.time,  # 0.05 | units.Myr
            ),
            0.01 | units.Myr,
        )
        self.gas_code.gas_particles.add_particles(self.gas_particles)
        self.model_to_gas_code = self.gas_particles.new_channel_to(
            self.gas_code.gas_particles,
        )
        self.gas_code_to_model = self.gas_code.gas_particles.new_channel_to(
            self.gas_particles,
        )

        # We want to control cooling ourselves
        # self.gas_code.parameters.gamma = 1
        self.gas_code.parameters.gamma = 5./3.
        # self.gas_code.parameters.isothermal_flag = True
        self.gas_code.parameters.isothermal_flag = False

        # Sensible to set these, even though they are already default
        self.gas_code.parameters.use_hydro_flag = True
        self.gas_code.parameters.self_gravity_flag = True
        self.gas_code.parameters.verbosity = 0

        # We want to stop and create stars when a certain density is reached
        # print(c.stopping_conditions.supported_conditions())
        self.gas_code.parameters.stopping_condition_maximum_density = \
            sfe_to_density(1, alpha=self.alpha_sfe)
        self.gas_stopping_conditions = \
            self.gas_code.stopping_conditions.density_limit_detection
        self.gas_stopping_conditions.enable()

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

    def evolve_model(self, tend):
        "Evolve model to specified time and synchronise model"
        self.model_to_gas_code.copy()
        while (
                abs(tend - self.model_time) > self.gas_code.parameters.timestep
        ):
            print(
                "Evolving to %s (now at %s)" % (
                    tend.in_(units.Myr),
                    self.model_time.in_(units.Myr),
                )
            )
            self.gas_code.evolve_model(tend)
            # Check stopping conditions
            if self.gas_stopping_conditions.is_set():
                print("Gas code stopped - max density reached")
        self.gas_code_to_model.copy()

    def stop(self):
        "Stop code"
        self.gas_code.stop()


def main():
    "Test class with a molecular cloud"
    from amuse.ext.molecular_cloud import molecular_cloud
    converter = nbody_system.nbody_to_si(
        10000 | units.MSun,
        10 | units.parsec,
    )
    gas = molecular_cloud(targetN=10000, convert_nbody=converter).result
    print("Number of gas particles: %i" % (len(gas)))

    model = Gas(gas=gas, converter=converter)
    print(model.gas_code.parameters)
    timestep = 0.1 | units.Myr
    for step in range(10):
        time = step * timestep
        model.evolve_model(time)
        print("Evolved to %s" % model.model_time)
        print(
            "Maximum density / stopping density = %s" % (
                model.gas_particles.density.max()
                / model.gas_code.parameters.stopping_condition_maximum_density,
            )
        )


if __name__ == "__main__":
    main()
