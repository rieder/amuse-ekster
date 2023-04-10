#!/usr/bin/env python3
"""
Program to increase the resolution of SPH particles in a snapshot.
First developed by Thomas Bending (Bending et al. 2020, 2020MNRAS.495.1672B),
modified for AMUSE by Steven Rieder.
"""
import sys
import numpy

from amuse.io import write_set_to_file
from amuse.datamodel import Particles
from amuse.datamodel.rotation import rotated
from amuse.units import units, nbody_system
from amuse.units.trigo import pi, sin, cos, arcsin, arccos
from amuse.community.fi.interface import Fi


def cubic(q):
    # sigma is the normalising constant
    sigma = 1./numpy.pi
    if q >= 0. and q < 1.:
        wofq = ((2.-q)**3/4.-(1.-q)**3)
    elif q >= 1. and q < 2.:
        wofq = ((2.-q)**3/4.)
    elif q >= 2.:
        wofq = 0.
    else:
        print('q is negative. q = ', q)
    return wofq*sigma


def find_shell_struct(
        res_increase_factor=85,
):
    particles_per_shell = numpy.zeros(
        (0),
        dtype=int,
    )
    shell_radii = numpy.array([0])  # in q-space
    shell_spacing = numpy.zeros((0), dtype=float)
    shells = 0

    shell_spacing_values = {
        30: 0.48,
        85: 0.336,
        311: 0.216,
        528: 0.181,
        823: 0.156,
        1187: 0.138,
        1632: 0.124,
    }

    shell_spacing_value = shell_spacing_values[res_increase_factor]

    shell_spacing = numpy.append(shell_spacing, shell_spacing_value)
    w0 = cubic(0.)**-(1./3.)

    # Find shell locations through iteration
    while (True):
        shells += 1
        shell_radii = numpy.append(
            shell_radii,
            shell_radii[shells-1] + shell_spacing[shells-1])
        if (
                shell_radii[shells] >= 2.0
                or shell_radii[shells-1] >= 2.0
        ):
            break
        for its in range(30):
            # print("its %i" % its)
            avrspac = 0.
            for j in range(101):
                rsplit = (
                    shell_radii[shells-1]
                    + (
                        shell_radii[shells]
                        - shell_radii[shells-1]
                    ) * j/100
                )
                avrspac += shell_spacing_value*cubic(rsplit)**-(1./3.)/w0
            avrspac /= 100.
            shell_radii[shells] = shell_radii[shells-1]+avrspac

            if (shell_radii[shells] >= 2.0):
                break
        if (shell_radii[shells] >= 2.0):
            break
        shell_spacing = numpy.append(
            shell_spacing,
            shell_spacing_value*cubic(shell_radii[shells])**-(1./3.)/w0
        )
    shell_radii = shell_radii[0:-1]

    # Approximate number of particles on each shell
    particles_per_shell = numpy.empty((shells), dtype=int)
    particles_per_shell[0] = 1
    for i in range(1, shells):
        particles_per_shell[i] = numpy.ceil(
            pi
            / (
                arcsin(shell_spacing[i]*0.5/shell_radii[i])
            )**2
        )

    return shells, particles_per_shell, shell_radii


def pos_shift(
    shell_radii,
    particles_per_shell,
    res_increase_factor=85,
):
    """
    Return the relative positions shift for this shell as a numpy array.
    """
    total_particles = sum(particles_per_shell)
    relative_positions = numpy.zeros(
        total_particles*3,
        dtype=float,
    ).reshape(total_particles, 3)
    ipos = 1

    for shell in range(1, len(shell_radii)):
        num_pts = particles_per_shell[shell]
        indices = numpy.arange(0, num_pts, dtype=float) + 0.5

        phi = arccos(1 - 2*indices/num_pts)
        theta = pi * (1 + 5**0.5) * indices

        ipos2 = ipos + num_pts
        x = shell_radii[shell] * cos(theta) * sin(phi)
        y = shell_radii[shell] * sin(theta) * sin(phi)
        z = shell_radii[shell] * cos(phi)

        relative_positions[ipos:ipos2, 0:3] = numpy.dstack((x, y, z))
        ipos = ipos2

    return relative_positions


def res_increase(
    gas=None,
    recalculate_h_density=False,
    seed=123,
    make_cutout=False,
    make_circular_cutout=False,
    circular_rmax=3000 | units.pc,
    x_center=None,
    y_center=None,
    width=None,
    res_increase_factor=85,
):
    numpy.random.seed(seed)
    if gas is None:
        if len(sys.argv) > 2:
            from amuse.io import read_set_from_file
            filename = sys.argv[1]
            res_increase_factor = int(sys.argv[2])
            gas = read_set_from_file(filename, 'amuse')
            if hasattr(gas, "itype"):
                gas = gas[gas.itype == 1]
                del gas.itype
        else:
            from amuse.ic.gasplummer import new_plummer_gas_model
            converter = nbody_system.nbody_to_si(
                10000 | units.MSun, 10 | units.pc
            )
            filename = "test"
            gas = new_plummer_gas_model(10000, converter)
            res_increase_factor = 85
            sph = Fi(converter, mode="openmp")
            gas_in_code = sph.gas_particles.add_particles(gas)
            gas.h_smooth = gas_in_code.h_smooth
            gas.density = gas_in_code.density
            sph.stop()
            write_set_to_file(gas, "old-%s" % filename, "amuse")
            print("old gas created")

    if make_circular_cutout:
        r2 = gas.x**2 + gas.y**2
        cutout = gas[r2 <= circular_rmax**2]
        gas = cutout
        converter = nbody_system.nbody_to_si(gas.total_mass(), width)
        sph = Fi(converter, mode="openmp")
        gas_in_code = sph.gas_particles.add_particles(gas)
        gas.h_smooth = gas_in_code.h_smooth
        gas.density = gas_in_code.density
        sph.stop()

    if make_cutout:
        if (
            x_center is None
            or y_center is None
            or width is None
        ):
            raise Exception("Need to set x_center, y_center and width!")
        cutout = gas.sorted_by_attribute("x")
        cutout = cutout[cutout.x - x_center < width/2]
        cutout = cutout[cutout.x - x_center > -width/2]
        cutout = cutout.sorted_by_attribute("y")
        cutout = cutout[cutout.y - y_center < width/2]
        cutout = cutout[cutout.y - y_center > -width/2]
        gas = cutout
        converter = nbody_system.nbody_to_si(gas.total_mass(), width)
        sph = Fi(converter, mode="openmp")
        gas_in_code = sph.gas_particles.add_particles(gas)
        gas.h_smooth = gas_in_code.h_smooth
        gas.density = gas_in_code.density
        sph.stop()
        # boundary = test_cutout.h_smooth.max()

    if res_increase_factor == 1:
        return gas

    original_number_of_particles = len(gas)
    new_number_of_particles = (
        res_increase_factor * original_number_of_particles
    )

    converter = nbody_system.nbody_to_si(
        gas.total_mass(),
        1 | units.kpc,
    )

    new_gas = Particles(new_number_of_particles)
    # new_gas.h_smooth = gas.h_smooth

    shells, particles_per_shell, shell_radii = find_shell_struct(
        res_increase_factor
    )

    relative_positions = pos_shift(
        shell_radii,
        particles_per_shell,
        res_increase_factor=res_increase_factor,
    )
    relative_velocities = numpy.zeros(
        res_increase_factor*3, dtype=float
    ).reshape(res_increase_factor, 3) | gas.velocity.unit

    random_samples = 50
    number_of_particles = len(gas)
    starting_index = 0
    for r in range(random_samples):
        print("%i / %i random sample done" % (r, random_samples))
        number_of_particles_remaining = len(gas)
        number_of_particles_in_sample = min(
            number_of_particles_remaining,
            int(1 + number_of_particles / random_samples)
        )

        gas_sample = gas.random_sample(number_of_particles_in_sample).copy()
        gas.remove_particles(gas_sample)
        end_index = (
            starting_index
            + number_of_particles_in_sample * res_increase_factor
        )
        new_gas_sample = new_gas[starting_index:end_index]
        psi = 2 * numpy.pi * numpy.random.random()
        theta = 2 * numpy.pi * numpy.random.random()
        phi = 2 * numpy.pi * numpy.random.random()
        relative_positions = rotated(relative_positions, phi, theta, psi)
        # print(len(gas_sample), len(new_gas_sample))
        for i in range(res_increase_factor):
            new_gas_sample[i::res_increase_factor].mass = (
                gas_sample.mass / res_increase_factor
            )
            new_gas_sample[i::res_increase_factor].x = (
                gas_sample.x
                + relative_positions[i, 0] * gas_sample.h_smooth
            )
            new_gas_sample[i::res_increase_factor].y = (
                gas_sample.y
                + relative_positions[i, 1] * gas_sample.h_smooth
            )
            new_gas_sample[i::res_increase_factor].z = (
                gas_sample.z
                + relative_positions[i, 2] * gas_sample.h_smooth
            )
            new_gas_sample[i::res_increase_factor].vx = (
                gas_sample.vx
                + relative_velocities[i, 0]
            )
            new_gas_sample[i::res_increase_factor].vy = (
                gas_sample.vy
                + relative_velocities[i, 1]
            )
            new_gas_sample[i::res_increase_factor].vz = (
                gas_sample.vz
                + relative_velocities[i, 2]
            )
            new_gas_sample[i::res_increase_factor].density = gas_sample.density
            new_gas_sample[i::res_increase_factor].u = gas_sample.u
        starting_index += number_of_particles_in_sample * res_increase_factor
    new_gas.h_smooth = (
        (3 * new_gas.mass / (4 * pi * new_gas.density))**(1/3)
     )

    # sph = Fi(converter, mode="openmp", redirection="none")
    # new_gas_in_code = sph.gas_particles.add_particles(new_gas)
    # new_gas.h_smooth = new_gas_in_code.h_smooth
    # new_gas.density = new_gas_in_code.density
    # sph.stop()

    print(
        "particles now have a mass of %s" % (new_gas[0].mass.in_(units.MSun))
    )
    return new_gas


if __name__ == "__main__":
    filename = sys.argv[1]
    new_gas = res_increase(
        recalculate_h_density=False,
        seed=123,
        make_cutout=True,
        x_center=-1800 | units.pc,
        y_center=-1800 | units.pc,
        width=500 | units.pc,
    )
    write_set_to_file(new_gas, "new-%s" % filename, "amuse")
