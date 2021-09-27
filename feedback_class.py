#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback class
"""

import numpy
import pandas

from amuse.units import units, constants
from amuse.units.trigo import sin, cos, arccos, arctan
from amuse.datamodel import Particle, Particles


def test_stellar_feedback(gas_, stars):
    # Extremely simplified stellar feedback onto gas
    gas = gas_.copy()
    mass_cutoff = 5 | units.MSun
    rmax = 0.05 | units.pc
    massive_stars = stars.select(
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
        # (1 | units.AU) / self.feedback_timestep
        base_velocity = (20 | units.kms)
        gas_near_star.vx += sin(theta) * cos(phi) * base_velocity
        gas_near_star.vy += sin(theta) * sin(phi) * base_velocity
        gas_near_star.vz += cos(theta) * base_velocity

        gas.position += star.position
    return gas

def basic_stroemgren_volume_method(
    gas_,
    stars,
    mass_cutoff=5|units.MSun,
    gas_mean_molecular_weight=2.381,
    recombination_coefficient=3e-13|units.cm**3/units.s,
    cutoff_ratio=0.01,
    ):
    """
    Basic Stroemgren volume method as described in Dale et al (2007).
    Update the gas temperature for those gas particles that receive
    positive flux from the stars.
    """
    gas_particles = gas_.copy()
    massive_stars = stars.copy().select_array(
        lambda m:
        m >= mass_cutoff,
        ["mass"]
    )
    # number of iteration after latest positive flux
    Nmax = min(200, int(cutoff_ratio*len(gas_particles)))


    # average_density = numpy.average(
    #     gas_particles.density.value_in(units.g/units.cm**3)
    # ) | units.g/units.cm**3
    # alpha = 3e-13 |units.cm**3/units.s
    # for star in massive_stars:
    #     # average_density = 1.47e-19 | units.g/units.cm**3
    #     average_density /= (2.3 * 1.67e-27|units.kg)
    #     QH = star.luminosity / (13.6|units.eV)
    #     rst = ((3 * QH)/(4*numpy.pi*average_density**2*alpha))**(1.0/3)
    #     print(rst.in_(units.pc))
    #
    #     exit()

    flux_list = []
    distance_origin_list = []
    distance_star_list = []
    DF = pandas.DataFrame()

    for star in massive_stars:
        print(f'DEBUG: Star is #{star.key} with mass {star.mass.in_(units.MSun)}')
        gas_particles.distance_to_star = (
            gas_particles.position - star.position
        ).lengths()

        # Remove last. This is part of the test.
        # gas_particles = gas_particles.select_array(
        #     lambda x: x < 0.1|units.pc, ['distance_to_star']
        # )

        gas_particles = gas_particles.sorted_by_attribute('distance_to_star')
        gas_particles.is_linked = False
        gas_particles.is_dead_end = False

        loop_counter = 0
        break_neighbour_counter = break_angle_counter = break_dead_counter = success_counter = 0

        for target in gas_particles:
            print(f'DEBUG: Target is #{target.key}, loop counter = {loop_counter}')

            if loop_counter == Nmax:
                print(f'{Nmax} loops after last +ve successful flux! Break' )
                break

            main_LOS_vector = target.position - star.position
            main_LOS_distance = main_LOS_vector.length()
            main_LOS_unit_vector = main_LOS_vector / main_LOS_distance

            gasset = gas_particles.copy()
            gasi = target.copy()  # 'i' starts from target, unlike Dale2007
            reached_star = False
            eval_dens = numpy.array([gasi.density])
            eval_bin_widths = numpy.array([])
            LOS_gas_keys = numpy.array([])

            while not reached_star:
                gasi_to_star_vector = gasi.position - star.position
                gasi_to_star_distance = gasi_to_star_vector.length()
                gasi_to_star_unit_vector = (
                    gasi_to_star_vector / gasi_to_star_distance
                )

                # if gas_i is next to the star
                if gasi_to_star_distance < gasi.h_smooth:
                    eval_bin_width = (
                        gasi_to_star_distance
                        * gasi_to_star_unit_vector.dot(main_LOS_unit_vector)
                    )
                    eval_bin_widths = numpy.append(
                        eval_bin_width, eval_bin_widths
                    )
                    point_mass_dens = 0 | units.g/units.cm**3
                    eval_dens = numpy.append(point_mass_dens, eval_dens)

                    reached_star = True

                    gas_particles[
                        gas_particles.key == gasi.key
                    ].is_linked = True
                    gas_particles[
                        gas_particles.key == gasi.key
                    ].is_linked_to = star.key
                    print(f'DEBUG: Last gas #{gasi.key} linked to star')

                else:
                    LOS_gas_keys = numpy.append(gasi.key, LOS_gas_keys)

                    # if gas_i is linked to gas_j, choose gas_j as neighbour
                    if gasi.is_linked:
                        selected_neighbour = gasset.copy()[
                            gasset.key == gasi.is_linked_to
                        ][0]
                        selected_distance = (
                            gasi.position - selected_neighbour.position
                        ).length()
                        selected_gasi_to_gasset_unit_vector = (
                            gasi.position - selected_neighbour.position
                        ) / selected_distance

                        print(f'DEBUG: #{gasi.key} autolinked to #{selected_neighbour.key}')

                    elif gasi.is_dead_end:
                        break_dead_counter += 1
                        loop_counter += 1
                        print(f'DEBUG: #{gasi.key} is a dead end. Break!')
                        break

                    # if gas_i is not linked to other gas or is a dead end,
                    # then find a neighbour for gas_i
                    else:
                        # Find neighbours of gasi
                        gasset.gasi_to_gasset_vector_x = gasi.x - gasset.x
                        gasset.gasi_to_gasset_vector_y = gasi.y - gasset.y
                        gasset.gasi_to_gasset_vector_z = gasi.z - gasset.z
                        gasset.gasi_to_gasset_distance = (
                            gasi.position - gasset.position
                        ).lengths()
                        neighbours = gasset.copy().select_array(
                            lambda x: x < gasi.h_smooth,
                            ['gasi_to_gasset_distance']
                        )

                        # Remove current and previous gas
                        to_be_ignored_indices = numpy.where(
                            [True if k in LOS_gas_keys else False
                            for k in neighbours.key]
                        )[0]
                        to_be_ignored = neighbours.copy()[to_be_ignored_indices]
                        neighbours.remove_particles(to_be_ignored)
                        # neighbours.remove_particle(gasi.copy())

                        if neighbours.is_empty():
                            print(f'DEBUG: No neighbours, cannot find path on #{gasi.key}! Break')
                            break_neighbour_counter += 1
                            loop_counter += 1
                            gas_particles[
                                gas_particles.key == gasi.key
                            ].is_dead_end = True
                            break

                        # Find the gas particle among neighbours that has smallest
                        # angle to the current line-of-sight (LOS)


                        ux = neighbours.gasi_to_gasset_vector_x.value_in(units.m)
                        uy = neighbours.gasi_to_gasset_vector_y.value_in(units.m)
                        uz = neighbours.gasi_to_gasset_vector_z.value_in(units.m)
                        umag = neighbours.gasi_to_gasset_distance.value_in(units.m)

                        gasi_to_gasset_unit_vector = [
                            numpy.array([ux[i], uy[i], uz[i]])/umag[i]
                            for i in range(len(neighbours))
                        ]
                        neighbours.angle_to_current_LOS = [
                            numpy.arccos(gasi_to_star_unit_vector.dot(vi))
                            for vi in gasi_to_gasset_unit_vector
                        ]
                        min_angle = neighbours.angle_to_current_LOS.min()
                        if min_angle > numpy.pi/2:
                            print(f'DEBUG: Min angle {min_angle} > 1.571! Break at #{gasi.key}')
                            break_angle_counter += 1
                            loop_counter += 1
                            gas_particles[
                                gas_particles.key == gasi.key
                            ].is_dead_end = True
                            break

                        selected_neighbour = neighbours.copy()[
                            neighbours.angle_to_current_LOS == min_angle
                        ][0]
                        selected_index = numpy.where(
                            neighbours.angle_to_current_LOS == min_angle
                        )[0][0]
                        selected_distance = (
                            selected_neighbour.gasi_to_gasset_distance
                        )
                        selected_gasi_to_gasset_unit_vector = (
                            gasi_to_gasset_unit_vector[selected_index]
                        )

                        # Link
                        gas_particles[
                            gas_particles.key == gasi.key
                        ].is_linked = True
                        gas_particles[
                            gas_particles.key == gasi.key
                        ].is_linked_to = selected_neighbour.key

                        # target.is_linked = True
                        # target.is_linked_to = selected_neighbour.key
                        print(f'DEBUG: First calculating #{gasi.key}, link to #{selected_neighbour.key}' )


                    # Calculate evaluation bin width
                    eval_bin_width = (
                        selected_distance
                        * selected_gasi_to_gasset_unit_vector.dot(
                            main_LOS_unit_vector
                        )
                    )
                    eval_bin_widths = numpy.append(
                        eval_bin_width, eval_bin_widths
                    )
                    eval_dens = numpy.append(
                        selected_neighbour.density, eval_dens
                    )
                    print(f'DEBUG: Done #{gasi.key}')

                    # Now, neighbour is gasi and the process repeats
                    gasi = selected_neighbour.copy()

            if reached_star:
                # Calculate lumninosity integral
                eval_radii = numpy.cumsum(eval_bin_widths)
                molecular_mass = (
                    gas_mean_molecular_weight * (1.6735575e-27 | units.kg)
                )
                average_eval_dens = [
                    (eval_dens[i]+eval_dens[i+1]) / (2*molecular_mass)
                    for i in range(len(eval_dens) - 1)
                ]
                average_eval_dens = numpy.asarray(average_eval_dens)
                luminosity_integral = (
                    4 * numpy.pi * recombination_coefficient * numpy.sum(
                        eval_radii**2
                        * average_eval_dens**2
                        * eval_bin_widths
                    )
                )

                print('N_LOS =', len(average_eval_dens))
                success_counter += 1
                # Finally calculate flux on target
                flux = (
                    (star.luminosity/(13.6|units.eV) - luminosity_integral)
                    / (4 * numpy.pi * main_LOS_distance**2)
                )

                print(flux)
                flux_list.append(flux.value_in(units.s**-1 * units.m**-2))
                distance_origin_list.append(target.position.length().value_in(units.pc))
                distance_star_list.append(numpy.max(eval_radii).value_in(units.pc))

                if flux > 0 | units.s**-1 * units.m**-2:
                    # If flux is positive, change temperature to 10000K. Ignore
                    # flux direction for now
                    # Do stuff here
                    # target.temperature = 10000 | units.K
                    # target.sync_back_to_original_set_or_something
                    loop_counter = 0

                else:
                    loop_counter += 1
                    # if loop_counter == Nmax:
                    #     print(f'{Nmax} loops after last +ve successful flux! Break' )
                    #     break


        print(f'DEBUG: still in star #{star.key} loop')

        print(f'DEBUG: break neighbour {break_neighbour_counter}')
        print(f'DEBUG: break angle {break_angle_counter}')
        print(f'DEBUG: break dead end {break_dead_counter}')
        print(f'DEBUG: success {success_counter}')

    DF['flux'] = flux_list
    DF['distance_origin'] = distance_origin_list
    DF['distance_star'] = distance_star_list
    DF.to_csv('checkflux.csv')



    exit()
    return None
