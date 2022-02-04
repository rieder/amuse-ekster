#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback class
"""
import logging

import numpy
import pandas
# from grispy import GriSPy
from sklearn.neighbors import KDTree
from numba import njit, typed, types

from amuse.units import units, constants
from amuse.units.trigo import sin, cos, arccos, arctan
from amuse.datamodel import Particle, Particles, ParticlesSuperset
from amuse.io import write_set_to_file

from plotting_class import (
    gas_mean_molecular_weight, temperature_to_u, u_to_temperature
)
import ekster_settings
settings = ekster_settings.Settings()


def stellar_feedback(gas_, stars):
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


@njit
def gas_stars_distances(
    gas_pos_pc,
    stars_pos_pc
):
    """
    Calculate the distance and unit vector between gas
    (indexed i) and stars (indexed a). Numba-jitted for
    speed.

    Parameters:
    gas_pos_pc: (Ngas, 3) numpy.ndarray
        Position of gas from the origin in pc
    stars_pos_pc: (Nstars, 3) numpy.ndarray
        Position of stars from the origin in pc

    Returns:
    i_a_dists_pc: (Nstars, Ngas) numpy.ndarray
        Distance of each gas-star pair in pc
    i_a_uvs: (Nstars, Ngas, 3) numpy.ndarray
        Unit vector from gas of each gas-star pair
    """
    Ngas = len(gas_pos_pc)
    Nstars = len(stars_pos_pc)
    i_a_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float64)
    i_a_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float64)

    for i in range(Ngas):
        for a in range(Nstars):
            i_a_vector = gas_pos_pc[i] - stars_pos_pc[a]
            i_a_dist = numpy.sqrt(i_a_vector.dot(i_a_vector))
            i_a_uv = i_a_vector / i_a_dist
            i_a_dists_pc[a, i] = i_a_dist
            i_a_uvs[a, i] = i_a_uv

    return i_a_dists_pc, i_a_uvs


@njit
def nearby_gas(
    i_a_dists_pc,
    dist_threshold_pc
):
    """
    Find the gas particles nearby to any star. Numba-jitted
    for speed.

    This function should work in theory as long as
    angle_threshold < pi/2 in generate_network().

    Parameters:
    i_a_dists_pc: (Nstars, Ngas) numpy.ndarray
        Distance of each gas-star pair in pc
    dist_threshold_pc (Nstars,): numpy.ndarray
        Threshold distance from the stars in pc

    Returns:
    nearby_gas_marker: (Ngas,) numpy.ndarray
        1 if the gas is nearby to any star; 0 otherwise.
    """
    Nstars, Ngas = i_a_dists_pc.shape
    nearby_gas_marker = numpy.zeros(Ngas, dtype=numpy.int16)
    for a in range(Nstars):
        indices = numpy.where(i_a_dists_pc[a] < dist_threshold_pc[a])[0]
        nearby_gas_marker[indices] = 1
    return nearby_gas_marker


def search_all_neighbours(
    bubble_dist,
    bubble_ind,
    gas_pos_pc,
    gas_h_pc,
    superset_pos_pc,
    nearby_gas_marker
):
    """
    Search the neighbours of the nearby gas using kd-tree.

    Parameters:
    bubble_dist: numba.typed.Dict
        Empty dictionary to be updated.
    bubble_ind: numba.typed.Dict
        Empty dictionary to be updated.
    gas_pos_pc: (Ngas, 3) numpy.ndarray
        Position of gas from the origin in pc
    gas_h_pc: (Ngas,) numpy.ndarray
        Smoothing length of gas in pc
    superset_pos_pc: (Ngas+Nstars, 3) numpy.ndarray
        Position of gas and stars from the origin in pc.
        Stars start from the index Ngas.
    nearby_gas_marker: (Ngas,) numpy.ndarray
        1 if the gas is nearby to any star; 0 otherwise.

    Returns:
    bubble_dist: numba.typed.Dict
        For each gas i with key i, the value is an array of
        neighbour distances from gas i.
    bubble_ind: numba.typed.Dict
        For each gas i with key i, the value is an array of
        neighbour indices of gas i.
    tree: sklearn.neighbors.KDTree object
        Training data to find neighbours.
    """
    # Find neighbours of nearby gas only
    indices = numpy.where(nearby_gas_marker == 1)[0]
    nearby_gas_pos_pc = gas_pos_pc[indices]
    nearby_gas_h_pc = gas_h_pc[indices]

    # Create kd-tree and find neighbours
    tree = KDTree(superset_pos_pc)
    bubble_ind_kdtree, bubble_dist_kdtree = tree.query_radius(
        nearby_gas_pos_pc,
        2*nearby_gas_h_pc,
        return_distance=True
    )

    # Copy to numba dictionaries
    Ngas = len(gas_pos_pc)
    nearby_indices = numpy.where(nearby_gas_marker == 1)[0]
    non_nearby_indices = numpy.where(nearby_gas_marker == 0)[0]
    d_dist = dict(zip(nearby_indices, bubble_dist_kdtree))
    d_dist.update(dict.fromkeys(non_nearby_indices, numpy.array([])))
    d_ind = dict(zip(nearby_indices, bubble_ind_kdtree))
    d_ind.update(dict.fromkeys(non_nearby_indices, numpy.array([])))
    for i in range(Ngas):
        bubble_dist[i] = d_dist[i].astype(numpy.float64)
        bubble_ind[i] = d_ind[i].astype(numpy.int64)

    return bubble_dist, bubble_ind, tree


def search_neighbours_i(
    tree,
    gas_pos_pc_i,
    search_radius,
):
    """
    Search the neighbours of gas i.

    Parameters:
    tree: sklearn.neighbors.KDTree object
        Training data to find neighbours.
    gas_pos_pc_i: (3,) numpy.ndarray
        Position of gas i from the origin in pc.
    search_radius: scalar
        Distance from gas i to perform neighbour search.

    Returns:
    dist_i: numpy.ndarray
        Neighbour distances from gas i.
    ind_i: numpy.ndarray
        Neighbour indices of gas i.
    """

    ind_i, dist_i = tree.query_radius(
        [gas_pos_pc_i], search_radius, return_distance=True
    )
    ind_i = ind_i[0]
    dist_i = dist_i[0]
    return dist_i, ind_i


@njit
def search_predecessors(
    i,
    network_a,
    max_count=2**8
):
    """
    Breadth-first-search method to find predecessors of gas
    i. This function is used to check for cycles in the
    network. In theory, it is not needed if angle_threshold
    < pi/3 in generate_network().

    Parameters:
    i: int
        Index of the gas to find its predecessors.
    network_a: numpy.ndarray
        a-th row of the network matrix in
        generate_network().
    max_count: int, default = 2**8
        Maximum count to perform the search.

    Returns:
    pred: numpy.ndarray
        Index of predecessors of gas i.
    """
    pred = numpy.argwhere(network_a==i).flatten()
    queue = pred.copy()
    while queue.size > 0 and max_count > 0:
        s = queue[0]
        queue = numpy.delete(queue, 0)
        new_pred = numpy.argwhere(network_a==s).flatten()
        for k in new_pred:
            if k not in pred:
                pred = numpy.append(pred, k)
                queue = numpy.append(queue, k)
        max_count -= 1
    return pred


def generate_network(
    gas_pos_pc,
    gas_h_pc,
    stars_pos_pc,
    superset_pos_pc,
    bubble_dist,
    bubble_ind,
    tree,
    i_a_dists_pc,
    i_a_uvs,
    nearby_gas_marker,
    angle_threshold=numpy.pi/3,
):
    """
    gas_pos_pc: (Ngas, 3)
    stars_pos_pc: (Nstars, 3)
    gas_h_pc: (Ngas)
    """
    Ngas = len(gas_pos_pc)
    Nstars = len(stars_pos_pc)
    cos_angle_threshold = numpy.cos(angle_threshold)

    # Arrays to be returned
    network = numpy.full((Nstars, Ngas), -1, numpy.int64)
    # i_a_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float64)
    # i_a_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float64)
    link_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float64)
    link_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float64)

    # Initialise for numba
    temp_bubble_dist_i = numpy.zeros(1, numpy.float64)
    temp_bubble_ind_i = numpy.zeros(1, numpy.int64)
    predecessors_ind = numpy.zeros(1, numpy.int64)

    counter = i = failsafe = 0
    temporary = False
    cycle = False
    Nloops = Ngas * Nstars
    count_real_neigh = count_cycle_neigh = count_angle = count_cycle = 0
    while counter < Nloops and failsafe < (2*Nloops):

        if counter % 100000 == 0:
            progress = counter / Nloops
            print('Progress:', progress*100, '%')

        # If the gas is not nearby any star, skip to next gas
        if nearby_gas_marker[i] == 0:
            i += 1
            counter = Nstars * i
            failsafe = Nstars * i
            continue

        a = 0
        target_position = gas_pos_pc[i]
        search_radius = 2 * gas_h_pc[i]
        new_search_radius = search_radius
        while a < Nstars:
            # print(i, a, counter)
            star_position = stars_pos_pc[a]
            i_a_vector = target_position - star_position
            i_a_dist = i_a_dists_pc[a, i]
            i_a_uv = i_a_uvs[a, i]
            # i_a_dist = numpy.sqrt(i_a_vector.dot(i_a_vector))
            # i_a_uv = i_a_vector / i_a_dist

            # Save and to be returned for feedback calculation
            # i_a_dists_pc[a, i] = i_a_dist
            # i_a_uvs[a, i] = i_a_uv

            # If there is temporary increase in neighbour list
            if temporary:
                bubble_dist_i = temp_bubble_dist_i
                bubble_ind_i = temp_bubble_ind_i
            else:
                bubble_dist_i = bubble_dist[i]
                bubble_ind_i = bubble_ind[i]

            # if gas i is nearby star a, link to star a
            if (a + Ngas) in bubble_ind_i:
                network[a, i] = a + Ngas
                link_dists_pc[a, i] = i_a_dist
                link_uvs[a, i] = i_a_uv

            # else, compute and link to the next gas
            else:
                # Obtain true neighbours of gas i
                conditions = (bubble_ind_i < Ngas) & (bubble_dist_i > 0.0)
                neighbour_ind = bubble_ind_i[conditions]
                i_neighbour_dist = bubble_dist_i[conditions]

                # Remove predecessors from neighbour list if any
                if cycle:
                    non_int_neighbour_ind = numpy.array(
                        list(
                            (set(predecessors_ind) | set(neighbour_ind))
                            - set(predecessors_ind)
                        )
                    )
                    non_int_neighbour_ind_ind = numpy.array([
                        numpy.argwhere(neighbour_ind == l)[0][0]
                        for l in non_int_neighbour_ind
                    ])
                    neighbour_ind = non_int_neighbour_ind
                    i_neighbour_dist = i_neighbour_dist[
                        non_int_neighbour_ind_ind
                    ]

                # Extend neighbour list if empty
                if neighbour_ind.shape[0] == 0:
                    new_search_radius += search_radius
                    new_bubble_dist_i, new_bubble_ind_i = search_neighbours_i(
                        tree,
                        target_position,
                        new_search_radius
                    )
                    if cycle:
                        # print(
                        #     a, i, "No neighbour after removing predecessors"
                        # )
                        temp_bubble_dist_i = new_bubble_dist_i
                        temp_bubble_ind_i = new_bubble_ind_i
                        temporary = True
                        count_cycle_neigh += 1
                    else:
                        # print(a, i, "No real neighbour")
                        bubble_dist[i] = new_bubble_dist_i
                        bubble_ind[i] = new_bubble_ind_i
                        count_real_neigh += 1
                    continue

                # Calculate min angle (max cos angle) from i-a line of sight
                i_neighbour_vector = (
                    target_position - gas_pos_pc[neighbour_ind]
                )
                # i_neighbour_uv = [
                #     i_neighbour_vector[k]/i_neighbour_dist[k]
                #     for k in range(len(neighbour_ind))
                # ]
                i_neighbour_uv = numpy.einsum(
                    "ij,i->ij", i_neighbour_vector, 1/i_neighbour_dist
                )
                # angle_to_LOS = numpy.array([
                #     numpy.arccos(i_a_uv.dot(uvi)) for uvi in i_neighbour_uv
                # ])
                # min_angle = numpy.min(angle_to_LOS)
                cos_angle_to_LOS = numpy.einsum(
                    "i,ji->j", i_a_uv, i_neighbour_uv
                )
                max_cos_angle = numpy.max(cos_angle_to_LOS)
                if max_cos_angle < cos_angle_threshold:


                # if min_angle > angle_threshold:
                    # print(
                    #     a, i, "Min angle", min_angle, ">", angle_threshold
                    # )
                    new_search_radius += search_radius
                    new_bubble_dist_i, new_bubble_ind_i = search_neighbours_i(
                        tree,
                        target_position,
                        new_search_radius
                    )
                    temp_bubble_dist_i = new_bubble_dist_i
                    temp_bubble_ind_i = new_bubble_ind_i
                    temporary = True
                    count_angle += 1
                    continue

                selected_neighbour_ind_ind = numpy.where(
                    cos_angle_to_LOS == max_cos_angle
                )[0]
                selected_neighbour_ind = neighbour_ind[
                    selected_neighbour_ind_ind
                ][0]

                # Check if selected neighbour creates a
                # cycle
                # # NOTE: IT IS (I THINK) MATHEMATICAL IMPOSSIBLE
                # # CYCLE WILL BE CREATED IF THRESHOLD ANGLE < pi/3
                # # Commented to speed things up
                # predecessors_ind = search_predecessors(
                #     i,
                #     network[a]
                # )
                # if selected_neighbour_ind in predecessors_ind:
                #     # print(
                #     #     a, i, 'Cycle with', selected_neighbour_ind
                #     # )
                #     cycle = True
                #     count_cycle += 1
                #     continue

                network[a, i] = selected_neighbour_ind
                link_dists_pc[a, i] = i_neighbour_dist[
                    selected_neighbour_ind_ind
                ][0]
                link_uvs[a, i] = i_neighbour_uv[
                    selected_neighbour_ind_ind[0]
                ]

            cycle = False
            temporary = False
            a += 1
            counter = Nstars*i + a
            failsafe = Nstars*i + a

        # Add gas index
        i += 1

    return (
        network, link_dists_pc, link_uvs,
        [count_real_neigh, count_cycle_neigh, count_angle, count_cycle]
    )


@njit
def create_path_dict_numba(numba_path_dict, network, Ngas, Nstars, nearby_gas_marker):
    for i in range(Ngas):
        if nearby_gas_marker[i] == 0:
            continue

        for a in range(Nstars):
            key = (a, i)
            path = numpy.array([i], numpy.int64)
            while i < Ngas:
                j = network[a, i]
                path = numpy.append(path, j)
                i = j
            numba_path_dict[key] = path
    return numba_path_dict


@njit
def one_feedback_iteration(
    numba_path_dict,
    i_a_dists_pc,
    i_a_uvs,
    link_dists_pc,
    link_uvs,
    i_a_density_gcm,
    photon_flux,
    Nsources,
    nearby_gas_marker,
    molecular_mass=1.26,
    recombination_coefficient=3e-13,
):
    """
    Calculate one iteration of ionising feedback (Dale et
    al. 2007, Dale and Bonnell 2011) using numba.

    Parameters:
    numba_path_dict: numba.typed.Dict
        tuple(gas i, star a): numpy.array of nodes in
        between i and a inclusive
    i_a_dists_pc: (Nstars, Ngas) numpy.ndarray
        Distance of each gas-star pair in pc
    i_a_uvs: (Nstars, Ngas, 3) numpy.ndarray
        Unit vector from gas of each gas-star pair
    link_dists_pc: (Nstars, Ngas) numpy.ndarray
        Distance of each gas to next linked gas/star in pc
    link_uvs: (Nstars, Ngas, 3) numpy.ndarray
        Unit vector from gas to next linked gas/star
    i_a_density_gcm: (Ngas+Nstars) numpy.ndarray
        Density of gas and stars in g/cm**3. Density of
        stars is 0 by construction
    photon_flux: (Nstars) numpy.array
        Photon flux by stars in s**-1
    Nsources: (Ngas) numpy.array
        Number of sources that ionised each gas
    molecular_mass: float
        Molecular mass in a.m.u.. Default is 1.26
    recombination_coefficient: float
        Recombination coefficient in cm**3 s**-1. Default
        is 3e-13

    Returns:
    f_ion_array: (Ngas) numpy.array
        Ionised fraction of the gas
    new_Nsources: (Ngas) numpy.array
        New number of sources that ionised each gas
    """
    Nstars, Ngas = i_a_dists_pc.shape
    molecular_mass *= 1.67355e-24   # Convert from a.m.u. to grams
    pc_to_cm = 3.086e18

    fluxes = numpy.zeros((Nstars, Ngas))
    f_ion_array = numpy.zeros(Ngas)

    # To avoid division by 0
    refined_Nsources = Nsources.copy()
    refined_Nsources[refined_Nsources == 0] = 1

    for i in range(Ngas):
        # Skip non-closeby gas
        if nearby_gas_marker[i] == 0:
            continue

        for a in range(Nstars):
            # Find all gas between gas i and star a
            gas_star_indices = numba_path_dict[(a, i)]
            gas_indices = gas_star_indices[:-1]

            # Calculate evaluation bin widths (delta r_i)
            i_a_dist = i_a_dists_pc[a, i]
            # i_a_uv = i_a_uvs[:, a].transpose()[i]
            i_a_uv = i_a_uvs[a, i]
            link_dist = link_dists_pc[a][gas_indices]
            # link_uv = link_uvs[:, a].transpose()[gas_indices]
            link_uv = link_uvs[a, i]
            dot_prod = link_uv.dot(i_a_uv)
            eval_bin_widths = link_dist * dot_prod

            # Calculate average evaluation density
            eval_dens = i_a_density_gcm[gas_star_indices]
            average_eval_dens = [
                (eval_dens[i]+eval_dens[i+1]) / (2*molecular_mass)
                for i in range(len(eval_dens) - 1)
            ]
            average_eval_dens = numpy.array(average_eval_dens)

            # Calculate evaluation radii
            eval_radii = numpy.cumsum(eval_bin_widths[::-1])[::-1]

            # Find evaluation number of sources (Dale+11)
            eval_Nsources = refined_Nsources[gas_indices]

            # Calculate flux on gas i
            to_be_summed = (
                eval_radii**2
                * average_eval_dens**2
                * eval_bin_widths
                / eval_Nsources
            )
            luminosity_integral = (
                4 * numpy.pi * recombination_coefficient * numpy.sum(
                    to_be_summed
                )
            ) * pc_to_cm**3
            flux = (
                (photon_flux[a] - luminosity_integral)
                / (4 * numpy.pi * i_a_dist**2)
            ) * pc_to_cm**-2

            # If flux is positive, calculate ionised fraction
            if flux > 0:
                fluxes[a, i] = flux

                # Find the flux without gas i
                lum_int_last = (
                    4 * numpy.pi * recombination_coefficient * to_be_summed[-1]
                ) * pc_to_cm**3
                lum_int_wo_target = luminosity_integral - lum_int_last
                if len(gas_indices) == 1:
                    dist = eval_radii[-1] / 2
                else:
                    dist = eval_radii[-2]
                flux_wo_target = (
                    (photon_flux[a] - lum_int_wo_target)
                    / (4 * numpy.pi * dist**2)
                ) * pc_to_cm**-2
                f_ion = flux_wo_target / flux
                f_ion_array[i] = min(1.0, f_ion_array[i] + f_ion)
                # print(f_ion)

    new_Nsources = [numpy.count_nonzero(fluxes[:, i]) for i in range(Ngas)]
    new_Nsources = numpy.array(new_Nsources)

    return f_ion_array, new_Nsources, fluxes


def main_stellar_feedback(
    gas,
    stars_,
    time,
    mass_cutoff=5|units.MSun,
    angle_threshold=numpy.pi/3,
    recombination_coefficient=3e-13|units.cm**3/units.s,
    temp_range=[10,10000]|units.K,
    logger=None,
    **keyword_arguments
):
    """
    Main stellar feedback scheme.
    """
    logger = logger or logging.getLogger(__name__)

    logger.info("Most massive star %s", stars_.mass.value_in(units.MSun).max())
    # Choose only massive stars
    stars = stars_.select_array(
        lambda x: x >= mass_cutoff,
        ["mass"]
    )
    logger.info(
        "%i massive stars > %s",
        len(stars), mass_cutoff
    )

    if stars.is_empty():
        logger.info("No massive stars, so no feedback")
        return gas

    # ONLY FOR STARBENCH TEST: FIX PHOTON FLUX
    stars.luminosity = (1e49 * 13.6|units.eV * units.s**-1)


    # This has to be 1 amu
    gmmw = 1 | units.amu
    # gmmw = gas_mean_molecular_weight(0.5)
    max_u = gas.u.max().in_(units.cm**2 / units.s**2)
    logger.info(
        "Max gas internal energy: %s",
        max_u
    )

    logger.info(
        "Max gas temperature: %s",
        u_to_temperature(max_u, gmmw=gmmw).in_(units.K)
    )

    min_u = gas.u.min()
    logger.info(
        "Min gas temperature: %s",
        u_to_temperature(min_u, gmmw=gmmw).in_(units.K)
    )

    threshold_u = temperature_to_u(0.999*temp_range[1], gmmw=gmmw)
    if max_u >= threshold_u:
        hot_gas = gas.select_array(
            lambda x: x >= threshold_u, ['u']
        )
        logger.info("%i hot gas", len(hot_gas))

    Ngas = len(gas)
    Nstars = len(stars)
    superset = ParticlesSuperset([gas, stars])

    # Numpy arrays for numba jitted-functions
    gas_pos_pc = gas.position.value_in(units.pc)
    gas_h_pc = gas.h_smooth.value_in(units.pc)
    stars_pos_pc = stars.position.value_in(units.pc)
    superset_pos_pc = superset.position.value_in(units.pc)

    # Store neighbour distances and indices
    bubble_dist = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:]
    )
    bubble_ind = typed.Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )

    print('Calculating distances and selecting nearby gas...')
    logger.info('Calculating distances and selecting nearby gas...')
    # Calculate distance and unit vector between all gas
    # to all star
    i_a_dists_pc, i_a_uvs = gas_stars_distances(
        gas_pos_pc,
        stars_pos_pc
    )

    # Use Stroemgren radii, Hosokawa-Inutsuka approximation
    # and Raga extension to estimate search radii
    initial_rho = 5e-21 | units.g * units.cm**-3    #5e-21
    initial_n = initial_rho / gmmw
    recombination_coefficient = 2.7e-13 | units.cm**3 * units.s**-1
    photon_flux = (stars.luminosity/(13.6|units.eV))
    stroemgren_radii = (
        (3*photon_flux) / (4*numpy.pi*initial_n**2*recombination_coefficient)
    )**(1.0/3)
    c_sounds = numpy.sqrt(constants.kB * temp_range / gmmw)
    hosokawa_inutsuka = (
        stroemgren_radii * (
            1 + (7/4 * numpy.sqrt(4/3) * (c_sounds[1]*time)/stroemgren_radii)
        )
    ).value_in(units.pc)
    stagnation_radii = (
        (c_sounds[1]/c_sounds[0])**(4/3) * stroemgren_radii
    ).value_in(units.pc)
    dist_threshold_pc = 2 * numpy.array([
        min(hosokawa_inutsuka[a], stagnation_radii[a]) for a in range(Nstars)
    ])
    dist_threshold_pc_average = numpy.mean(dist_threshold_pc)
    # dist_threshold_pc = 2 * stroemgren_radii.value_in(units.pc)
    # dist_threshold_pc_average = numpy.mean(dist_threshold_pc)
    print(f'Average threshold distance: {dist_threshold_pc_average} pc')
    logger.info(
        "Average threshold distance: %s pc", dist_threshold_pc_average
    )

    # Find gas which are nearby to any star
    nearby_gas_marker = nearby_gas(
        i_a_dists_pc,
        dist_threshold_pc
    )

    Ngas_nearby = len(numpy.where(nearby_gas_marker == 1)[0])
    print(f"{Ngas_nearby} nearby gas (out of {Ngas})")
    logger.info("%i nearby gas (out of %i)", Ngas_nearby, Ngas)

    print('Seaching neighbours...')
    logger.info('Searching neighbours...')
    bubble_dist, bubble_ind, tree = search_all_neighbours(
        bubble_dist,
        bubble_ind,
        gas_pos_pc,
        gas_h_pc,
        superset_pos_pc,
        nearby_gas_marker,
    )

    # WARNING! Two redundant arrays below
    print('Generating network...')
    logger.info('Generating network...')
    network, link_dists_pc, link_uvs, stat = (
        generate_network(
            gas_pos_pc,
            gas_h_pc,
            stars_pos_pc,
            superset_pos_pc,
            bubble_dist,
            bubble_ind,
            tree,
            i_a_dists_pc,
            i_a_uvs,
            nearby_gas_marker,
        )
    )
    logger.info(
        "Loop stat: %i real_neigh, %i cycle_neigh, %i angle, %i cycle",
        stat[0], stat[1], stat[2], stat[3]
    )

    print("Creating path dictionary...")
    logger.info("Creating path dictionary...")
    numba_path_dict = typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.int64[:]
    )
    numba_path_dict = create_path_dict_numba(
        numba_path_dict, network, Ngas, Nstars, nearby_gas_marker
    )

    print("Calculating feedback...")
    logger.info("Calculating feedback...")
    photon_flux = (stars.luminosity/(13.6|units.eV)).value_in(units.s**-1)
    molecular_mass = gmmw / (1 | units.amu)
    recombination_coefficient = recombination_coefficient.value_in(
        units.cm**3 * units.s**-1
    )
    i_a_density_gcm = numpy.zeros(Ngas+Nstars, numpy.float64)
    i_a_density_gcm[:Ngas] = gas.density.value_in(units.g/units.cm**3)

    Nsources = numpy.zeros(Ngas)
    Nionised_gas = 0
    error = 10000
    delta_Nionised_gas = 10000
    count = 0
    while error > 0.01 and delta_Nionised_gas > 2:
        f_ion_array, new_Nsources, fluxes = one_feedback_iteration(
            numba_path_dict,
            i_a_dists_pc,
            i_a_uvs,
            link_dists_pc,
            link_uvs,
            i_a_density_gcm,
            photon_flux,
            Nsources,
            nearby_gas_marker,
            molecular_mass=molecular_mass,
            recombination_coefficient=recombination_coefficient,
        )
        new_Nionised_gas = len(new_Nsources[new_Nsources > 0])
        print(f'No. of ionised gas: {Nionised_gas} --> {new_Nionised_gas}')
        logger.info(
            "No. of ionised gas: %i --> %i",
            Nionised_gas, new_Nionised_gas
        )

        # Check for error
        delta_Nionised_gas = new_Nionised_gas - Nionised_gas
        if Nionised_gas > 0: error = delta_Nionised_gas / Nionised_gas
        Nsources = new_Nsources.copy()
        Nionised_gas = new_Nionised_gas
        print(f'Count {count}, error = {error}, delta_Nionised_gas = {delta_Nionised_gas}')
        logger.info(
            "Count %i, error %s, delta_Nionised_gas %i",
            count, error, delta_Nionised_gas
        )
        count += 1

    print("Populating temperatures...")
    logger.info("Populating temperatures...")
    temperatures = (
        f_ion_array * (temp_range[1] - temp_range[0])
        + temp_range[0]
    )
    internal_energies = temperature_to_u(
        temperatures, gmmw=gmmw
    )
    gas.u = internal_energies
    gas.h2ratio = 0
    gas.proton_abundance = f_ion_array

    return gas
