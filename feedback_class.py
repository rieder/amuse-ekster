#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback class
"""
import logging

import numpy
import pandas
# import igraph
import networkit
from grispy import GriSPy
from numba import njit, typed, types

from amuse.units import units, constants
from amuse.units.trigo import sin, cos, arccos, arctan
from amuse.datamodel import Particle, Particles, ParticlesSuperset

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


def generate_network(
    gas,
    stars,
    angle_threshold=numpy.pi/3,
    logger=None,
):
    """
    Create network that links the gas to the sources. Refer
    to Figure 1 of Dale et al. (2007)

    Parameters:
    gas: AMUSE Particles
        Gas particle set
    stars: AMUSE Particles
        Ionising sources (massive stars) particle set
    angle_threshold: float
        Maximum angle to line of sight. Default is pi/3

    Returns:
    graph_dict: dictionary
        Star index (element of range(Nstars-1)):
        igraph.Graph network
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
    """
    logger = logger or logging.getLogger(__name__)

    Ngas = len(gas)
    Nstars = len(stars)
    superset = ParticlesSuperset([gas, stars])
    superset_ind_dict = dict(zip(superset.key, range(Ngas+Nstars)))
    ind_superset_dict = dict(zip(range(Ngas+Nstars), superset.key))

    # Search neighbours
    print("Searching for neighbours...")
    logger.info("Searching for neighbours...")
    gsp = GriSPy(superset.position.value_in(units.pc))
    bubble_dist, bubble_ind = gsp.bubble_neighbors(
        gas.position.value_in(units.pc),
        distance_upper_bound=2*gas.h_smooth.value_in(units.pc)
    )
    print("Neighbour search done.")
    logger.info("Neighbour search done")

    # Dictionary to contain Nstars of graphs
    graph_dict = {}
    # empty_graph = igraph.Graph(directed=True)
    # empty_graph.add_vertices(range(Ngas+Nstars))
    # for i in range(Nstars):
    #     graph_dict[i] = empty_graph.copy()

    for i in range(Nstars):
        graph = networkit.Graph(Ngas+Nstars, directed=True)
        graph_dict[i] = graph


    # Arrays to be returned
    i_a_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float)
    i_a_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float)
    link_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float)
    link_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float)
    i_a_density_gcm = numpy.zeros(Ngas+Nstars, numpy.float)
    i_a_density_gcm[:Ngas] = gas.density.value_in(units.g/units.cm**3)

    print("Generating network...")
    logger.info("Generating network...")
    counter = i = 0
    temporary = False
    predecessors_ind = []
    Nloops = Ngas * Nstars
    while counter < Nloops:
        a = 0
        target = gas[i]
        target_position = target.position
        search_radius = 2 * target.h_smooth.value_in(units.pc)
        new_search_radius = search_radius
        while a < Nstars:
            # print(i, a, counter)
            star = stars[a]
            i_a_vector = (target_position - star.position).value_in(units.pc)
            i_a_dist = numpy.sqrt(i_a_vector.dot(i_a_vector))
            i_a_uv = i_a_vector / i_a_dist

            # Save and to be returned for feedback calculation
            i_a_dists_pc[a, i] = i_a_dist
            i_a_uvs[a, i] = i_a_uv

            # If there is temporary increase in neighbour list
            if temporary:
                bubble_dist_i = temp_bubble_dist_i
                bubble_ind_i = temp_bubble_ind_i
            else:
                bubble_dist_i = bubble_dist[i]
                bubble_ind_i = bubble_ind[i]

            # if gas i is nearby star a, link to star a
            if (a + Ngas) in bubble_ind_i:
                # graph_dict[a].add_edge(i, a+Ngas)
                graph_dict[a].addEdge(i, a+Ngas)
                link_dists_pc[a, i] = i_a_dist
                link_uvs[a, i] = i_a_uv

            # else, compute and link to the next gas
            else:
                # Obtain true neighbours of gas i
                conditions = (bubble_ind_i < Ngas) & (bubble_dist_i > 0)
                neighbour_ind = bubble_ind_i[conditions]
                i_neighbour_dist = bubble_dist_i[conditions]

                # Remove predecessors from neighbour list if any
                if predecessors_ind:
                    intersecting_pred_ind = list(
                        set.intersection(set(predecessors_ind), set(neighbour_ind))
                    )
                    not_in_intersection = numpy.isin(
                        neighbour_ind, intersecting_pred_ind, invert=True
                    )
                    neighbour_ind = neighbour_ind[not_in_intersection]
                    i_neighbour_dist = i_neighbour_dist[not_in_intersection]

                # Extend neighbour list if empty
                neighbours = gas[neighbour_ind]
                if neighbours.is_empty():
                    new_search_radius += search_radius
                    new_bubble_dist_i, new_bubble_ind_i = gsp.bubble_neighbors(
                        numpy.array([target_position.value_in(units.pc)]),
                        distance_upper_bound=new_search_radius
                    )
                    if predecessors_ind:
                        print(
                            f'({i}, {a}): No neighbours after removing predecessors'
                        )
                        logger.info(
                            "Gas %i (#%i) to star %i (#%i): No neighbours after removing predecessors",
                            i, target.key, a, star.key
                        )
                        temp_bubble_dist_i = new_bubble_dist_i[0]
                        temp_bubble_ind_i = new_bubble_ind_i[0]
                        temporary = True
                    else:
                        print(
                            f'({i}, {a}): No real neighbours'
                        )
                        logger.info(
                            "Gas %i (#%i) to star %i (#%i): No real neighbours",
                            i, target.key, a, star.key
                        )
                        bubble_dist[i] = new_bubble_dist_i[0]
                        bubble_ind[i] = new_bubble_ind_i[0]
                    continue

                # Calculate min angle from i-a line of sight
                i_neighbour_vector = (
                    target_position - neighbours.position
                ).value_in(units.pc)
                i_neighbour_uv = [
                    i_neighbour_vector[k]/i_neighbour_dist[k]
                    for k in range(len(neighbours))
                ]
                angle_to_LOS = [
                    numpy.arccos(i_a_uv.dot(uvi)) for uvi in i_neighbour_uv
                ]
                min_angle = numpy.min(angle_to_LOS)
                if min_angle > angle_threshold:
                    print(
                        f'({i}, {a}): Min angle {min_angle} > {angle_threshold}'
                    )
                    logger.info(
                        "Gas %i (#%i) to star %i (#%i): Min angle %s > %s",
                        i, target.key, a, star.key, min_angle, angle_threshold
                    )
                    new_search_radius += search_radius
                    new_bubble_dist_i, new_bubble_ind_i = gsp.bubble_neighbors(
                        numpy.array([target_position.value_in(units.pc)]),
                        distance_upper_bound=new_search_radius
                    )
                    temp_bubble_dist_i = new_bubble_dist_i[0]
                    temp_bubble_ind_i = new_bubble_ind_i[0]
                    temporary = True
                    continue

                # Add selected neighbour into graph
                selected_neighbour_ind_ind = numpy.where(
                    numpy.array(angle_to_LOS) == min_angle
                )[0]
                selected_neighbour_ind = neighbour_ind[
                    selected_neighbour_ind_ind
                ][0]
                # graph_dict[a].add_edge(i, selected_neighbour_ind)
                graph_dict[a].addEdge(i, selected_neighbour_ind)

                # If graph is not directed acyclic (a.k.a not a forest)
                # if not graph_dict[a].is_dag():
                if networkit.components.StronglyConnectedComponents(
                    graph_dict[a]
                ).run().numberOfComponents() < (Ngas + Nstars):
                    print(
                        f'({i}, {a}): Cycle with gas {selected_neighbour_ind}'
                    )
                    logger.info(
                        "Gas %i (#%i) to star %i (#%i): Cycle with gas %i (#%i)",
                        i, target.key, a, star.key,
                        selected_neighbour_ind,
                        ind_superset_dict[selected_neighbour_ind]
                    )
                    # graph_dict[a].delete_edges(
                    #     graph_dict[a].get_eid(i, selected_neighbour_ind)
                    # )
                    # predecessors_ind = graph_dict[a].neighborhood(
                    #     i, order=8, mode='in', mindist=1
                    # )

                    graph_dict[a].removeEdge(i, selected_neighbour_ind)
                    wcc = networkit.components.WeaklyConnectedComponents(
                        graph_dict[a]
                    )
                    wcc.run()
                    predecessors_ind = wcc.getComponents()[
                        wcc.componentOfNode(i)
                    ]
                    predecessors_ind.remove(i)
                    continue

                link_dists_pc[a, i] = i_neighbour_dist[
                    selected_neighbour_ind_ind
                ][0]
                link_uvs[a, i] = i_neighbour_uv[
                    selected_neighbour_ind_ind[0]
                ]

            predecessors_ind = []
            temporary = False
            a += 1
            counter = Nstars*i + a

        # Add gas index
        i += 1

    print("Network generation done.")
    logger.info("Network generation done.")

    return (
        graph_dict, i_a_dists_pc, i_a_uvs, link_dists_pc, link_uvs,
        i_a_density_gcm
    )


def create_path_dict(graph_dict, Ngas, Nstars):
    """
    Create numba.typed.Dict of that gives all nodes between
    gas i and star a inclusive.

    Parameters:
    graph_dict: dictionary
        Star index (element of range(Nstars-1)):
        igraph.Graph network
    Ngas: int
        Number of gas
    Nstars: int
        Number of stars

    Returns:
    numba_path_dict: numba.typed.Dict
        tuple(gas i, star a): numpy.array of nodes in
        between i and a inclusive
    """

    # normal_path_dict = {
    #     (a, i):graph_dict[a].subcomponent(i, mode='out')
    #     for i in range(Ngas) for a in range(Nstars)
    # }

    normal_path_dict = {
        (a, i): [i] + networkit.distance.BidirectionalBFS(
            graph_dict[a], i, a+Ngas
        ).run().getPath() + [a+Ngas]
        for i in range(Ngas) for a in range(Nstars)
    }
    numba_path_dict = typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.int64[:]
    )
    for k, v in normal_path_dict.items():
        k1, k2 = k
        v = numpy.array(v, dtype=numpy.int64)
        numba_path_dict[k] = v

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
                print(f_ion)

    new_Nsources = [numpy.count_nonzero(fluxes[:, i]) for i in range(Ngas)]
    new_Nsources = numpy.array(new_Nsources)

    return f_ion_array, new_Nsources


def main_stellar_feedback_old(
    gas,
    stars_,
    mass_cutoff=5|units.MSun,
    angle_threshold=numpy.pi/3,
    recombination_coefficient=3e-13|units.cm**3/units.s,
    temp_range=[10,10000]|units.K,
    logger=None,
    **keyword_arguments
):
    """
    Main stellar feedback routine.
    """
    logger = logger or logging.getLogger(__name__)

    # Choose only massive stars
    stars = stars_.select_array(
        lambda x: x >= mass_cutoff,
        ["mass"]
    )
    logger.info(
        "%i massive stars > %s", 
        len(stars), mass_cutoff
    )
    gmmw = gas_mean_molecular_weight(0)
    max_u = gas.u.max()
    logger.info(
        "Max gas temperature: %s",
        u_to_temperature(max_u, gmmw=gmmw).in_(units.K)
    )

    Ngas = len(gas)
    Nstars = len(stars)

    network_result = generate_network(
        gas,
        stars,
        angle_threshold=angle_threshold,
        logger=logger,
    )
    graph_dict = network_result[0]
    i_a_dists_pc = network_result[1]
    i_a_uvs = network_result[2]
    link_dists_pc = network_result[3]
    link_uvs = network_result[4]
    i_a_density_gcm = network_result[5]

    numba_path_dict = create_path_dict(
        graph_dict, Ngas, Nstars
    )

    print("Calculating feedback...")
    logger.info("Calculating feedback...")

    photon_flux = (stars.luminosity/(13.6|units.eV)).value_in(units.s**-1)
    molecular_mass = gmmw / (1 | units.amu)
    recombination_coefficient = recombination_coefficient.value_in(
        units.cm**3 * units.s**-1
    )
    Nsources = numpy.zeros(Ngas)
    Nionised_gas = 0

    error = 10000
    delta_Nionised_gas = 10000
    count = 0
    while error > 0.01 and delta_Nionised_gas > 2:
        f_ion_array, new_Nsources = one_feedback_iteration(
            numba_path_dict,
            i_a_dists_pc,
            i_a_uvs,
            link_dists_pc,
            link_uvs,
            i_a_density_gcm,
            photon_flux,
            Nsources,
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

    print("Feedback calculation done. Populating temperature...")
    logger.info("Feedback calculation done. Populating temperature...")

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

"""
----------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------
"""

@njit
def search_neighbours_i(
    gas_pos_pc_i,
    superset_pos_pc,
    search_radius,
):
    """
    Search neighbours of gas i
    """
    Nsuperset = len(superset_pos_pc)
    dist_i = numpy.zeros(Nsuperset)
    for j in range(Nsuperset):
        dist_vec_ij = superset_pos_pc[j] - gas_pos_pc_i
        dist_ij = numpy.sqrt(dist_vec_ij.dot(dist_vec_ij))
        dist_i[j] = dist_ij
    condition = dist_i < search_radius
    ind_i = numpy.argwhere(condition).flatten()
    dist_i = dist_i[condition]
    return dist_i, ind_i


# @njit(parallel=True, cache=True)
def search_all_neighbours(
    bubble_dist,
    bubble_ind,
    gas_pos_pc,
    gas_h_pc,
    superset_pos_pc,
):
    """
    Search neighbours of all gas using search_neighbours_i
    Using GrisPy here because apparently NumPy is more efficient?
    """
    Ngas = len(gas_pos_pc)
    gsp = GriSPy(superset_pos_pc)
    bubble_dist_grispy, bubble_ind_grispy = gsp.bubble_neighbors(
        gas_pos_pc,
        distance_upper_bound=2*gas_h_pc
    )
    print('done grispying')
    for i in range(Ngas):
        bubble_ind_grispy_i = bubble_ind_grispy[i].astype(numpy.int64)
        bubble_dist[i] = bubble_dist_grispy[i]
        bubble_ind[i] = bubble_ind_grispy_i
    print('done dictionary')

    # Ngas = len(gas_pos_pc)
    # for i in range(Ngas):
    #     dist_i, ind_i = search_neighbours_i(
    #         gas_pos_pc[i],
    #         superset_pos_pc,
    #         2*gas_h_pc[i]
    #     )
    #     bubble_dist[i] = dist_i
    #     bubble_ind[i] = ind_i

    return bubble_dist, bubble_ind


@njit
def search_predecessors(
    i,
    network_a,
    max_count=2**8
):
    """
    Breadth-first-search method to find predecessors of i
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


@njit
def numba_generate_network(
    gas_pos_pc,
    gas_h_pc,
    stars_pos_pc,
    superset_pos_pc,
    bubble_dist,
    bubble_ind,
    angle_threshold=numpy.pi/3
):
    """
    gas_pos_pc: (Ngas, 3)
    stars_pos_pc: (Nstars, 3)
    gas_h_pc: (Ngas)
    """
    Ngas = len(gas_pos_pc)
    Nstars = len(stars_pos_pc)

    # Arrays to be returned
    network = numpy.full((Nstars, Ngas), -1, numpy.int64)
    i_a_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float64)
    i_a_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float64)
    link_dists_pc = numpy.zeros((Nstars, Ngas), numpy.float64)
    link_uvs = numpy.zeros((Nstars, Ngas, 3), numpy.float64)

    # Initialise for numba
    temp_bubble_dist_i = numpy.zeros(1, numpy.float64)
    temp_bubble_ind_i = numpy.zeros(1, numpy.int64)
    predecessors_ind = numpy.zeros(1, numpy.int64)

    counter = i = 0
    temporary = False
    cycle = False
    Nloops = Ngas * Nstars
    count_real_neigh = count_cycle_neigh = count_angle = count_cycle = 0
    while counter < Nloops:
        a = 0
        target_position = gas_pos_pc[i]
        search_radius = 2 * gas_h_pc[i]
        new_search_radius = search_radius
        while a < Nstars:
            # print(i, a, counter)
            star_position = stars_pos_pc[a]
            i_a_vector = target_position - star_position
            i_a_dist = numpy.sqrt(i_a_vector.dot(i_a_vector))
            i_a_uv = i_a_vector / i_a_dist

            # Save and to be returned for feedback calculation
            i_a_dists_pc[a, i] = i_a_dist
            i_a_uvs[a, i] = i_a_uv

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
                        target_position,
                        superset_pos_pc,
                        new_search_radius
                    )
                    if cycle:
                        print(
                            a, i, "No neighbour after removing predecessors"
                        )
                        temp_bubble_dist_i = new_bubble_dist_i
                        temp_bubble_ind_i = new_bubble_ind_i
                        temporary = True
                        count_cycle_neigh += 1
                    else:
                        print(a, i, "No real neighbour")
                        bubble_dist[i] = new_bubble_dist_i
                        bubble_ind[i] = new_bubble_ind_i
                        count_real_neigh += 1
                    continue

                # Calculate min angle from i-a line of sight
                i_neighbour_vector = (
                    target_position - gas_pos_pc[neighbour_ind]
                )
                i_neighbour_uv = [
                    i_neighbour_vector[k]/i_neighbour_dist[k]
                    for k in range(len(neighbour_ind))
                ]
                angle_to_LOS = numpy.array([
                    numpy.arccos(i_a_uv.dot(uvi)) for uvi in i_neighbour_uv
                ])
                min_angle = numpy.min(angle_to_LOS)
                if min_angle > angle_threshold:
                    print(
                        a, i, "Min angle", min_angle, ">", angle_threshold
                    )
                    new_search_radius += search_radius
                    new_bubble_dist_i, new_bubble_ind_i = search_neighbours_i(
                        target_position,
                        superset_pos_pc,
                        new_search_radius
                    )
                    temp_bubble_dist_i = new_bubble_dist_i
                    temp_bubble_ind_i = new_bubble_ind_i
                    temporary = True
                    count_angle += 1
                    continue

                selected_neighbour_ind_ind = numpy.where(
                    angle_to_LOS == min_angle
                )[0]
                selected_neighbour_ind = neighbour_ind[
                    selected_neighbour_ind_ind
                ][0]

                # Check if selected neighbour creates a
                # cycle
                predecessors_ind = search_predecessors(
                    i,
                    network[a]
                )
                if selected_neighbour_ind in predecessors_ind:
                    print(
                        a, i, 'Cycle with', selected_neighbour_ind
                    )
                    cycle = True
                    count_cycle += 1
                    continue

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

        # Add gas index
        i += 1

    return (
        network, i_a_dists_pc, i_a_uvs, link_dists_pc, link_uvs,
        [count_real_neigh, count_cycle_neigh, count_angle, count_cycle]
    )


# @njit
# def find_path(network, i, a, Ngas):
#     """
#     Find path from gas i to star a
#     """
#     path = numpy.array([i])
#     while i < Ngas:
#         j = network[a, i]
#         path = numpy.append(path, j)
#         i = j
#     return path


@njit
def create_path_dict_numba(numba_path_dict, network, Ngas, Nstars):
    for i in range(Ngas):
        for a in range(Nstars):
            key = (a, i)
            path = numpy.array([i], numpy.int64)
            while i < Ngas:
                j = network[a, i]
                path = numpy.append(path, j)
                i = j
            numba_path_dict[key] = path
    return numba_path_dict


def main_stellar_feedback(
    gas,
    stars_,
    mass_cutoff=1|units.MSun,
    angle_threshold=numpy.pi/3,
    recombination_coefficient=3e-13|units.cm**3/units.s,
    temp_range=[10,10000]|units.K,
    logger=None,
    **keyword_arguments
):
    """
    Using as much numba as possible and no dependency on GriSPy
    and networkit
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

    gmmw = gas_mean_molecular_weight(0)
    max_u = gas.u.max()
    logger.info(
        "Max gas temperature: %s",
        u_to_temperature(max_u, gmmw=gmmw).in_(units.K)
    )

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

    print('Seaching neighbours...')
    logger.info('Searching neighbours...')
    bubble_dist, bubble_ind = search_all_neighbours(
        bubble_dist,
        bubble_ind,
        gas_pos_pc,
        gas_h_pc,
        superset_pos_pc,
    )

    print('Generating network...')
    logger.info('Generating network...')
    network, i_a_dists_pc, i_a_uvs, link_dists_pc, link_uvs, stat = (
        numba_generate_network(
            gas_pos_pc,
            gas_h_pc,
            stars_pos_pc,
            superset_pos_pc,
            bubble_dist,
            bubble_ind,
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
        numba_path_dict, network, Ngas, Nstars
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
        f_ion_array, new_Nsources = one_feedback_iteration(
            numba_path_dict,
            i_a_dists_pc,
            i_a_uvs,
            link_dists_pc,
            link_uvs,
            i_a_density_gcm,
            photon_flux,
            Nsources,
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


def basic_stroemgren_volume_method_old(
    gas_,
    stars,
    mass_cutoff=5|units.MSun,
    recombination_coefficient=3e-13|units.cm**3/units.s,
    cutoff_ratio=0.01,
    target_temp=10000|units.K,
    ):
    """
    Basic Stroemgren volume method as described in Dale et al (2007).
    Update the gas temperature for those gas particles that receive
    positive flux from the stars.

    This is the pre 8/10/2021 version.
    """
    gas_particles = gas_.copy()
    massive_stars = stars.copy().select_array(
        lambda m:
        m >= mass_cutoff,
        ["mass"]
    )
    # number of iteration after latest positive flux
    Nmax = min(200, int(cutoff_ratio*len(gas_particles)))

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
        last_positive_flux_radius = 0 | units.pc

        for target in gas_particles:
            target_u = temperature_to_u(
                target_temp,
                gmmw=gas_mean_molecular_weight(0),
            )
            print(f'DEBUG: Target is #{target.key}, loop counter = {loop_counter}')

            if loop_counter == Nmax:
                print(f'{Nmax} loops after last +ve successful flux! Break' )
                break

            main_LOS_vector = target.position - star.position
            main_LOS_distance = main_LOS_vector.length()
            main_LOS_unit_vector = main_LOS_vector / main_LOS_distance

            gasset = gas_particles.copy()
            gasi = target.copy()  # 'i' starts from target, unlike Dale2007
            # target_in_gas_particles = gas_particles[
            #     gas_particles.key == gasi.key
            # ]

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

                    # target_in_gas_particles.is_linked == gasi.key
                    # target_in_gas_particles.is_linked_to = star.key

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
                            # target_in_gas_particles.is_dead_end = True

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
                            # target_in_gas_particles.is_dead_end = True


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
                        # target_in_gas_particles.is_linked = True
                        # target_in_gas_particles.is_linked_to = selected_neighbour.key

                        gas_particles[
                            gas_particles.key == gasi.key
                        ].is_linked = True
                        gas_particles[
                            gas_particles.key == gasi.key
                        ].is_linked_to = selected_neighbour.key

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
                molecular_mass = gas_mean_molecular_weight(0)
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
                    # target_in_gas_particles.u = target_u
                    # target_in_gas_particles.h2ratio = 0
                    # target_in_gas_particles.proton_abundance = 1

                    gas_particles[
                        gas_particles.key == gasi.key
                    ].u = target_u
                    gas_particles[
                        gas_particles.key == gasi.key
                    ].h2ratio = 0
                    gas_particles[
                        gas_particles.key == gasi.key
                    ].proton_abundance = 1

                    last_positive_flux_radius = numpy.max(eval_radii)
                    loop_counter = 0

                else:
                    loop_counter += 1
                    # if loop_counter == Nmax:
                    #     print(f'{Nmax} loops after last +ve successful flux! Break' )
                    #     break



        # Let all gas within certain distance from star to ionise absolutely
        # Should improve here
        r_abs = last_positive_flux_radius/3
        nearby_gas_particles = gas_particles.select_array(
            lambda x: x < r_abs, ['distance_to_star']
        )
        if not nearby_gas_particles.is_empty():
            print(f'{len(nearby_gas_particles)} nearby gas particles to star')
            nearby_gas_particles.u = target_u     # This is wrong if target_u is not constant
            nearby_gas_particles.h2ratio = 0
            nearby_gas_particles.proton_abundance = 1





        print(f'DEBUG: still in star #{star.key} loop')

        print(f'DEBUG: break neighbour {break_neighbour_counter}')
        print(f'DEBUG: break angle {break_angle_counter}')
        print(f'DEBUG: break dead end {break_dead_counter}')
        print(f'DEBUG: success {success_counter}')

    # DF['flux'] = flux_list
    # DF['distance_origin'] = distance_origin_list
    # DF['distance_star'] = distance_star_list
    # DF.to_csv('checkflux.csv')



    return gas_particles
