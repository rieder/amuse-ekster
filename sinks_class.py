#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sink particles
"""
import numpy
import logging
from amuse.units import units
from amuse.datamodel import Particles


def should_a_sink_form(
        all_origin_gas,
        gas,
        check_thermal=True,
        accretion_radius=0.1 | units.pc,
        logger=None,
):
    logger = logger or logging.getLogger(__name__)
    # Check if conditions for forming a sink are met
    # This applies to the ~50 SPH neighbour particles
    # - ratio of thermal to gravitational energies is <= 1/2
    # - sum of thermal and rotational energies over gravitational energy
    #   is <= 1
    # - total energy is negative
    # - divergence of particles' acceleration is negative
    # OR
    # - density = 10 * critical
    # 1) get the 50 neighbour particles
    flags = []
    messages = []
    for origin_gas in all_origin_gas.as_set():
        logger.info(
            "Checking if particle %s should form a sink", origin_gas.key
        )
        neighbour_radius = accretion_radius  # origin_gas.h_smooth * 5
        neighbours = gas[
            numpy.where(
                (gas.position - origin_gas.position).lengths()
                < neighbour_radius
            )
        ].copy()
        if origin_gas.h_smooth > accretion_radius/2:
            flags.append(False)
            messages.append("smoothing length too large")
            logger.info("No - smoothing length too large")
            break
        if len(gas) < 50:
            return False, "not enough gas particles (this should never happen)"

        if len(neighbours) < 50:
            # return False, "not enough neighbours"
            flags.append(False)
            messages.append("not enough neighbours")
            logger.info("No - not enough neighbours")
            break
        neighbours.position -= origin_gas.position
        neighbours.velocity -= origin_gas.velocity
        neighbours.distance = neighbours.position.lengths()
        neighbours = neighbours.sorted_by_attribute("distance")
        e_kin = neighbours.kinetic_energy()
        e_pot = neighbours.potential_energy()
        if check_thermal:
            e_th = neighbours.thermal_energy()
        else:
            e_th = 0 * e_pot

        dx = neighbours.x
        dy = neighbours.y
        dz = neighbours.z
        dvx = neighbours.vx
        dvy = neighbours.vy
        dvz = neighbours.vz
        rcrossvx = (dy*dvz - dz*dvy)
        rcrossvy = (dz*dvx - dx*dvz)
        rcrossvz = (dx*dvy - dy*dvx)
        radxy2 = dx*dx + dy*dy
        radyz2 = dy*dy + dz*dz
        radxz2 = dx*dx + dz*dz

        selection_yz = radyz2 > 0 | dx.unit**2
        selection_xz = radxz2 > 0 | dx.unit**2
        selection_xy = radxy2 > 0 | dx.unit**2
        e_rot_x = (
            neighbours[selection_yz].mass
            * rcrossvx[selection_yz]**2/radyz2[selection_yz]
        ).sum()
        e_rot_y = (
            neighbours[selection_xz].mass
            * rcrossvy[selection_xz]**2/radxz2[selection_xz]
        ).sum()
        e_rot_z = (
            neighbours[selection_xy].mass
            * rcrossvz[selection_xy]**2/radxy2[selection_xy]
        ).sum()

        e_rot = (e_rot_x**2 + e_rot_y**2 + e_rot_z**2)**0.5

        alpha_grav = abs(e_th / e_pot)
        try:
            if alpha_grav > 0.5:
                # print("e_th / e_pot = %s" % alpha_grav)
                # return False, "e_th/e_pot > 0.5"
                flags.append(False)
                messages.append("e_th/e_pot > 0.5")
                logger.info("No - %s", messages[-1])
                break
        except AttributeError:
            print(
                "ERROR: e_th = %s e_pot = %s"
                % (e_th, e_pot)
            )
            flags.append(False)
            messages.append("error")
            break
        logger.info("e_th/e_pot <= 0.5")
        alphabeta_grav = alpha_grav + abs(e_rot / e_pot)
        if alphabeta_grav > 1.0:
            flags.append(False)
            messages.append("e_rot too big")
            logger.info("No - %s", messages[-1])
            break
        logger.info("e_th/e_pot + e_rot/e_pot <= 1")
        # if not (e_th + e_rot) / e_pot <= 1:
        #     break
        if (e_th+e_kin+e_pot) >= 0 | units.erg:
            flags.append(False)
            messages.append("e_tot >= 0")
            logger.info("No - %s", messages[-1])
            break
        logger.info("e_tot < 0")

        # if accelleration is diverging:
        #     break

        #    these must be in a sensible-sized sphere around the central one
        #    so make a cutout, then calculate distance, then sort and use [:50]
        #    particles.thermal_energy (make sure u corresponds to 10K)
        #    particles.kinetic_energy
        #    particles.potential_energy
        flags.append(True)
        messages.append("forming")
        logger.info("All checks clear - forming a sink")
    return flags, messages, neighbours


def accrete_gas(sink, gas):
    "Accrete gas within sink radius"
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
