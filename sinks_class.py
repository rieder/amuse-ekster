#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sink particles
"""
import numpy
import logging
from amuse.units import units, nbody_system
from amuse.datamodel import Particles  # , ParticlesOverlay
from amuse.community.hermite.interface import Hermite


def should_a_sink_form(
        all_origin_gas,
        gas,
        check_thermal=False,
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
        logger.info("Checking if particle %s should form a sink", origin_gas.key)
        if origin_gas.h_smooth > accretion_radius/2:
            flags.append(False)
            messages.append("smoothing length too large")
            logger.info("No - smoothing length too large")
            break
        if len(gas) < 50:
            return False, "not enough gas particles (this should never happen)"
        neighbour_radius = origin_gas.h_smooth * 5
        neighbours = gas[
            numpy.where(
                (gas.position - origin_gas.position).lengths()
                < neighbour_radius
            )
        ].copy()
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
        converter = nbody_system.nbody_to_si(
            neighbours.total_mass(), neighbours.distance.mean()
        )
        helper = Hermite(converter)
        helper.particles.add_particles(neighbours)
        e_kin = helper.kinetic_energy
        # e_rot = #FIXME
        e_pot = helper.potential_energy
        helper.stop()
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
            neighbours[selection_yz].mass * rcrossvx[selection_yz]**2/radyz2[selection_yz]
        ).sum()
        e_rot_y = (
            neighbours[selection_xz].mass * rcrossvy[selection_xz]**2/radxz2[selection_xz]
        ).sum()
        e_rot_z = (
            neighbours[selection_xy].mass * rcrossvz[selection_xy]**2/radxy2[selection_xy]
        ).sum()

        e_rot = (e_rot_x**2 + e_rot_y**2 + e_rot_z**2)**0.5

        alpha_grav = abs(e_th / e_pot)
        try:
            if alpha_grav > 0.5:
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
            # return False, "error"
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
            # return False, "e_tot < 0"
            flags.append(False)
            messages.append("e_tot >= 0")
            logger.info("No - %s", messages[-1])
            break
        logger.info("e_tot < 0")
        # else:
        #     if (e_pot+e_kin) >= 0 | units.erg:
        #         # return False, "e_tot < 0"
        #         flags.append(False)
        #         messages.append("e_tot < 0")
        #         logger.info("No - %s", messages[-1])
        #         break


        # if accelleration is diverging:
        #     break


        #    these must be in a sensible-sized sphere around the central one
        #    so make a cutout, then calculate distance, then sort and use [:50]
        #    particles.thermal_energy (make sure u corresponds to 10K)
        #    particles.kinetic_energy
        #    particles.potential_energy
        flags.append(True)
        messages.append("forming")
    return flags, messages


#
# class SinkParticles(ParticlesOverlay):
#     "Sink particle type"
#     def __init__(
#             self, original_particles, sink_radius=None, mass=None,
#             position=None, velocity=None, angular_momentum=None,
#             looping_over="sinks",
#     ):
#         ParticlesOverlay.__init__(self, original_particles)
#         self._private.looping_over = looping_over
#         self.sink_radius = sink_radius or original_particles.radius
#         if not hasattr(original_particles, "mass"):
#             self.mass = mass or (([0.] * len(self)) | units.kg)
# 
#         if not hasattr(original_particles, "x"):
#             self.position = (
#                 position
#                 or (
#                     ([[0., 0., 0.]] * len(self))
#                     | units.m
#                 )
#             )
# 
#         if not hasattr(original_particles, "vx"):
#             self.velocity = (
#                 velocity
#                 or (
#                     ([[0., 0., 0.]] * len(self))
#                     | units.m / units.s
#                 )
#             )
# 
#         if not hasattr(original_particles, "lx"):
#             self.angular_momentum = (
#                 angular_momentum
#                 or (
#                     ([[0., 0., 0.]] * len(self))
#                     | units.g * units.m**2 / units.s
#                 )
#             )
# 
#     def accrete(self, particles):
#         if self._private.looping_over == "sinks":
#             return self.accrete_looping_over_sinks(particles)
#         else:
#             return self.accrete_looping_over_sources(particles)
# 
#     def add_particles_to_store(self, keys, attributes=[], values=[]):
#         (
#             (attributes_inbase, values_inbase),
#             (attributes_inoverlay, values_inoverlay)
#         ) = self._split_attributes_and_values(attributes, values)
# 
#         self._private.overlay_set.add_particles_to_store(
#             keys,
#             attributes_inoverlay,
#             values_inoverlay,
#         )
# 
#         particles = self._private.base_set._original_set()._subset(keys)
#         self._private.base_set = self._private.base_set + particles
# 
#     def add_sinks(
#             self, original_particles, sink_radius=None, mass=None,
#             position=None, velocity=None, angular_momentum=None,
#     ):
#         new_sinks = self.add_particles(original_particles)
#         new_sinks.sink_radius = sink_radius or original_particles.radius
# 
#         if not hasattr(original_particles, "mass"):
#             new_sinks.mass = mass or (
#                 ([0.] * len(new_sinks))
#                 | units.kg
#             )
# 
#         if not hasattr(original_particles, "x"):
#             new_sinks.position = position or (
#                 ([[0., 0., 0.]] * len(new_sinks))
#                 | units.m
#             )
# 
#         if not hasattr(original_particles, "vx"):
#             new_sinks.velocity = velocity or (
#                 ([[0., 0., 0.]] * len(new_sinks))
#                 | units.m / units.s
#             )
# 
#         if not hasattr(original_particles, "lx"):
#             new_sinks.angular_momentum = angular_momentum or (
#                 ([[0., 0., 0.]] * len(new_sinks))
#                 | units.g * units.m**2 / units.s
#             )
# 
#     def add_sink(self, particle):
#         self.add_sinks(particle.as_set())
# 
#     def select_too_close(self, others):
#         too_close = []
#         for pos, r_squared in zip(self.position, self.sink_radius**2):
#             subset = others[
#                 (other.position - pos).lengths_squared()
#                 < r_squared
#             ]
#             too_close.append(subset)
#         return too_close
# 
#     def accrete_looping_over_sinks(self, original_particles):
#         particles = original_particles.copy()
#         others = (
#             particles - self.get_intersecting_subset_in(particles)
#         )
#         too_close = self.select_too_close(others)
#         try:
#             all_too_close = sum(too_close, particles[0:0])
#         except AmuseException:
#             too_close = self.resolve_duplicates(too_close, particles)
#             all_too_close = sum(too_close, particles[0:0])
#         if not all_too_close.is_empty():
#             self.aggregate_mass(too_close)
#             original_particles.remove_particles(all_too_close)
#         return all_too_close
# 
#     def resolve_duplicates(self, too_close, particles):
#         """
#         Find particles that are within the radius of more than one sink
#         """
#         duplicates = particles[0:0]
#         keys = set()
#         for subset in too_close:
#             for particle in subset:
#                 if (
#                         (particle.key in keys) 
#                         and (particle.key not in duplicates.key)
#                 ):
#                     duplicates += particle
#                 else:
#                     keys.add(particle.key)
# 
#         strongest_sinks = []
#         for duplicate in duplicates:
#             candidate_sinks = []
#             for index, subset in enumerate(too_close):
#                 if duplicate in subset:
#                     candidate_sinks.append(index)
# 

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
