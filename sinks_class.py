#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sink particles
"""
import numpy
from amuse.units import units
from amuse.datamodel import Particles  # , ParticlesOverlay


def should_a_sink_form(
        origin_gas,
        gas,
        check_thermal=False,
    ):
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
        return False, "not enough neighbours"
    neighbours.position -= origin_gas.position
    neighbours.velocity -= origin_gas.velocity
    neighbours.distance = neighbours.position.lengths()
    neighbours = neighbours.sorted_by_attribute("distance")
    e_kin = neighbours[:50].kinetic_energy()
    # e_rot = #FIXME
    e_pot = neighbours[:50].potential_energy()
    e_th = neighbours[:50].thermal_energy()

    if check_thermal:
        try:
            if not e_th/e_pot <= 0.5:
                return False, "e_th/e_pot > 0.5"
        except AttributeError:
            print(
                "ERROR: e_th = %s e_pot = %s"
                % (e_th, e_pot)
            )
            return False, "error"
        # if not (e_th + e_rot) / e_pot <= 1:
        #     break
        if (e_th+e_kin+e_pot) >= 0 | units.erg:
            return False, "e_tot < 0"
    else:
        if (e_pot+e_kin) >= 0 | units.erg:
            return False, "e_tot < 0"
    # if accelleration is diverging:
    #     break

    #    these must be in a sensible-sized sphere around the central one
    #    so make a cutout, then calculate distance, then sort and use [:50]
    #    particles.thermal_energy (make sure u corresponds to 10K)
    #    particles.kinetic_energy
    #    particles.potential_energy
    return True, "forming"


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
