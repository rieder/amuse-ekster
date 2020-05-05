#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the Ekster module
"""
import numpy
from amuse.ext.masc import new_star_cluster
from amuse.units import units
from amuse.ic.gasplummer import new_plummer_gas_model

from star_cluster_class import StarCluster
from gas_class import GasCode
# from stellar_dynamics_class import StellarDynamicsCode
# from stellar_evolution_class import StellarEvolutionCode

def test_starcluster_init():
    "Test if the particles match"
    numpy.random.seed(123)
    number_of_stars = 100
    stars = new_star_cluster(number_of_stars=number_of_stars)
    masses = stars.mass
    cluster = StarCluster(stars)
    assert len(cluster.particles) == number_of_stars
    assert len(cluster.evo_code.particles) == number_of_stars
    assert len(cluster.star_code.particles) == number_of_stars
    assert cluster.particles.mass.sum() == masses.sum()
    assert cluster.evo_code.particles.mass.sum() == masses.sum()
    assert cluster.star_code.particles.mass.sum() == masses.sum()
    cluster.stop()

def test_starcluster_evostep():
    "Test evolution step"
    numpy.random.seed(123)
    number_of_stars = 1000
    time = 0.01 | units.Myr
    stars = new_star_cluster(number_of_stars=number_of_stars)
    masses = stars.mass
    cluster = StarCluster(stars)
    cluster.evolve_model(time)
    assert cluster.model_time >= time
    assert cluster.star_code.model_time >= time
    assert cluster.evo_code.model_time >= time

    assert (
        cluster.particles.mass.sum()
        == cluster.evo_code.particles.mass.sum()
    )
    assert (
        cluster.particles.mass.sum()
        <= masses.sum()
    )

    time_step = cluster.evo_code.particles.time_step.min()
    cluster.evolve_model(time + time_step)
    assert (
        cluster.particles.mass.sum()
        < masses.sum()
    )
    cluster.stop()

def test_gas_init():
    "Test if the gas class works"
    numpy.random.seed(123)
    number_of_gas = 1000
    gas = new_plummer_gas_model(number_of_gas)
    masses = gas.mass
    cloud = GasCode(gas)
    assert len(cloud.particles) == number_of_gas
    assert cloud.particles.mass.sum() == masses.sum()
    cloud.stop()
