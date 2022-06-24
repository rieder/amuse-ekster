from amuse.test.amusetest import TestWithMPI

from amuse.datamodel.particles import Particle
from amuse.units import (nbody_system, units)
from amuse.ic.gasplummer import new_plummer_gas_model

from sinks_class import should_a_sink_form
from plotting_class import temperature_to_u


class TestSinkAccretion(TestWithMPI):
    converter = nbody_system.nbody_to_si(2 | units.pc, 10000 | units.MSun)
    gas = new_plummer_gas_model(100000, converter)
    gas.h_smooth = 0.1 | units.pc
    gas.u = temperature_to_u(10 | units.K)
    origin_gas = Particle()
    origin_gas.x = 0 | units.pc
    origin_gas.y = 0 | units.pc
    origin_gas.z = 0 | units.pc
    origin_gas.vx = 0 | units.kms
    origin_gas.vy = 0 | units.kms
    origin_gas.vz = 0 | units.kms
    origin_gas.mass = gas[0].mass
    origin_gas.h_smooth = 0.03 | units.pc
    origin_gas.u = 100 | units.kms**2

    def test_forming(self):
        """
        Tests if this cloud indeed forms a sink from origin particle
        """
        result, message = should_a_sink_form(
            self.origin_gas, self.gas,
            check_thermal=True,
            accretion_radius=0.1 | units.pc,
        )
        self.assertEqual([True], result)

    def test_divergence(self):
        self.skip("Not yet implemented")

    def test_smoothing_length_too_large(self):
        """
        Tests if this cloud fails to form a sink from origin particle because
        its smoothing length is too large
        """
        h_smooth = 0.05 | units.pc
        origin_gas = self.origin_gas.copy()
        origin_gas.h_smooth = h_smooth
        result, message = should_a_sink_form(
            origin_gas, self.gas,
            check_thermal=True,
            accretion_radius=1.99*h_smooth,
        )
        self.assertEqual([False], result)
        self.assertEqual(["smoothing length too large"], message)

    def test_too_hot(self):
        """
        Tests if this cloud fails to form a sink from origin particle because
        its thermal energy is too high
        """
        gas = self.gas.copy()
        gas.u = temperature_to_u(20 | units.K)
        result, message = should_a_sink_form(
            self.origin_gas, gas,
            check_thermal=True,
            accretion_radius=0.1 | units.pc,
        )
        self.assertEqual([False], result)
        self.assertEqual(["e_th/e_pot > 0.5"], message)

    def test_expanding(self):
        """
        Tests if this cloud fails to form a sink from origin particle because
        its kinetic energy is too high
        """
        gas = self.gas.copy()
        gas.vx = gas.x / (10 | units.kyr)
        gas.vy = gas.y / (10 | units.kyr)
        gas.vz = gas.z / (10 | units.kyr)
        result, message = should_a_sink_form(
            self.origin_gas, gas,
            check_thermal=True,
            accretion_radius=0.1 | units.pc,
        )
        self.assertEqual([False], result)
        self.assertEqual(["e_tot < 0"], message)

    def test_rotating(self):
        """
        Tests if this cloud fails to form a sink from origin particle because
        it is rotating
        """
        gas = self.gas.copy()
        gas.vx = gas.y / (10 | units.kyr)
        gas.vy = -gas.x / (10 | units.kyr)
        result, message = should_a_sink_form(
            self.origin_gas, gas,
            check_thermal=True,
            accretion_radius=0.1 | units.pc,
        )
        self.assertEqual(["e_rot too big"], message)
        self.assertEqual([False], result)
