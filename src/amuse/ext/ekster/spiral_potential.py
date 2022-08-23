#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the spiral arm potential
"""
import logging

from numpy import (
    # pi, sin, cos, tan, arctan, cosh, log, exp, tanh
    pi, cosh, log, exp, tanh,
)
from amuse.units.trigo import sin, cos, tan, arctan

from amuse.units import units
from amuse.units.constants import G

from amuse.support.literature import LiteratureReferencesMixIn

logger = logging.getLogger(__name__)


class DefaultSpiralModelParameters:
    "Default parameters for spiral arms model"
    def __init__(self):
        self.fiducial_radius = 2.47518e22 | units.cm
        self.number_of_arms = 2
        self.pitch_angle = 0.261799
        self.radial_scale_length = 2.16578e22 | units.cm
        self.density_at_fiducial_radius = (
            2.12889e-24 | units.g * units.cm**-3
        )
        self.phir = 6.3e-16 | units.s**-1
        self.scale_height = 5.56916e20 | units.cm

    def print_parameters(self):
        "Print the current parameters"
        print(self)

    def save_parameters(self):
        "Write parameters to file"
        print(self)


class DefaultLogarithmicModelParameters:
    "Default parameters for logarithmic potential (e.g. Binney & Tremaine)"
    def __init__(self):
        # speed squared and constants
        self.Co = 2.31e14 | units.cm**2 * units.s**-2
        # distance
        self.Rc = 3.09398e21 | units.cm
        # dimensionless
        self.zq = 0.7

    def print_parameters(self):
        "Print the current parameters"
        print(self)

    def save_parameters(self):
        "Write parameters to file"
        print(self)


class TimeDependentSpiralArmsDiskModel(
        LiteratureReferencesMixIn,
):
    """
    Spiral arms + disk model
    """

    def __init__(
            self,
            t_start=0 | units.yr,
            spiral_type="normal",
    ):
        # self.__name__ = "TimeDependentSpiralArmsDiskModel"

        LiteratureReferencesMixIn.__init__(self)

        logger.info("Creating LogarithmicDiskProfile")
        self.disk = LogarithmicDiskProfile(
        )
        logger.info("Creating SpiralArmsProfile")
        if spiral_type == "normal":
            logger.info("Using spiral type normal")
            self.spiralarms = SpiralArmsProfile(
                t_start=t_start,
            )
        elif spiral_type == "strong":
            logger.info("Using spiral type strong")
            self.spiralarms = StrongSpiralArmsProfile(
                t_start=t_start,
            )
        else:
            logger.info("Unknown spiral type, defaulting to normal")
            self.spiralarms = SpiralArmsProfile(
                t_start=t_start,
            )

    @property
    def __name__(self):
        return "%s with %s" % (
            self.disk.__name__,
            self.spiralarms.__name__,
            )

    def evolve_model(self, time):
        "Evolve model to specified time"
        self.disk.evolve_model(time)
        self.spiralarms.evolve_model(time)
        self.model_time = min(
            self.disk.model_time,
            self.spiralarms.model_time,
        )

    def get_gravity_at_point(self, eps, x, y, z):
        "return gravity at specified location"
        disk_force = self.disk.get_gravity_at_point(eps, x, y, z)
        spiral_force = self.spiralarms.get_gravity_at_point(eps, x, y, z)
        force = []
        for i in range(3):
            force.append(disk_force[i] + spiral_force[i])
        return force

    def get_potential_at_point(self, eps, x, y, z):
        "return potential at specified location"
        return(
            self.disk.get_potential_at_point(eps, x, y, z)
            + self.spiralarms.get_potential_at_point(eps, x, y, z)
        )

    def vel_circ(self, radius):
        "return circular velocity at specified radial distance"
        return -1

    def stop(self):
        "standard method"
        print("Stopping %s" % self.__name__)
        return


class LogarithmicDiskProfile(
        DefaultLogarithmicModelParameters,
        LiteratureReferencesMixIn,
):
    """
    Logarithmic potential

    .. [#] Binney & Tremaine

    """

    def __init__(self,):
        # self.__name__ = "LogarithmicDiskProfile"

        LiteratureReferencesMixIn.__init__(self)
        logger.info("Setting DefaultLogarithmicModelParameters")
        DefaultLogarithmicModelParameters.__init__(self)

        self.model_time = 0 | units.Myr

    @property
    def __name__(self):
        return "Logarithmic profile"

    def evolve_model(self, time):
        "just change the model time since model is time-independent"
        self.model_time = time

    def get_potential_at_point(self, eps, x, y, z):
        """Return the potential at specified point"""
        # BT 2.54a:
        # phi_L = 1/2 v0**2 ln(Rc**2 + R**2 + z**2/q**2) + const
        # Co = v0**2
        # Rc = Rc
        # zq = q

        # Which unit we use here is not important, since it only determines the
        # constant value (which we ignore). But we still have to choose some
        # length unit, since we can't take the log of a unit.
        length_unit = units.kpc

        r2 = x**2 + y**2

        V = self.Co * log(
            (self.Rc**2 + r2 + (z**2 / self.zq**2)).value_in(length_unit**2)
        )
        return V

    def get_gravity_at_point(self, eps, x, y, z):
        """
        Returns gravity at specified point
        Input: eps, x, y, z
        Returns fx,fy,fz
        """

        r2 = x**2 + y**2

        # Forces from logarithmic potential
        fx = -2. * self.Co * x / (
            self.Rc**2 + r2 + (z/self.zq)**2
        )
        fy = -2. * self.Co * y / (
            self.Rc**2 + r2 + (z/self.zq)**2
        )
        fz = -2. * self.Co * z / (
            (self.Rc**2 + r2 + (z/self.zq)**2) * self.zq**2
        )

        return (
            fx.in_(units.parsec*units.Myr**-2),
            fy.in_(units.parsec*units.Myr**-2),
            fz.in_(units.parsec*units.Myr**-2),
        )


class SpiralArmsProfile(
        DefaultSpiralModelParameters,
        LiteratureReferencesMixIn,
):
    """
    Spiral arms potential model

    .. [#] Cox & Gomez (2002)

    """

    def __init__(
            self,
            t_start=0 | units.Myr,
    ):
        # self.__name__ = "SpiralArmsProfile"

        logger.info("Setting DefaultSpiralModelParameters")
        DefaultSpiralModelParameters.__init__(self)
        LiteratureReferencesMixIn.__init__(self)
        self.time_initial = t_start
        self.model_time = 0 | units.Myr
        # Cz
        self.Cz = [
            8 / (3*pi),
            0.5,
            8 / (15*pi)
        ]

    @property
    def __name__(self):
        return "Spiral arms profile"

    def evolve_model(self, time):
        self.model_time = time

    def get_potential_at_point(self, eps, x, y, z):
        """
        Calculate the spiral arm potential at specified point

        from Cox & Gomez 2002
        Input: eps, x, y, z
        Returns: potential
        """

        phi = arctan(y/x)

        # if x < (0 | units.parsec):
        #     phi = pi + phi

        # d2
        r = (x**2+y**2)**0.5

        gamma = (
            self.number_of_arms
            * (
                phi
                + self.phir * (self.time_initial + self.model_time)
                - log(r/self.fiducial_radius) / tan(self.pitch_angle)
            )
        )

        result = 0 | units.parsec
        for n in range(3):
            Kn = (n+1) * self.number_of_arms / (r*sin(self.pitch_angle))
            Bn = Kn * self.scale_height * (1 + 0.4 * Kn * self.scale_height)
            Dn = (
                1 + Kn * self.scale_height + 0.3 * (Kn * self.scale_height)**2
            ) / (1 + 0.3 * Kn * self.scale_height)

            result = (
                result
                + (self.Cz[n] / (Dn * Kn))
                * cos((n+1) * gamma)
                * (1 / cosh((Kn*z) / Bn))**Bn
            )

        spiral_value = (
            -4 * pi * G
            * self.scale_height
            * self.density_at_fiducial_radius
            * exp(
                -(r - self.fiducial_radius)
                / self.radial_scale_length
            )
            * result
        )
        return spiral_value.in_(units.parsec**2 * units.Myr**-2)

    def get_gravity_at_point(self, eps, x, y, z):
        """
        Returns gravity at specified point
        Input: eps, x, y, z
        Returns fx,fy,fz
        """

        # Forces from spiral potential
        dh = 1 | units.AU  # / 1000.
        V = self.get_potential_at_point(eps, x, y, z)
        Vx = self.get_potential_at_point(eps, x+dh, y, z)
        Vy = self.get_potential_at_point(eps, x, y+dh, z)
        Vz = self.get_potential_at_point(eps, x, y, z+dh)
        fx = (V-Vx) / dh
        fy = (V-Vy) / dh
        fz = (V-Vz) / dh

        return (
            fx.in_(units.parsec * units.Myr**-2),
            fy.in_(units.parsec * units.Myr**-2),
            fz.in_(units.parsec * units.Myr**-2),
        )


class StrongSpiralArmsProfile:
    "Stronger spiral arms profile"
    def __init__(
            self,
            t_start=0 | units.Myr,
    ):
        logger.info("Using stronger spiral arms model")
        self.BasicSpiralArmsProfile = SpiralArmsProfile(t_start=t_start)

    @property
    def model_time(self):
        return self.BasicSpiralArmsProfile.model_time

    @property
    def __name__(self):
        return "Stronger spiral arms profile"

    def evolve_model(self, time):
        self.BasicSpiralArmsProfile.model_time = time

    def get_potential_at_point(self, eps, x, y, z):
        basic_spiral_value = \
            self.BasicSpiralArmsProfile.get_potential_at_point(eps, x, y, z)
        A = 0.25
        R_t = 6.2 | units.parsec
        R2 = x**2 + y**2 + z**2
        R = R2**0.5
        spiral_value = (
            (tanh(A * (R_t - R).value_in(units.kpc)) + 1)
            * basic_spiral_value
        )
        return spiral_value.in_(units.parsec**2 * units.Myr**-2)

    def get_gravity_at_point(self, eps, x, y, z):
        """
        Returns gravity at specified point
        Input: eps, x, y, z
        Returns fx,fy,fz
        """

        # Forces from spiral potential
        dh = 1 | units.AU  # / 1000.
        V = self.get_potential_at_point(eps, x, y, z)
        Vx = self.get_potential_at_point(eps, x+dh, y, z)
        Vy = self.get_potential_at_point(eps, x, y+dh, z)
        Vz = self.get_potential_at_point(eps, x, y, z+dh)
        fx = (V-Vx) / dh
        fy = (V-Vy) / dh
        fz = (V-Vz) / dh

        return (
            fx.in_(units.parsec * units.Myr**-2),
            fy.in_(units.parsec * units.Myr**-2),
            fz.in_(units.parsec * units.Myr**-2),
        )


def main():
    galaxy = StrongSpiralArmsProfile()
    x = 8 | units.kpc
    y = 0 | units.kpc
    z = 0 | units.kpc
    eps = 100 | units.parsec
    for t in range(10):
        time = t * (0.1 | units.Myr)
        galaxy.evolve_model(time)
        potential = galaxy.get_potential_at_point(eps, x, y, z)
        gravity = galaxy.get_gravity_at_point(eps, x, y, z)
        print("Time: %s" % galaxy.model_time.in_(units.Myr))
        print(
            "Potential at x= %s y= %s z= %s time= %s: %s" % (
                x, y, z, galaxy.model_time.in_(units.Myr), potential,
            )
        )
        print(
            "Gravity: %s %s %s" % (gravity[0], gravity[1], gravity[2])
        )


def plot():
    import numpy
    # import matplotlib
    import matplotlib.pyplot as plt

    # t_start = (5.0802 * 1.4874E+15 | units.s)
    t_start = (2.2 * 1.4874E+15 | units.s)
    N = 200
    xmin = -30000 | units.parsec
    xmax = 30000 | units.parsec
    ymin = -30000 | units.parsec
    ymax = 30000 | units.parsec

    # pot = TimeDependentSpiralArmsDiskModel(t_start=t_start)
    # pot = SpiralArmsProfile(t_start=t_start)
    pot = StrongSpiralArmsProfile(t_start=t_start)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = numpy.indices((N+1, N+1))
    xwidth = (xmax-xmin)
    ywidth = (ymax-ymin)
    x = xmin + xwidth*(x.flatten())/N
    y = ymin + ywidth*(y.flatten())/N
    z = x * 0.
    eps = x * 0.

    fi = pot.get_potential_at_point(eps, x, y, z).value_in(units.kms**2)
    fi = fi.reshape((N+1, N+1)).transpose()

    fiplot = ax.imshow(
        fi, origin='lower',
        extent=[
            xmin.value_in(units.parsec), xmax.value_in(units.parsec),
            ymin.value_in(units.parsec), ymax.value_in(units.parsec),
        ]
    )
    plt.colorbar(fiplot)
    # ax.set_xlim((xmin.value_in(units.parsec), xmax.value_in(units.parsec)))
    # ax.set_ylim((ymin.value_in(units.parsec), ymax.value_in(units.parsec)))
    plt.show()


if __name__ == "__main__":
    plot()
