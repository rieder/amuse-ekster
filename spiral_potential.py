"""
Calculate the spiral arm potential
"""
from __future__ import print_function, division

from numpy import (pi, sin, cos, tan, arctan, cosh, sqrt, log, exp)

from amuse.units import units
from amuse.units.constants import G

from amuse.support.literature import LiteratureReferencesMixIn


class TimeDependentSpiralArmsDiskModel(LiteratureReferencesMixIn):
    """
    Spiral arms + disk model
    """

    def __init__(
            self,
            t_start=0 | units.yr,
            r0=2.47518e22 | units.cm,
            NN=2,
            alpha=0.261799,
            Rs=2.16578e22 | units.cm,
            rho0=2.12889e-24 | units.g * units.cm**-3,
            phir=6.3e-16 | units.s**-1,
            Hz=5.56916e20 | units.cm,
            Co=2.31e14 | units.cm**2 * units.s**-2,
            Rc=3.09398e21 | units.cm,
            zq=0.7,
    ):
        LiteratureReferencesMixIn.__init__(self)
        self.model_time = t_start
        # Parameters for spiral potential
        self.fiducial_radius = r0
        self.number_of_arms = NN
        self.pitch_angle = alpha
        self.radial_scale_length = Rs
        self.density_at_fiducial_radius = rho0
        self.phir = phir
        self.scale_height = Hz
        self.Co = Co
        self.Rc = Rc
        self.zq = zq

        self.disk = LogarithmicDisk_profile(
            Co=self.Co,
            Rc=self.Rc,
            zq=self.zq,
        )
        self.spiralarms = SpiralArms_profile(
            t_start=self.model_time,
            fiducial_radius=self.fiducial_radius,
            number_of_arms=self.number_of_arms,
            pitch_angle=self.pitch_angle,
            radial_scale_length=self.radial_scale_length,
            density_at_fiducial_radius=self.density_at_fiducial_radius,
            phir=self.phir,
            scale_height=self.scale_height,
        )

    @property
    def __name__(self):
        return "%s with %s" % (
            self.disk.__name__,
            self.spiralarms.__name__,
            )

    def evolve_model(self, time):
        self.disk.evolve_model(time)
        self.spiralarms.evolve_model(time)
        self.model_time = min(self.disk.model_time, self.spiralarms.model_time)

    def get_gravity_at_point(self, eps, x, y, z):
        disk_force = self.disk.get_gravity_at_point(eps, x, y, z)
        spiral_force = self.spiralarms.get_gravity_at_point(eps, x, y, z)
        f = []
        for i in range(3):
            f.append(disk_force[i] + spiral_force[i])
        return f

    def get_potential_at_point(self, eps, x, y, z):
        return(
            self.disk.get_potential_at_point(eps, x, y, z)
            + self.spiralarms.get_potential_at_point(eps, x, y, z)
        )

    def vel_circ(self, r):
        return -1

    def stop(self):
        return


class LogarithmicDisk_profile(LiteratureReferencesMixIn):
    """
    Logarithmic potential

    .. [#] Binney & Tremaine

    """

    def __init__(
            self,
            Co=2.31e14 | units.cm**2 * units.s**-2,
            Rc=3.09398e21 | units.cm,
            zq=0.7,
    ):
        LiteratureReferencesMixIn.__init__(self)

        self.model_time = 0 | units.Myr
        # Parameters for logarithmic potential (e.g. Binney & Tremaine)
        self.Co = Co  # speed squared and constants
        self.Rc = Rc  # distance
        self.zq = zq  # dimensionless

    @property
    def __name__(self):
        return "Logarithmic profile"

    def evolve_model(self, time):
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


class SpiralArms_profile(LiteratureReferencesMixIn):
    """
    Spiral arms potential model

    .. [#] Cox & Gomez (2002)

    """

    def __init__(
            self,
            t_start=0 | units.Myr,
            fiducial_radius=2.47518e22 | units.cm,
            number_of_arms=2,
            pitch_angle=0.261799,
            radial_scale_length=2.16578e22 | units.cm,
            density_at_fiducial_radius=2.12889e-24 | units.g * units.cm**-3,
            phir=6.3e-16 | units.s**-1,
            scale_height=5.56916e20 | units.cm,
    ):
        LiteratureReferencesMixIn.__init__(self)
        self.time_initial = t_start
        self.model_time = 0 | units.Myr
        self.fiducial_radius = fiducial_radius
        self.number_of_arms = number_of_arms
        self.pitch_angle = pitch_angle
        self.radial_scale_length = radial_scale_length
        self.density_at_fiducial_radius = density_at_fiducial_radius
        self.phir = phir
        self.scale_height = scale_height
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


class SpiralArmsModel(LiteratureReferencesMixIn):
    """
    Spiral arms potential model

    .. [#] Cox & Gomez (2002)

    """

    def __init__(
            self,
            t_start=0 | units.Myr,
            fiducial_radius=2.47518e22 | units.cm,
            number_of_arms=2,
            pitch_angle=0.261799,
            radial_scale_length=2.16578e22 | units.cm,
            density_at_fiducial_radius=2.12889e-24 | units.g * units.cm**-3,
            phir=6.3e-16 | units.s**-1,
            scale_height=5.56916e20 | units.cm,
    ):
        LiteratureReferencesMixIn.__init__(self)
        # Parameters for spiral potential
        # r0
        self.fiducial_radius = 2.47518e22 | units.cm
        # NN
        self.number_of_arms = 2
        # alpha
        self.pitch_angle = 0.261799
        # rS
        self.radial_scale_length = 2.16578e22 | units.cm
        # p0
        self.density_at_fiducial_radius = 2.12889e-24 | units.g * units.cm**-3
        # phir
        self.phir = 6.3e-16 | units.s**-1
        # Hz
        self.scale_height = 5.56916e20 | units.cm
        # Cz
        self.Cz = [
            8 / (3*pi),
            0.5,
            8 / (15*pi)
        ]

        # Parameters for logarithmic potential (e.g. Binney & Tremaine)
        self.Co = 2.31e14 | units.cm**2 * units.s**-2
        self.Rc = 3.09398e21 | units.cm
        self.zq = 0.7

        # Time at initial conditions (~100Myr)
        # t0
        self.time_initial = 3.153e15 | units.s
        self.model_time = time

    def evolve_model(self, time):
        """Set current time of the model"""
        self.model_time = time

    def get_potential_at_point(self, eps, x, y, z):
        """
        Calculate the spiral arm potential at specified point

        from Cox & Gomez 2002
        Input: eps, x, y, z
        Returns: potential
        """

        phi = arctan(y/x)
        if x < (0 | units.parsec):
            phi = pi + phi

        # d2
        r = sqrt(x**2+y**2)

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
            Bn = Kn * self.scale_height * (1. + 0.4 * Kn * self.scale_height)
            Dn = (
                1. + Kn * self.scale_height + 0.3 *
                (Kn * self.scale_height)**2.
            ) / (1. + 0.3 * Kn * self.scale_height)

            result = (
                result
                + (self.Cz[n] / (Dn * Kn))
                * cos((n+1) * gamma)
                * (1 / cosh((Kn*z) / Bn))**Bn
            )

        spiral_value = (
            -4. * pi * G
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

        d2 = (x**2 + y**2)

        # Forces from logarithmic potential
        fx = -2. * self.Co * x / (self.Rc**2 + d2 + (z/self.zq)**2)
        fy = -2. * self.Co * y / (self.Rc**2 + d2 + (z/self.zq)**2)
        fz = -2. * self.Co * z / (
            (self.Rc**2 + d2 + (z/self.zq)**2) * self.zq**2
        )

        # Forces from spiral potential
        dh = eps / 1000.

        V = self.get_potential_at_point(eps, x, y, z)

        Vx = self.get_potential_at_point(eps, x+dh, y, z)
        fx = fx - (Vx-V) / dh

        Vy = self.get_potential_at_point(eps, x, y+dh, z)
        fy = fy - (Vy-V) / dh

        Vz = self.get_potential_at_point(eps, x, y, z+dh)
        fz = fz - (Vz-V) / dh

        return (
            fx.in_(units.parsec*units.Myr**-2),
            fy.in_(units.parsec*units.Myr**-2),
            fz.in_(units.parsec*units.Myr**-2),
        )

    def enclosed_mass(self, r):
        return -1

    def circular_velocity(self, r):
        return -1


def main():
    galaxy = SpiralArmsModel()
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


if __name__ == "__main__":
    main()
