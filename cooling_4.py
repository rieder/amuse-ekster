"""
Cooling functions
"""

from numpy import sqrt, exp
from amuse.units import units, constants


def u_to_temperature(
        internal_energy,
        # gas_mean_molecular_weight=(2.33 / 6.02214179e+23) | units.g,
        gas_mean_molecular_weight=None,
):
    if gas_mean_molecular_weight is None:
        gas_mean_molecular_weight = (((1.0)+0.4) / (0.1+(1.)) / 6.02214179e+23) | units.g
        # print("GMMW = %s" % gas_mean_molecular_weight.value_in(units.g))
    # temperature = (
    #     internal_energy * (2.0 * gas_mean_molecular_weight)
    #     / (3.0 * constants.kB)
    # )
    temperature = (
        2/3*internal_energy/(constants.kB/gas_mean_molecular_weight)
    )
    # Rg = (constants.kB * 6.02214076e23).value_in(units.erg * units.K**-1)
    # gmwvar = 1.2727272727
    # uergg = 6.6720409999999996E-8

    # temperature = (
    #     2.0/3.0*internal_energy.value_in(units.kms**2) / (Rg/gmwvar/uergg)
    # ) | units.K
    return temperature


def cool(particles):

    temp = u_to_temperature(particles.u)
    # The cooling rate in erg cm^3/s = g cm^5/s^3
    Gamma = 2.0e-26 | units.erg * units.s**-1
    crate = (
        Gamma
        * ((
            1.e7 * exp(
                -1.18400e5
                / (
                    temp.value_in(units.K) + 1000
                )
            )
            + (
                1.4e-2
                * sqrt(temp.value_in(units.K))
                * exp(-92. / temp.value_in(units.K))
            )
        ) | units.cm**3)
    )
    
    # units are now cm^5/(g s^3) ! since [u] = erg/g = cm^2/s^2
    crate = crate/constants.proton_mass**2
    # multiply by rho (code) to get l^5/(m t^3) * m/l^3 = l^2/s^3 = [u[
    return -crate * particles.density

