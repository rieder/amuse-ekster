import numpy
from amuse.units import units, nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_distribution
from amuse.ic.brokenimf import new_kroupa_mass_distribution
# from amuse_masc import new_cluster

class StarFormingRegion(
        Particle,
):
    """
    Makes a Particle into a Star-forming region.
    Should be initialised after a sink has finished its initial accretion since
    otherwise the expected mass will be off.
    """

    def __init__(
            self,
            initial_mass,
            initial_radius,
            initial_mass_function="kroupa",
            star_distribution="plummer",
            binary_fraction=0,
            triple_fraction=0,
            upper_mass_limit=100 | units.MSun,
            maximum_mass=500 | units.MSun,
    ):
        Particle.__init__()
        self.mass = initial_mass
        self.radius = initial_radius
        self.star_distribution = star_distribution
        self.initial_mass_function = initial_mass_function
        self.binary_fraction = binary_fraction
        self.triple_fraction = triple_fraction

        self.generate_next_mass()

    def generate_next_mass(self):
        """
        Generate the next (set of) stellar mass(es) that will be formed in this
        region.
        Assumes that 'radius' will be the 'orbit' that a newly formed star will
        be on around the centre of mass of this region.
        """
        rnd = numpy.random.random()
        is_triple = False
        is_binary = False
        if rnd < self.triple_fraction:
            is_triple = True
        elif rnd < self.triple_fraction + self.binary_fraction:
            is_binary = True

        if not (is_binary or is_triple):
            n = 1
        elif is_binary:
            n = 2
        elif is_triple:
            n = 3

        self.next_mass = new_kroupa_mass_distribution(
            n,
            mass_max=self.upper_mass_limit,
        )

    def yield_next(self):
        """
        Determine if (a) new star(s) can now be formed, and if so, return these
        """
        if self.mass >= self.next_mass.sum():
            converter = nbody_system.nbody_to_si(
                self.mass,
                self.radius,
            )
            new_stars = new_plummer_distribution(converter, len(self.next_mass))
            new_stars.mass = self.next_mass
            new_stars.position += self.position
            new_stars.velocity += self.velocity
            new_stars.origin_cloud = self.key
            self.mass -= new_stars.total_mass()
            self.generate_next()

            # Make sure quantities are (mostly) conserved!


            return new_stars
        return False
