import numpy
from amuse.units import units, nbody_system
from amuse.datamodel import Particle, Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.units.trigo import sin, cos
# from amuse_masc import new_cluster

def new_kroupa_mass_distribution(
        number_of_particles,
        mass_min=None,
        mass_max=None,
        random=True,
):
    """
    Returns Kroupa (2001) mass distribution
    Modified from amuse.ic.brokenimf version - discard masses below mass_min
    """
    masses = [] | units.MSun
    while len(masses) < number_of_particles:
        next_mass = MultiplePartIMF(
            mass_boundaries = [0.01, 0.08, 0.5, 1000.0] | units.MSun,
            mass_max = mass_max,
            alphas = [-0.3, -1.3, -2.3],
            random=random,
        ).next_mass(1)
        if mass_min is None:
            masses.append(next_mass)
        elif next_mass >= mass_min:
            masses.append(next_mass)
    return masses


class StarFormingRegion(
        # object,
        Particle,
):
    """
    # Makes a Particle into a Star-forming region.
    Creates a Star-forming region.
    Should be initialised after a sink has finished its initial accretion since
    otherwise the expected mass will be off.
    """

#     def __init__(
#             self,
#             # # star_distribution="plummer",
#             # binary_fraction=0,
#             # triple_fraction=0,
#             # maximum_mass=500 | units.MSun,
#     ):
#         #Particle.__init__()
# 
#         #Particle.__init__()
#         print("Initted")
    def __init__(
            self,
            key=None,
            particles_set=None,
            set_index=None,
            set_version=-1,
            mass=0 | units.MSun,
            radius=0 | units.AU,
            position=[0,0,0] | units.parsec,
            velocity=[0,0,0] | units.kms,
            initial_mass_function="kroupa",
            binary_fraction=0,
            triple_fraction=0,
            upper_mass_limit=100 | units.MSun,
            **keyword_arguments):
        if particles_set is None:
            if key == None:
                particles_set = Particles(1)
                key = particles_set.get_all_keys_in_store()[0]
            else:
                particles_set = Particles(1, keys = [key])

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "particles_set", particles_set)
        object.__setattr__(self, "_set_index", set_index)
        object.__setattr__(self, "_set_version", set_version)

        for attribute_name in keyword_arguments:
            attribute_value = keyword_arguments[attribute_name]
            setattr(self, attribute_name, attribute_value)
        self.triple_fraction = triple_fraction
        self.mass = mass
        self.radius = radius
        self.position = position
        self.velocity = velocity
        self.initial_mass_function = initial_mass_function
        self.binary_fraction = binary_fraction
        self.upper_mass_limit = upper_mass_limit
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
            number_of_stars = len(self.next_mass)
            # converter = nbody_system.nbody_to_si(
            #     self.mass,
            #     self.radius,
            # )
            # new_stars = new_plummer_distribution(converter, len(self.next_mass))
            new_stars = Particles(number_of_stars)
            new_stars.mass = self.next_mass
            new_stars.position = self.position
            new_stars.velocity = self.velocity

            # Random position within the sink radius
            radius = self.radius
            rho = numpy.random.random(number_of_stars) * radius
            theta = numpy.random.random(number_of_stars) * 2 * numpy.pi | units.rad
            phi = numpy.random.random(number_of_stars) * numpy.pi | units.rad
            x = rho * sin(phi) * cos(theta)
            y = rho * sin(phi) * sin(theta)
            z = rho * cos(phi)
            new_stars.x += x
            new_stars.y += y
            new_stars.z += z

            # Random velocity, sample magnitude from gaussian with local sound speed
            # like Wall et al (2019)
            local_sound_speed = 1 | units.kms
            # TODO: do this properly - see e.g. formula 5.17 in AMUSE book
            velocity_magnitude = numpy.random.normal(
                # loc=0.0,  # <- since we already added the velocity of the sink
                scale=local_sound_speed.value_in(units.kms),
                size=number_of_stars,
            ) | units.kms
            velocity_theta = numpy.random.random(number_of_stars) * 2 * numpy.pi | units.rad
            velocity_phi = numpy.random.random(number_of_stars) * numpy.pi | units.rad
            vx = velocity_magnitude * sin(velocity_phi) * cos(velocity_theta)
            vy = velocity_magnitude * sin(velocity_phi) * sin(velocity_theta)
            vz = velocity_magnitude * cos(velocity_phi)
            new_stars.vx += vx
            new_stars.vy += vy
            new_stars.vz += vz

            new_stars.origin_cloud = self.key
            
            # Make sure quantities are (mostly) conserved?
            self.mass -= new_stars.total_mass()
            
            # Determine which star(s) should form next
            self.generate_next_mass()
            return new_stars
        return False
