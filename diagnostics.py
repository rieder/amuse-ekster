from amuse.community.hop.interface import Hop


def identify_subgroups(unit_converter, particles):
    hop = Hop(unit_converter)
    hop.particles.add_particles(particles)
    hop.calculate_densities()
    mean_density = hop.particles.density.mean()
    hop.parameters.peak_density_threshold = mean_density
    hop.parameters.saddle_density_threshold = 0.99*mean_density
    hop.parameters.outer_density_threshold = 0.01*mean_density
    hop.do_hop()
    result = [x.get_intersecting_subset_in(particles) for x in hop.groups()]
    hop.stop()
    return result
