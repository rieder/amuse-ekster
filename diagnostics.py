"Diagnostic routines"
from amuse.community.hop.interface import Hop


<<<<<<< HEAD
def identify_subgroups(
        unit_converter,
        particles,
        saddle_density_threshold=None,
        outer_density_threshold=None,
        mean_density="auto",
        ):
    "Identify groups of particles by particle densities"
    hop = Hop(unit_converter)
    hop.particles.add_particles(particles)
    hop.calculate_densities()
    if mean_density == "auto":
        mean_density = hop.particles.density.mean()
    hop.parameters.peak_density_threshold = mean_density
    hop.parameters.saddle_density_threshold = (
        0.99*mean_density
        if saddle_density_threshold is not None
        else saddle_density_threshold
    )
    hop.parameters.outer_density_threshold = (
        0.01*mean_density
        if outer_density_threshold is not None
        else outer_density_threshold
    )
    hop.do_hop()
    result = [x.get_intersecting_subset_in(particles) for x in hop.groups()]
    hop.stop()
    return result
