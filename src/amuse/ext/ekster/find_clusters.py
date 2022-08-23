#!/usr/bin/env python3
"""
Read a set of stars and return individual groups.
"""

from amuse.datamodel import Particles
from amuse.units import units, nbody_system
from amuse.community.hop.interface import Hop


def match_clusters(
    groups,
    groups_to_match,
    cluster_cores,
    time=None,
    time_to_match=None,
    match_level=0.7,
    distance_criterion=1.0 | units.parsec,
    cluster_labels=None
):
    # snapshot = groups[0].snapshot
    # snapshot_to_match = groups_to_match.snapshot
    for i, group in enumerate(groups):
        com = group.center_of_mass()
        core_id = group[0].core_id
        for j, group_to_match in enumerate(groups_to_match):
            com_to_match = group_to_match.center_of_mass()
            if time is not None and time_to_match is not None:
                comv_to_match = group_to_match.center_of_mass_velocity()
                match_position = com_to_match + comv_to_match * (time-time_to_match)
            else:
                match_position = com_to_match

            # print("position - expected position: %s" % (com - match_position).length())
            if (com - match_position).length() < distance_criterion:
                # print("Possible match")
                overlapping_stars = group_to_match.get_intersecting_subset_in(group)
                fraction = len(overlapping_stars) / len(group)
                # print("%4.1f percent overlap" % (100 * fraction))
                if fraction > match_level:
                    core_id_to_match = group_to_match[0].core_id
                    print("Match found - cluster %i with cluster %i" % (core_id, core_id_to_match))
                    # print("Cluster %i evolved into cluster %i")
                    cluster_cores[core_id].next = cluster_cores[core_id_to_match].key
                    cluster_cores[core_id_to_match].previous = cluster_cores[core_id].key
                    break


def find_clusters(
        stars,
        convert_nbody=None,
        scale_mass=1 | units.MSun,
        scale_radius=1 | units.parsec,
        mean_density=None,
    ):
    if convert_nbody == None:
        convert_nbody = nbody_system.nbody_to_si(scale_mass, scale_radius)
    groupfinder = Hop(convert_nbody)
    groupfinder.particles.add_particles(stars)
    groupfinder.calculate_densities()

    if mean_density is None:
        mean_density = groupfinder.particles.density.mean()

    groupfinder.parameters.peak_density_threshold = 10*mean_density
    groupfinder.parameters.saddle_density_threshold = 0.99*mean_density
    groupfinder.parameters.outer_density_threshold = 0.01*mean_density

    print("Mean density: %s" % mean_density)
    groupfinder.do_hop()
    result = [x.get_intersecting_subset_in(stars) for x in groupfinder.groups()]
    groupfinder.stop()
    return result


def main():
    import sys
    from amuse.io import read_set_from_file, write_set_to_file
    from amuse.support.console import set_preferred_units

    set_preferred_units(units.parsec, units.MSun, units.kms)

    converter = nbody_system.nbody_to_si(
        100 | units.MSun,
        1 | units.parsec,
    )

    # TODO: set a fixed density for Hop across snapshots

    # snapshot_numbers = []
    # snapshot_clusters = []
    cluster_cores = Particles()
    start_cluster_counter = 0
    for i, snapshot in enumerate(sys.argv[1:]):
        print("Reading snapshot %s" % snapshot)
        stars = read_set_from_file(snapshot, 'amuse', close_file=True)
        time = stars.get_timestamp()
        snapnum = snapshot.split('.')[0].split('-')[1]
        # snapshot_numbers.append(snapnum)
        # clusters = find_clusters(stars, mean_density=100000 | units.MSun * units.parsec**-3)
        clusters = find_clusters(stars, mean_density=None)
        print("Found %i clusters" % len(clusters))
        for j, cluster in enumerate(clusters):
            core = cluster.new_particle_from_cluster_core(converter)
            core.time = time
            lagrangian = cluster.LagrangianRadii(converter)
            core.lr100 = lagrangian[0][-1]
            core.lr90 = lagrangian[0][-2]
            core.lr75 = lagrangian[0][-3]
            core.lr50 = lagrangian[0][-4]
            core.lr20 = lagrangian[0][-5]
            core.lr10 = lagrangian[0][-6]
            core.lr05 = lagrangian[0][-7]
            core.lr02 = lagrangian[0][-8]
            core.lr01 = lagrangian[0][-9]
            core.mass = cluster.mass.sum()
            core.clusternumber = j
            core.snapshotnumber = snapnum
            cluster_cores.add_particle(core)
            cluster.core_id = len(cluster_cores) - 1
            # print("Cluster core id: %i" % cluster[0].core_id)
        if i > 0:
            match_clusters(
                clusters, clusters_to_match,
                cluster_cores,
                time=time, time_to_match=time_to_match,
            )
        clusters_to_match = clusters
        stars_to_match = stars
        i_to_match = i
        time_to_match = time
        start_cluster_counter += len(clusters)
    write_set_to_file(cluster_cores, "cluster_cores_strong.amuse", "amuse", append_to_file=False)
    exit()

    print("Found %i clusters:" % len(clusters))
    for i, cluster in enumerate(clusters):
        snap = sys.argv[1].split('.')[0].split('-')[1]
        print(
            "fresco.py -s %s -o cluster-%03i -a 0 -w %f --xo %f --yo %f --zo %f" % (
                sys.argv[1],
                i,
                5.0,
                cluster.center_of_mass().x.value_in(units.parsec),
                cluster.center_of_mass().y.value_in(units.parsec),
                cluster.center_of_mass().z.value_in(units.parsec),
            )
        )

    for i, cluster in enumerate(clusters):
        snap = sys.argv[1].split('.')[0].split('-')[1]
        print(
            "python plotting_class.py -g gas-%s.hdf5 -s stars-%s.hdf5 -x %f -y %f -w 50 -n 100 -o zoom-%03i" % (
                snap,
                snap,
                cluster.center_of_mass().x.value_in(units.parsec),
                cluster.center_of_mass().y.value_in(units.parsec),
                i,
            )
        )
    return


if __name__ == "__main__":
    main()
