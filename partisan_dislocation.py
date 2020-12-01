# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph)
import geopandas as gp
import numpy as np

import matplotlib.pyplot as plt
import pickle


class AssignmentArray:
    """
    GerryChain's 'Assignment' class is optimized for working with the different
    'parts' of an assignment -- i.e. the groupings of precincts contained within
    each district. However, it is comparatively slow for lookup of individual
    precinct assignments. For that use, storing the assignment as a numpy array
    is far more effective. This class provides the way to do that as a
    gerrychain updater.
    """
    def __init__(self, alias = 'assignment_array'):
        """
        Initializes the updater.

        Parameters:
            alias (str): If used as a gerrychain updater, the alias of the
                         updater.
        """
        self.alias = alias
        self.last_flip = None

    def __call__(self, partition):
        """
        Given a partition, determines the number of counties which are divided
        into multiple districts.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            assignment (ndarray): a numpy array mpaping precinct ids to their
                                  district assignments
        """
        # Initialize the updater
        if partition.parent is None:
            return self.initialize(partition)

        # Update if necessary
        else:
            return self.update(partition)

    def initialize(self, partition):
        """
        This runs the first time the array is requested. It iterates through all
        of the precincts with a list comprehension.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            assignment_array (ndarray): a numpy array mpaping precinct ids to their district assignments
        """
        assignment = partition.assignment
        return np.array([assignment[i] for i in range(len(assignment))])

    def update(self, partition):
        """
        This is a lower-cost version designed to run when a single flip is made.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            assignment_array (ndarray): a numpy array mpaping precinct ids to their district assignments
        """
        # Get the previous info
        assignment_array = partition.parent[self.alias]
        flipped_node = list(partition.flips.keys())[0]

        # Make sure the change has been made
        assignment_array[flipped_node] = partition.flips[flipped_node]

        return assignment_array

def surrounding_precincts(g, congress="US_Distric", population="POP100"):
    """
    For each precinct, finds enough nearest neighbor precincts, such that the total population of the
    neighboring precincts is the ideal district population.

    Runs in O(n^2 log n) time.

    Parameters:
        g (GeoDataFrame): a dataFrame containing the shapefiles and populations of each precinct
        congress (str): a string identifying the field containing the congressional district in the dataframe
        population (str): a string identifying the field containing the population in the dataframe

    Returns:
        l (list): a dict mapping precinct IDs to the set of its nearest neighbor precinct IDs, such that
                the total population of the neighboring precincts is the ideal district population

    """
    # Get the number of congressional districts
    n = len(set(g[congress]))

    # Get the "ideal population" for a district
    ideal_pop = np.sum(g[population])/n

    # Make a container for the map
    l = []
    c = g.centroid

    # Iterate through each precinct
    for i in range(len(g)):

        # Rank each precinct by its centroid distance from the ith precinct. O(n log n)
        ranked = np.argsort(c.distance(c[i]))

        # Find the number of neighbor precincts necessary to get the ideal population. O(n)
        total_pop = 0
        for j, pop in enumerate(g.iloc[ranked][population]):
            total_pop += pop
            if total_pop > ideal_pop:
                k = j
                break

        # k = np.argmin(g.iloc[ranked][population].cumsum() < ideal_pop)

        # Slice off the closest k precincts, cast into a set, and store
        l.append(set(ranked[:k]))

    return l

def get_partisan_environments(g, dem_alias="DEM", rep_alias="REP", s=None):
    """
    This function takes about 3 min to run. It is the initialization for computing
    partisan displocation scores. Unless you have new data, just use the previous
    results, stored in partisan_environments.pkl

    Parameters:
        g (GeoDataFrame): geodataframe containing partisan data, populations,
                        and shapefiles of precincts
        dem_alias (str): column of geodataframe containing the Democratic vote share
        rep_alias (str): column of geodataframe containing the Republican vote share

    Returns:
        p (list): list mapping precinct IDs to the R vote share of their environment
    """
    if s is None:
        # Compute the surrounding precincts (the long part)
        s = surrounding_precincts(g)

    compositions = []

    # Iterate through the precincts
    for i, nearest in enumerate(s):
        total_dem = 0
        total_rep = 0

        for j in nearest:
            # Sum the partisan populations
            total_dem += g.iloc[j][dem_alias]
            total_rep += g.iloc[j][rep_alias]

        compositions.append(total_rep/(total_rep + total_dem))

    # pickle.dump(compositions, open("partisan_environments1.pkl", 'wb'))
    return np.array(compositions)

def partisan_dislocation_score(election_results, assignment_array, partisan_environments, populations, mode = "avgabsolute"):
    """
    Returns the average absolute partisan dislocation.

    Parameters:
        election_results (gerrychain ElectionResults object): Election results from gerrychain, accessed using partition['<election_name>']
        assignment_array (ndarray): Assignment mapping precincts to districts (partition['assignment_array'])
        partisan_environments (ndarray): ndarray mapping precinct ids to partisan environments
        populations (ndarray): Ndarray mapping precinct ids to precinct populations
        mode (str): Method of calculating partisan dislocation. For now, the only implemented method is 'avgabsolute', which takes the  population-weighted average
            of the absolute partisan dislocation in each precinct

    Returns:
        pd (float): Average absolute partisan dislocation
    """
    # Calculate R vote shares in an ndarray
    shares = np.array([election_results.percent('Rep', n) for n in np.arange(len(election_results.races))], dtype=np.float64)

    # Calculate partisan displacment in each precinct
    partisan_dislocation = shares[assignment_array] - partisan_environments

    if mode == 'avgabsolute':
        # Calculate average absolute partisan dislocation, weighting by population
        return np.mean(np.abs(partisan_dislocation * populations))/np.mean(populations)

    else:
        raise NotImplementedError("Other modes are not yet implemented.")

def graph_partisan_dislocation(assignment, precincts='UtahData/18_Precincts_combined_contiguous-Copy1.shp', graph = 'graph_combined_vs_2018.json', environment='UtahData/partisan_environments_combined.pkl', dem_alias = 'COMB_DEM', rep_alias = 'COMB_REP', cmap='bwr', size=(3,2), dpi=300, save=False, savetitle=None):
    """
    Given a districting assignment, plots the partisan dislocation of the state of Utah.

    Parameters:
        assignment (gerrychain.Assignment): a districting assignment. A dictionary will work as long as it has the right keys
        precincts (str): filename of precinct shapefiles. Alternatively, pass them in directly
        graph (str): filename of graph. Alternatively, pass in directly
        environment (str): filename of an ndarray of the partisan environments in Utah. Alternatively, pass them in directly
        cmap: The matplotlib cmap used in plotting.
        size ((2,) tuple): figure size
        dpi (int): figure resolution

    """
    # Load in the partisan data (if necessary)
    if type(environment) == str:
        environment = pickle.load(open(environment, 'rb'))

    # Load in the shapefiles
    if type(precincts) == str:
        precincts = gp.read_file(precincts)
    else:
        precincts = precincts.copy()

    # Load in the graph
    if type(graph) == str:
        graph = Graph.from_json(graph)

    # Make a container
    district_vote_shares = np.zeros((4,2), dtype=np.float64)

    # Iterate through the assignment
    for i, dist in assignment.items():
        district_vote_shares[dist, 0] += graph.nodes[i]['DEM']
        district_vote_shares[dist, 1] += graph.nodes[i]['REP']

    shares = district_vote_shares[:, 1]/district_vote_shares.sum(axis=1)

    districts = [assignment[i] for i in range(len(assignment))]

    precincts['partisan_dislocation'] = shares[districts] - environment

    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    precincts.plot(column='partisan_dislocation', ax=ax, cmap=cmap)
    plt.axis("off")

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches="tight")
    plt.show()
