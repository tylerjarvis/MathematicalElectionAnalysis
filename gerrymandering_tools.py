# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, grid)
from gerrychain.metrics import mean_median, partisan_bias, polsby_popper, efficiency_gap, partisan_gini
import pandas as pd
import geopandas as gp
import numpy as np
from scipy import optimize as opt
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import itertools

from tqdm.auto import tqdm
import pickle
import inspect

# COMPUTING COUNTY SPLITS -----------------------------------------------------

utah_counties = {1: "Beaver",
           2: "Box Elder",
           3: "Cache",
           4: "Carbon",
           5: "Daggett",
           6: "Davis",
           7: "Duchesne",
           8: "Emery",
           9: "Garfield",
           10: "Grand",
           11: "Iron",
           12: "Juab",
           13: "Kane",
           14: "Millard",
           15: "Morgan",
           16: "Piute",
           17: "Rich",
           18: "Salt Lake",
           19: "San Juan",
           20: "Sanpete",
           21: "Sevier",
           22: "Summit",
           23: "Tooele",
           24: "Uintah",
           25: "Utah",
           26: "Wasatch",
           27: "Washington",
           28: "Wayne",
           29: "Weber"}

# Metrics for scoring plans

class SplitCountiesSimple:
    """
    Given a partition, determines the number of counties which are divided into multiple districts.
    Use this class if you simply want to know the integer number of split counties.

    Parameters:
        partition: Gerrychain GeographicPartition object

    Returns:
        splits (int): the number of split counties
    """
    def __init__(self, alias =  "split_counties_simple", county_field = "CountyID"):
        """
        Accepts a string which denotes the field where the county info is stored in the graph
        """
        self.f = "CountyID"
        self.alias = alias

    def __call__(self, partition):
        """
        Given a partition, determines the number of counties which are divided into multiple districts.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            splits (int)
        """
        # Initialize the updater
        if partition.parent is None:
            return self.initialize(partition)

        # Update if necessary
        else:
            return self.update(partition)

    def initialize(self, partition):
        """
        This runs the first time the number of county splits is requested. It iterates through all
        of the cut edges to determine the number of county splits.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            splits (int): the number of split counties
        """

        split_counties = {}

        # Iterate through all the cut edges
        for edge in partition["cut_edges"]:

            # Determine if the edge splits a county
            county1 = initial_partition.graph.nodes[edge[0]][self.f]
            county2 = initial_partition.graph.nodes[edge[1]][self.f]

            if county1 == county2:
                # There is a cut edge inside the county
                split_counties[county1] = split_counties.get(county1, 0) + 1

        self.split_counties = split_counties

        return sum([1 for key in split_counties if split_counties[key] > 0])

    def update(self, partition):
        """
        This is a lower-cost version designed to run when a single flip is made. It updates
        the previous count based on the flip information.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            splits (int): the number of split counties
        """
        # Get the previous number of split counties
        old = partition.parent[self.alias]

        # Get the flip information
        node = list(partition.flips.keys())[0]
        new_assignment = partition.flips[node]
        old_assignment = partition.parent.assignment[node]

        # Track

        flow = 0
        county = partition.graph.nodes[node][self.f]

        # Iterate through all the neighbors
        for neighbor in partition.graph[node]:

            neighbor_county = partition.graph.nodes[neighbor][self.f]

            # Iterate through all neighbors which are in the same county
            if county == neighbor_county:

                neighbor_assignment = partition.assignment[neighbor]

                if neighbor_assignment == new_assignment:
                    # The county was split but now it is not
                    flow -= 1

                elif neighbor_assignment == old_assignment:
                    # The county wasn't split but now it is
                    flow += 1

        return old + flow

class SplitCounties:
    """
    Given a partition, determines the number of counties which are divided into multiple districts.
    This method has a more useful output that allows you to calculate the Mattingly score for county splits.

    Parameters:
        partition: Gerrychain GeographicPartition object

    Returns:
        split_counties (dict): A dict mapping counties to another dict,
                                    which maps districts to the number of precincts
                                    in that county which fall in that district.
    """

    def __init__(self, alias =  "split_counties", county_field = "CountyID", district_field = "US_Distric", start_at_one=False):
        """
        Accepts a string which denotes the field where the county info is stored in the graph.

        Parameters:
            alias (str): When used as an updater function in gerrychain, the alias
            county_field (str): the node attribute in the graph which stores the county information
            district_field (str): the node attribute in the graph which stores the district information
            start_at_one (False): whether or not the county numbering starts at one
        """
        self.c = county_field
        self.d = district_field
        self.alias = alias
        self.last_flip = None
        self.start_at_one = start_at_one

    def __call__(self, partition):
        """
        Given a partition, determines the number of counties which are divided into multiple districts.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            split_counties (dict): A dict mapping counties to another dict,
                                    which maps districts to the number of precincts
                                    in that county which fall in that district.
        """
        # Initialize the updater
        if partition.parent is None:
            return self.initialize(partition)

        # Update if necessary
        else:
            return self.update(partition)

    def initialize(self, partition):
        """
        This runs the first time the number of county splits is requested. It iterates through all
        of the cut edges to determine the number of county splits.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            split_counties (dict): A dict mapping counties to another dict,
                                    which maps districts to the number of precincts
                                    in that county which fall in that district.
        """
        # Set parameters
        num_districts = len(partition.parts)
        county_content = {}

        # Iterate through all the nodes in the graph
        for node in partition.graph.nodes:

            # Store the node's information
            county = partition.graph.nodes[node][self.c]
            district = partition.assignment[node]

            # If the county isn't stored yet, store it
            if county not in county_content:
                if not self.start_at_one:
                    county_content[county] = {i: 0 for i in range(num_districts)}
                else:
                    county_content[county] = {i: 0 for i in range(1,num_districts+1)}

            # Update the totals
            county_content[county][district] += 1

        return county_content


    def update(self, partition):
        """
        This is a lower-cost version designed to run when a single flip is made. It updates
        the previous count based on the flip information.

        Parameters:
            partition: Gerrychain GeographicPartition object

        Returns:
            split_counties (dict): A dict mapping counties to another dict,
                                    which maps districts to the number of precincts
                                    in that county which fall in that district.
        """
        # Get the previous info
        county_content = partition.parent[self.alias]

        # Check to see if the last flip worked
        if self.last_flip != partition.flips and self.last_flip is not None:
            flipped_node = list(self.last_flip.keys())[0]
            new_district = self.last_flip[flipped_node]

            if partition.assignment[flipped_node] != new_district:

                # The flip wasn't actually carried out. We need to correct it
                county = partition.graph.nodes[flipped_node][self.c]
                old_district = partition.parent.assignment[flipped_node]
                county_content[county][new_district] -= 1
                county_content[county][old_district] += 1

        if self.last_flip != partition.flips:
            # Get the flip information
            flipped_node = list(partition.flips.keys())[0]
            county = partition.graph.nodes[flipped_node][self.c]
            new_district = partition.assignment[flipped_node]
            old_district = partition.parent.assignment[flipped_node]

            county_content[county][new_district] += 1
            county_content[county][old_district] -= 1

        #county_content_true = self.initialize(partition)

        #if county_content_true != county_content:
        #    print("Last flip:")
        #    print(self.last_flip)
        #    print("Current flip:")
        #    print(partition.flips)
        #    print("Computed county content:")
        #    print(county_content)
        #    print("Real county content:")
        #    print(county_content_true)
        #
        #    raise ValueError

        self.last_flip = partition.flips

        return county_content

def county_splits_score(county_content, three_penalty=100, start_at_one=False, mode="mattingly"):
    """
    Calculates a county splits score for a district based on county splits data.

    Parameters:
        county_content: output from SplitCounties function (dict of dicts)
        mode (str): a keyword defining which calculation mode to use
            "simple": returns the number of split counties, an integer
            "mattingly": uses Mattingly's score function for split counties
        start_at_one (bool): whether or not the district numbering starts at one

    Returns:
        score (float): the calculated fitness score (lower is better)
    """
    # Set Parameters
    two_split_counties = {}
    three_split_counties = {}
    num_districts = len(county_content[1])

    # Iterate through all the counties
    for county, districts in county_content.items():

        # Set counters
        zeros = 0
        nonzero_districts = []

        # Iterate through districts

        for i in range(start_at_one, start_at_one + num_districts):
            if districts[i] == 0:
                zeros += 1
            else:
                nonzero_districts.append(i)

        # Determine nature of split
        if zeros == num_districts - 2:
            # County is split two ways
            two_split_counties[county] = nonzero_districts

        elif zeros <= num_districts - 3:
            # County is split three ways
            three_split_counties[county] = nonzero_districts

            # We assume that counties will rarely be split > 3 times.
            # If so, they fall into this category as well.

    # Find the number of splits
    num_two_splits = len(two_split_counties)
    num_three_splits = len(three_split_counties)


    if mode == "simple":
        return num_two_splits + num_three_splits

    # For the twice-split counties:
    # Sum the proportion of each county in the 2nd largest district
    two_proportion_score = 0
    for county, districts in two_split_counties.items():

        district1 = county_content[county][districts[0]]
        district2 = county_content[county][districts[1]]

        # Find the 2nd largest district by number of precincts
        try:
            two_proportion_score += np.sqrt(min(district1, district2)/(district1+district2))
        except FloatingPointError:
            print("These are the district populations:" )
            print(district1, district2)
            two_proportion_score = 0


    # For the 3x-split counties:
    # Sum the proportion of each county in the 3rd largest district
    three_proportion_score = 0
    for county, districts in three_split_counties.items():

        district1 = county_content[county][districts[0]]
        district2 = county_content[county][districts[1]]
        district3 = county_content[county][districts[2]]

        # Of the three districts, find the district with the fewest precincts
        try:
            three_proportion_score += np.sqrt(min(district1, district2, district3)/(district1+district2+district3))
        except FloatingPointError:
            print("These are the district populations:" )
            print(district1, district2, district3)
            three_proportion_score = 0


    if mode == "mattingly":

        # Calculate the score with Mattingly's method
        return num_two_splits * two_proportion_score + three_penalty * num_three_splits * three_proportion_score

        # In Mattingly's method, we impose a greater penalty for triply-split counties, weighted by three_penalty
        # We also assume that counties will rarely be split more than 3 times.
        # If so, they fall into the same bucket as the triply-split counties


# COMPUTING PARTISAN DISLOCATION -----------------------------------------------

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

def get_partisan_environments(g, dem_alias="DEM", rep_alias="REP"):
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

# Class for custom acceptance

class CustomAccept:
    """
    A Metropolis-Hastings acceptance function for GerryChain. This acceptance
    function tends to reward more 'fit' plans, where fitness is defined by a
    weighted sum of various score functions on partitions. By default, the
    following score functions are included in the weight function:
    - Compactness ('cut_edges')
    - Population equality ('pop_mattingly')

    We include these by default because in the absence of constraints on
    population equality and compactness, the plans produced by Gerrychain
    proposals will not be good samples from the distribution of appropriate
    districting maps. Additionallly, the user can choose to include any of the
    following score functions in the fitness function:

    - Other methods for scoring population equality:
        - using the max-min of district populations ('pop_max-min')

    - Custom score functions, specified by the 'custom' argument, which must
      accept a GerryChain partition and return a nonnegative float (where 0
      indicates the most fit partition).

    If the chain has an election updater, specified with 'election' argument:
    - Any of the metrics already included in GerryChain (partisan_bias,
      partisan_gini, efficiency_gap, mean_median, wins)
    - Partisan dislocation ('partisan_dislocation') (speed up by using the
      AssignmentArray updater)

    If the chain has a CountySplits or CountySplitsSimple updater:
    - County splits: the number of splits ('county_splits_simple'), or
      Mattingly's county split score function ('county_splits_mattingly')

    If the chain has a polsby_popper updater
    - Other methods for calculating compactness:
        - using mean Polsby Popper score ('pp_mean')
        - using max Polsby Popper score ('pp_max')
        - using Mattingly's Polsby Popper method ('pp_mattingly')

    The resulting function will allow the Markov chain to sample from a weighted
    distribution rather than sampling from the uniform distribution. It may also
    allow the chain to explore the sample space more freely, because there is no
    hard cutoff, but rather a 'slope' of acceptability.
    """
    def __init__(self, weights={'cut_edges': 0.5, 'pop_mattingly': 100}, beta=1, **kwargs):
        """
        Initializes a Metropolis-Hastings acceptance function for GerryChain.

        Parameters:
          weights (dict): Length 5 ndarray specifying weights for score
                          functions in the fitness function
          beta (float): strictness of the acceptance function

        Optional Parameters:

          election (str): GerryChain election updater alias (required if using a
                          GerryChain partisan metric or partisan_dislocation)
          custom (func): function which assigns a nonzero float to a GerryChain
                         partition

          populations (str): filename of pickled population data (required if
                             using partisan_dislocation)

          environments (str): filename of pickled partisan environment data
                              (required if using partisan_dislocation)

        List of score functions:

        Default:
        'cut_edges', 'pop_mattingly'

        Partisan metrics:
        'partisan_bias', 'partisan_gini', 'efficiency_gap', 'mean_median', 'wins'
        'partisan_dislocation'

        Additional nonpartisan metrics:
        'county_splits_simple', 'county_splits_mattingly'
        'pp_mean', 'pp_max', 'pp_mattingly'

        Custom metric:
        'custom'
        """
        # Score functions
        default = ['cut_edges', 'pop_mattingly', 'custom']
        partisan = ['partisan_bias', 'partisan_gini', 'efficiency_gap',
                            'mean_median', 'wins', 'partisan_dislocation']
        nonpartisan = ['county_splits_simple, county_splits_mattingly',
                               'pp_mean', 'pp_max', 'pp_mattingly']

        allowable_score_functions = default + partisan + nonpartisan

        # Record score functions
        self.score_functions = [key for key, val in weights.items() if val != 0]

        # Check
        for score_function in self.score_functions:
            assert score_function in allowable_score_functions

        for score_function in partisan:
            if score_function in self.score_functions:
                assert 'election' in kwargs
                self.election = election

        if 'partisan_dislocation' in self.score_functions:
            assert 'populations' in kwargs
            self.populations = pickle.load(open(populations, 'rb'))
            assert 'environments' in kwargs
            self.environments = pickle.load(open(environments, 'rb'))

        if 'custom' in self.score_functions:
            assert 'custom' in kwargs
            self.custom = custom

        # Construct an array of weights in the same order as the score functions
        self.weights = np.array([weights[sf] for sf in self.score_functions], dtype=np.float64)
        self.beta = beta

    def fitness(self, partition):
        """
        Determines the absolute fitness of a partition in terms of its compactness
        and population max-min, using the given weights and scaling.

        Parameters:
          grid1: a Networkx Graph object, built as a GerryChain grid

        Returns:
          w (float): >= 0, a float indicating the "fitness" of the partition. 0 is best
        """
        # Get raw scores
        scores = np.zeros(len(self.weights), dtype=np.float64)

        for i, function in enumerate(self.score_functions):
            if function == 'cut_edges':
                scores[i] = len(partition['cut_edges'])

            elif function == 'pop_mattingly':
                scores[i] = self.population_equality_score(partition, mode='mattingly')

            elif function == 'custom':
                scores[i] = self.custom(partition)

            elif function == 'partisan_dislocation':
                scores[i] = partisan_dislocation_score(partition[self.election], partition[assignment_array], self.environments, self.populations, mode = 'avgabsolute')

            elif function == 'county_splits_simple':
                if 'split_counties_simple' in partition.updaters.keys():
                    scores[i] = partition['split_counties_simple']
                else:
                    scores[i] = county_splits_score(partition['split_counties'], three_penalty=100, start_at_one=False, mode='simple')

            elif function == 'county_splits_mattingly':
                scores[i] = county_splits_score(partition['split_counties'], three_penalty=100, start_at_one=False, mode='mattingly')

            elif function == 'pp_mean':
                scores[i] = polsby_popper_score(self, partition, mode="mean")

            elif function == 'pp_max':
                scores[i] = polsby_popper_score(self, partition, mode="max")

            elif function == 'pp_mattingly':
                scores[i] = polsby_popper_score(self, partition, mode="mattingly")

            elif function == 'partisan_bias':
                scores[i] = partisan_bias(partition[self.election])

            elif function == 'partisan_gini':
                scores[i] = partisan_gini(partition[self.election])

            elif function == 'efficiency_gap':
                scores[i] = efficiency_gap(partition[self.election])

            elif function == 'mean_median':
                scores[i] = mean_median(partition[self.election])

            elif function == 'wins':
                scores[i] = partition[self.election].wins('Rep')

        # Calculate our weighted coefficient (lower value is better)
        return self.beta * np.sum(self.weights * scores)

    def __call__(self, partition):
        """
        A probabilistic acceptance function which calculates how acceptable a
        partition is and returns a boolean value indicating whether or not to accept
        the partition.

        Parameters:
          grid1: a Networkx Graph object, built as a GerryChain grid
          absolute (bool): default True; whether to calculate absolute fitness or
                        fitness relative to the previous partition.

        Returns:
          accept (bool): True or False (whether to accept the grid)
        """

        # Use probabilistic acceptance
        # If the new partition is "better" then it will be accepted with probability 1
        # Otherwise, there is some probability that it will be rejected

        return np.exp(self.fitness(partition.parent)-self.fitness(partition)) > np.random.random(1)

    # Score functions

    def population_equality_score(self, partition, mode="mattingly"):
        """
        Calculates a population equality score for a district based on population data.

        Parameters:
            partition: a GerryChain Partition object
            mode (str): a keyword defining which calculation mode to use
                "max-min": calculates the population difference between the largest and smallest districts,
                            relative to the ideal population
                "mattingly": uses Mattingly's score function, which sums the squared devations of
                            district populations from the ideal

        Returns:
            score (float): the calculated fitness score (lower is better)
        """
        # Extract the data
        pops = np.array(partition['population'].values())

        # Calculate the ideal population
        if self.ideal_pop is None:
            self.ideal_pop = np.mean(mm)

        # Calculate the max-min score
        if mode == "max-min":
            return (np.max(mm) - np.min(mm))/self.ideal_pop

        # Calculate the square root of squared deviations from ideal
        elif mode == "mattingly":
            return np.sqrt(np.sum(np.square((mm/self.ideal_pop)-1)))

    def polsby_popper_score(self, partition, mode="mattingly"):
        """
        Calculates a compactness score for a district based on compactness data.

        Parameters:
            partition: a GerryChain Partition object
            mode (str): a keyword defining which calculation mode to use
                "max": calculates the max polsby-popper score, subtracted from one
                "mean": calculates the mean polsby-popper score, subtracted from one
                "mattingly": uses Mattingly's score function, which sums the reciprocals of polsby-popper scores

        Returns:
            score (float): the calculated fitness score (lower is better)
        """
        # Extract the data
        pp = np.array(partition['polsby_popper'].values())

        # Calculate the max Polsby-Popper score
        if mode == "max":
            return 1 - np.max(pp)

        # Calculate the mean Polsby-Popper score
        elif mode == "mean":
            return 1 - np.mean(pp)

        # Use Mattingly's score function
        elif mode == "mattingly":
            return np.sum(np.reciprocal(pp))

def percent_acceptance(d):
    """
    Utility which infers from a produced dataframe the percent acceptance rate
    of the acceptance function used.

    Parameters:
        d : ndarray or pd.DataFrame

    Returns:
        percent_acceptance (float)
    """
    counter = 0
    for i in range(1, d.shape[0]):
      if np.allclose(d.iloc[i, :], d.iloc[i-1, :]):
        counter += 1
    return (d.shape[0]-counter)/d.shape[0]


# Utilities for Merging Precincts

class Merge:

    def __init__(self, n=-1):
        """
        Creates a Merge object which stores information about which precincts will be merged. This object
        stores two attributes:
            self.merges (list): A list of sets of nodes which will be merged. The sets are pairwise disjoint.
            self.merged_objects (set): The set of all nodes in the above list.
            self.length (int): If this is defined, it indicates that all elements must be in range(n)
        """
        self.merges = []
        self.merged_objects = set()

        if n == -1:
            self.length = None
        else:
            self.length = n

    def add(self, part):
        """
        Adds a group of precincts to be merged. Makes sure that the merge sets remain pairwise disjoint.

        Parameters:
            part (set): A set of nodes to be merged.
        """
        # Check for conformity
        assert type(part) == set

        if not self.length is None:
            for i in part:
                assert type(i) == int or type(i) == np.int32 or type(i) == np.int64
                assert 0 <= i < self.length

        # Break into cases

        if len(part) <= 1:
            # Do nothing
            pass
        elif self.merges == []:
            # The first step
            self.merges = [part]
            self.merged_objects = self.merged_objects.union(part)
        else:
            # Check for any intersections
            intersections = []
            non_intersections = []
            for i, part2 in enumerate(self.merges):

                if part.intersection(part2) != set():
                    intersections.append(part2)
                else:
                    non_intersections.append(part2)

            # Retain all the unintersected parts, and add a new part with all the intersections
            self.merges = non_intersections + [part.union(*intersections)]
            self.merged_objects = self.merged_objects.union(part)

    def add_many(self, parts):
        """
        Adds groups of precincts to be merged. Makes sure that the merge sets remain pairwise disjoint.

        Parameters:
            parts (list): A list of sets of nodes to be merged.
        """
        for part in parts:
            self.add(part)

    def __add__(self, other):
        """
        Combines merge objects non-destructively into a sum object.

        Parameters:
            self (Merge)
            other (Merge)

        Returns:
            new (Merge): the sum of these merge objects
        """
        assert self.length == other.length
        new = Merge(self.length)
        new.add_many(self.merges)
        new.add_many(other.merges)

        return new

    def __eq__(self, other):
        """
        Determines if two merge objects are equivalent.
        """
        return (self.get_dissolve() == other.get_dissolve()).all()

    def find(self, element):
        """
        For an element contained in the merge, this method finds the set of objects it will be merged with.

        Parameters:
            element (type): An element contained in the merge

        Returns
            s (set): The set of objects the element will be merged with
        """
        for part in self.merges:
            if element in part:
                # The element will only be in one parts
                return part

    def __contains__(self, element):
        """
        Check for inclusion in the Merge.
        """
        if type(element) == set:
            return element in self.merges

        elif type(element) == Merge:
            return self + element == self

        else:
            return element in self.merged_objects

    def get_dissolve(self, n=-1):
        """
        Creates a dissolve array for use in geopandas.dissolve and other merging functions.

        Parameters:
            n (int): Length of the desired array

        Returns:
            dissolve (ndarray): An array specifying the new ids of each object, consistent with the merge.
        """
        if n == -1:
            assert self.length is not None
            n = self.length

        # Set Parameters
        counter = 0
        dissolve = np.full(n, -1, dtype=np.int32)


        # Iterate through nodes
        for i in range(n):
            if i not in self:
                dissolve[i] = counter
                counter += 1
            elif dissolve[i] == -1:
                s = self.find(i)
                for e in s:
                    dissolve[e] = counter

                counter += 1

        return dissolve

    def from_dissolve(dissolve):
        """
        Creates a Merge object from a dissolve array.

        Parameters:
             dissolve (ndarray): An array specifying the new ids of each object

        Returns:
            m (Merge): a merge object specifying which precincts were merged
        """
        m = Merge(len(dissolve))
        for val in set(dissolve):
            m.add(set(list(np.nonzero(dissolve == val)[0])))

        return m

    def __mul__(self, other):
        """
        Composes two merge objects.
        """
        return Merge.from_dissolve(other.get_dissolve()[self.get_dissolve()])

    def __str__(self):
        """
        Creates a string representation.
        """
        return str(self.merges)

    def __repr__(self):
        """
        Creates a string representation.
        """
        return str(self.merges)

    def __len__(self):
        """
        Returns the length.
        """
        return len(self.merges)

def perform_dissolve_gdf(precincts, dissolve,
                        columns_to_keep = ['geometry', ('CountyID', 'first'), ('VistaID', 'sum'), ('POP100', 'sum'), ('DEM', 'sum'), ('REP', 'sum'), ('US_Distric', 'first')],
                        new_column_names = ['geometry', 'CountyID', 'VistaID', 'POP100', 'DEM', 'REP', 'US_Distric']):
    """
    Dissolves precincts and returns the merged GeoDataFrame.

    Parameters:
        precincts (GeoDataFrame): GeoDataFrame of precinct data
        dissolve (ndarray): an array containing the merge information
        columns_to_keep (list): a list of column names that should be kept in the merged gdf
        new_column_names (list): a list of new names for the columns

    Returns:
        merged (GeoDataFrame): a new merged GeoDataFrame
    """

    # Dissolve the one-neighbor precincts into their containers

    merged = precincts.dissolve(by=dissolve, aggfunc=['sum', 'first'])


    for i, column in enumerate(merged.columns):
        if column not in columns_to_keep:
            merged.columns[i] = new_column_names[i]
        else:
            merged.drop(column, axis=1, inplace=True)

    return merged

def perform_dissolve_graph(graph, dissolve, attributes_to_sum = ['area', 'SHAPE_Area', 'DEM', 'REP', 'POP100'], attributes_to_join = ['VistaID', 'PrecinctID', 'SubPrecinc']):
    """
    Perform the dissolve operation on a graph by relabeling nodes and aggregating attributes
    Parameters:
        graph (gerrychain.graph.Graph): Adjacency graph of precinct data
        dissolve (ndarray): an array containing the merge information
        attributes_to_sum (list): an array of attributes that should be aggregated by summation
        attributes_to_join (list): an array of attributes that should be aggregated through string joining

    Returns:
        graph (gerrychain.graph.Graph): Adjacency graph of precinct data
    """

    # Merge the graphs
    n = len(graph)

    # Construct a dictionary from the dissovle array
    dissolve_dict = {old_id:new_id for old_id, new_id in enumerate(dissolve)} # old_id --> new_id

    # Relabel the nodes
    new_graph = nx.relabel.relabel_nodes(graph, dissolve_dict)

    # Relabel the attributes
    for new_id in dissolve_dict.values():

        original_nodes = list(np.nonzero(dissolve == new_id)[0])

        if len(original_nodes) > 1:
            # We have to fix the attributes

            original_nodes = list(np.nonzero(dissolve == new_id)[0])

            # Set attributes

            # summation aggregation: 'area', 'SHAPE_Area', 'DEM', 'REP', 'POP100'

            for attribute in attributes_to_sum:
                new_graph.nodes[new_id][attribute] = sum(graph.nodes[i][attribute] for i in original_nodes)
            # ex. new_graph.nodes[new_id]['SHAPE_Area'] = sum(graph.nodes[i]['SHAPE_Area'] for i in original_nodes)


            # joining aggregation: VistaID, PrecinctID, SubPrecinc

            for attributes in attributes_to_join:
                new_graph.nodes[new_id][attribute] = ', '.join([str(graph.nodes[i][attribute]) for i in original_nodes])
            # ex. new_graph.nodes[new_id]['PrecinctID'] = ', '.join([str(graph.nodes[i]['PrecinctID']) for i in original_nodes])

            # Misc attributes
            new_graph.nodes[new_id]['boundary_node'] = any(graph.nodes[i]['boundary_node'] for i in original_nodes)
            # CountyID should be preserved in any case
            try:
                new_graph.nodes[new_id]['VersionNbr'] = 1+max(graph.nodes[i]['VersionNbr'] for i in original_nodes)
            except:
                pass
            new_graph.nodes[new_id]['EffectiveD'] = '2020-04'
            # AliasName should be preserved in any case
            new_graph.nodes[new_id]['Comments'] = 'merged, 2020-04'
            new_graph.nodes[new_id]['RcvdDate'] = '2020-04'
            new_graph.nodes[new_id]['SHAPE_Leng'] = max(graph.nodes[i]['SHAPE_Leng'] for i in original_nodes)
            # US_District will be preserved

            # Merge the edges in the graph
            for j, val in new_graph[new_id].items():

                # j new id of neighbor. We need to get the preimage ids of neighbors
                neighbors = list(np.nonzero(dissolve == j)[0])

                shared_perim = sum(sum(graph[i][n]['shared_perim'] for n in neighbors if n in graph[i]) for i in original_nodes)

                new_graph[new_id][j]['shared_perim'] = shared_perim
                new_graph[j][new_id]['shared_perim'] = shared_perim

                id_ = min(new_id, j)

                new_graph[new_id][j]['id'] = id_
                new_graph[j][new_id]['id'] = id_

            if new_graph.nodes[new_id]['boundary_node']:
                new_graph.nodes[new_id]['boundary_perim'] = sum(graph.nodes[i].get('boundary_perim', 0) for i in original_nodes)

        else:
            # This node was not merged, no attribute changes required
            pass

    # Remove all self-loops
    new_graph.remove_edges_from(nx.selfloop_edges(new_graph))

    # Return the new graph
    return new_graph

def get_one_neighbor(graph):
    """
    Get all the one-neighbor precincts in a graph.

    Parameters:
        graph (nx.Graph): adjacency graph for precincts
        precincts (gp.GeoDataFrame): geoDataFrame with precinct information

    Returns:
        num_neighbors: an array mapping numbers of neighbors to numbers of precincts with that many neighbors (for histogram)
        ids: array of one-neighbor precincts
        neighbor_ids: array of their neighbors' ids
        containers: a dictionary mapping nodes adjacent to one-neighbor precincts to those precinct(s)
        merge: a Merge object representing the proposed merge
    """

    # Set parameters
    num_neighbors = np.zeros(35)
    ids, neighbor_ids = [], []
    containers = {}

    # Iterate through nodes
    for node in graph:
        neighbors = len(graph[node])
        num_neighbors[neighbors] += 1

        # Is it a one-neighbor node?
        if neighbors == 1:
            neighbor = list(graph[node].keys())[0]
            print("Node: ", node, ", Neighbor: ", neighbor, ", County: ", counties[graph.nodes[node]["CountyID"]], )

            # Mark it
            ids.append(node)
            neighbor_ids.append(neighbor)

            # Mark its container
            if neighbor not in containers.keys():
                containers[neighbor] = [node]
            else:
                containers[neighbor].append(node)

    ids = np.array(ids)
    neighbor_ids = np.array(neighbor_ids)

    print("Total: "+str(int(num_neighbors[1])))

    # Create merge object
    merge = Merge(len(graph))
    merge.add_many([set([key]).union(set(val)) for key, val in containers.items()])

    return num_neighbors, ids, neighbor_ids, containers, merge

def get_separators(graph):
    """
    Get all the precincts in the graph which, upon removal, separate the graph into
    multiple components. For example, if one precinct completely contains two others,
    but they each border each other, then they will each have two neighbors. However,
    there is no continguous districting plan in which they (and their containing precinct)
    are in the same district.

    Parameters:
        graph (nx.Graph): adjacency graph for precincts
        precincts (gp.GeoDataFrame): geoDataFrame with precinct information

    Returns:
        ids: array of one-neighbor precincts
        neighbor_ids: array of their neighbors' ids
        separations: a dictionary mapping separating nodes to the precincts they separate from the graph
        Merge: a Merge object representing the proposed merge
    """

    # Set parameters
    num_neighbors = np.zeros(35)
    ids, neighbor_ids = [], []
    separations = {}

    for node in graph:

        neighbors = len(graph[node])
        num_neighbors[neighbors] += 1

        copy = graph.copy()
        copy.remove_node(node)

        # See if the graph becomes disconnected by removing a particular node
        cc = list(nx.connected_components(copy))

        if len(cc) != 1:
            neighbor_ids.append(node)
            disconnected_nodes = [subpart for part in cc[1:] for subpart in part]
            separations[node] = disconnected_nodes
            for n in disconnected_nodes:
                ids.append(n)

    ids = np.array(ids)
    neighbor_ids = np.array(neighbor_ids)

    print("Total: "+str(int(len(ids))))

    # Construct the dissolve array

    counter = -1
    dissolve = np.zeros(len(graph), dtype=np.int32)

    # Create merge object
    merge = Merge(len(graph))
    merge.add_many([set([key]).union(set(val)) for key, val in separations.items()])

    return ids, neighbor_ids, separations, merge

def merge_multipolygons(graph, gdf):
    """
    This function attempts to create a merge which dissolves all multipolygons into polygons.

    Parameters:
        graph (nx.Graph): an adjacency graph for the precincts
        gdf (GeoDataFrame): a gdf of the precinct geometries

    Returns:
        merge (Merge): a Merge object containing the proposed merges
    """

    # Find all the multipolygons
    multipolygons = []
    for i, poly in enumerate(gdf['geometry']):
        if type(poly) != Polygon:
            multipolygons.append(i)

    merges = Merge(len(graph))

    # Iterate over the multipolygons
    for i, mp in enumerate(multipolygons):

        print(i, end=', ')

        # We don't want to merge it with a precinct in a different county
        # Only iterate through neighbors in the same county and congressional district
        neighbors_in_county = [n for n in graph[mp].keys()
                               if graph.nodes[n]['CountyID'] == graph.nodes[mp]['CountyID']
                               and graph.nodes[n]['US_Distric'] == graph.nodes[mp]['US_Distric']]

        # Determine whether any of its immediate neighbors are multipolygons
        single_mp = True # store it in this flag
        for n in neighbors_in_county:
            if n in multipolygons:
                single_mp = False
                break

        # Set parameters
        possibilities = []
        comb_number = 1

        # If no neighbors are multipolygons
        if single_mp:

            # hopefully we only have to go through this loop once
            while len(possibilities) == 0:

                # Iterate through all combinations of neighboring precincts
                # At the beginning, just try combinations of length 1
                for comb in itertools.combinations(neighbors_in_county, comb_number):

                    # Merge adjacent polygon(s)
                    s = unary_union(list(gdf.iloc[mp]['geometry']) + [gdf.iloc[n]['geometry'] for n in comb])

                    # If the merge worked, store it as a possibility
                    if type(s) == Polygon:
                        possibilities.append(list(comb))

                # If merging with one neighbor fails to work, we will have to try merging with two neighbors
                comb_number += 1

            # Maximizing the shared perimeter of the precincts being merged is a decent way to pick good merges
            # Find argmax graph[mp][p]['shared_perim'] for p in possibilities
            best = possibilities[np.argmax([sum(graph[mp][n]['shared_perim'] for n in comb) for comb in possibilities])]
            merges.add(set([mp] + best))

        # If some neighbors are multipolygons
        else:
            # Some possibilities might not completely merge into polygons. Store these in a second list
            second_possibilities = []

            # hopefully we only have to go through this loop once
            while len(possibilities) == 0 and len(second_possibilities) == 0:

                # Iterate through all combinations of neighboring precincts
                # At the beginning, just try combinations of length 1
                for comb in itertools.combinations(neighbors_in_county, comb_number):
                    # Merge adjacent polygon(s)
                    s = unary_union(list(gdf.iloc[mp]['geometry']) + [gdf.iloc[n]['geometry'] for n in comb])

                    if type(s) == Polygon:
                        possibilities.append(list(comb))

                        # Perhaps the merge worked for our selected multipolygon but not for its neighbors
                        # This is ok because later, the neighbors will be visited and merged
                    elif len(s) <= sum(len(gdf.iloc[mp1]['geometry']) for mp1 in comb if mp1 in multipolygons):
                        second_possibilities.append(list(comb))

                # If merging with one neighbor fails to work, we will have to try merging with two neighbors
                comb_number += 1

            # If there are no combinations that work, use the second-chance possibilities
            if possibilities == []:
                possibilities = second_possibilities

            # Find argmax graph[mp][p]['shared_perim'] for p in possibilities
            best = possibilities[np.argmax([sum(graph[mp][n]['shared_perim'] for n in comb) for comb in possibilities])]
            merges.add(set([mp] + best))

    # Run one last check, to see if we can clear anything up
    unfinished = []
    for subset in merges.merges:
        if type(unary_union([gdf.iloc[i]['geometry'] for i in subset])) != Polygon:
            unfinished.append(subset)



    for subset in unfinished:

        possible_neighbors = []
        for p in subset:
            possible_neighbors += list(graph[p].keys())

        neighbors_in_county = set([n for n in possible_neighbors if graph.nodes[n]['CountyID'] == graph.nodes[list(subset)[0]]['CountyID']])
        comb_number = 0
        possibilities = []

        while len(possibilities) == 0:

            for comb in itertools.combinations(neighbors_in_county, comb_number):
                # Merge adjacent polygon(s)
                s = unary_union([gdf.iloc[n]['geometry'] for n in set(list(subset) + list(comb))])

                # If the merge worked, store it as a possibility
                if type(s) == Polygon:
                    possibilities.append(list(comb))

            comb_number += 1
        # Find argmax graph[mp][p]['shared_perim'] for p in possibilities
        best = possibilities[np.argmax([sum(sum(graph[mp][n]['shared_perim'] for n in comb if n in graph[mp]) for mp in subset) for comb in possibilities])]
        merges.add(subset.union(set(best)))

    return merges

def merge_zero_population(graph, gdf):
    """
    Produces a merge object which merges zero-population nodes with their neighbors.

    Parameters:
        graph (nx.Graph): an adjacency graph for the precincts
        gdf (GeoDataFrame): a gdf of the precinct geometries

    Returns:
        merge (Merge): a Merge object containing the proposed merges
    """
    # Create the Merge object
    merges = Merge(len(graph))

    # Iterate through zero-population nodes
    for node in graph.nodes:
        if graph.nodes[node]['POP100'] == 0:
            neighbors = [n for n in graph[node].keys()
                               if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                               and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']
                               and graph.nodes[n]['POP100'] == 0]

            if len(neighbors) == 0:
                # Try using zero-population neighbors
                neighbors = [n for n in graph[node].keys()
                                   if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                                   and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']]

            # Maximizing the shared perimeter of the precincts being merged is a decent way to pick good merges
            best = neighbors[np.argmax([graph[node][n]['shared_perim'] for n in neighbors])]
            merges.add(set([node, best]))
            if node == best:
                print('!')


    return merges

# Plotting functions

def plot_district_map(assignment, size=(3,2), dpi=300, precincts='UtahData/18_Precincts_combined_contiguous-Copy1.shp', save=False, savetitle=None):
    """
    Given a districting assignment, plots the state of Utah divided into the given districts using 2018 precinct data.

    Parameters:
        assignment (gerrychain.Assignment): a districting assignment. A dictionary will work as long as it has the right keys
        size ((2,) tuple): figure size
        dpi (int): figure resolution
        precincts: Filename of a geodataframe containing the shapefiles. Alternatively, pass it in directly

    """
    # Load in the shapefiles
    if type(precincts) == str:
        precincts = gp.read_file(precincts)
    else:
        precincts = precincts.copy()

    # Load the district data into the geodataframe containing the shapefiles
    if type(assignment) != np.ndarray and type(assignment) != pd.Series:
        precincts['plot_val'] = [assignment[i] for i in range(len(assignment))]
    else:
        precincts['plot_val'] = assignment

    # Plot the data
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    precincts.plot(column='plot_val', ax=ax)
    plt.axis("off")

    # Save if desired
    if save: plt.savefig(savetitle+'.png', dpi=dpi)
    plt.show()

def plot_graph(precincts, graph, window=None, node_size=0.1, line_size=0.05, dpi=400, size=7, save=False, savetitle=None):
    """
    Plots the precinct adjacency graph, over the state of Utah.

    Selected Parameters:
        precincts (str): filename of precinct shapefiles. Alternatively, pass them in directly
        graph (str): filename of graph. Alternatively, pass in directly
        cmap: The matplotlib cmap used in plotting.
        size (int): figure size (we use a fixed aspect ratio for Utah)
        dpi (int): figure resolution
    """

    # Load in the shapefiles
    if type(precincts) == str:
        precincts = gp.read_file(precincts)
    else:
        precincts = precincts.copy()

    # Load in the graph
    if type(graph) == str:
        graph = Graph.from_json(graph)

    # Obtain a graph coloring
    d = nx.greedy_color(graph, strategy='largest_first')
    coloring = np.array([d[i] for i in range(len(graph))])

    precincts['color'] = coloring
    precincts['center'] = precincts.centroid   # the location of the nodes
    nodes = gp.GeoDataFrame(geometry=precincts.center)    # make a GeoDataFrame of nodes
    E = [LineString([precincts.loc[a,'center'],precincts.loc[b,'center']])
         for a,b in list(graph.edges)]         # Construct a line for each edge
    edges = gp.GeoDataFrame(list(graph.edges), geometry=E) # make a geoDataFrame of edges

    fig = plt.figure(dpi=dpi)               # Set up the figure
    fig.set_size_inches(size,size*2)        # Make it have the same proportions as Utah
    ax = plt.subplot(1,1,1)

    precincts.plot('color', cmap='tab20', ax=ax, alpha=0.5)   # Plot precincts
    nodes.plot(ax=ax,color='k', markersize=node_size)               # Plot nodes
    edges.plot(ax=ax, lw=line_size, color='k')                      # plot edges
    if window is None:
        plt.axis('off')                                     # no coordinates
    else:
        plt.axis(window)

    if save: plt.savefig(savetitle, bbox_inches='tight', dpi=dpi)       # Save it

def calc_percentile(val, data):
    return opt.bisect(lambda x: np.percentile(data, x) - val, 0, 100)

def make_box_plot(data, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, save=False, current_plan_name='2012 plan'):
    """
    Makes a box plot of the given data, with the specified parameters.
    Only pass in the columns of the dataframe which contain the vote shares.
    """
    n = len(data)
    m = len(data.iloc[0])

    # Parameter to help make the fig look good
    k = max(1, m//14)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axhline(0.5, color="#cccccc")
    data.boxplot(ax=ax, positions=range(1, m+1), sym='', zorder=1)
    ax.scatter(data.iloc[0].index, data.iloc[0], color="r", marker="o", s=25/k, alpha=0.5, zorder=5, label=current_plan_name)
    ax.legend(loc='lower right')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([i for i in range(1, m+1)])

    # Hide enough xlabels to that they don't overlap or look crowded
    if k > 1:
        for i,label in enumerate(ax.xaxis.get_ticklabels()):
            if i%k:
                label.set_visible(False)

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

def make_violin_plot(data, title='', ylabel='', xlabel='', figsize=(6,8), dpi=400, savetitle=None, save=False, current_plan_name='2012 plan'):
    """
    Make a violin plot of the given data, with the specified parameters.
    Only pass in the columns of the dataframe which contain the vote shares.
    """
    n = len(data)
    m = len(data.iloc[0])

    # Parameter to help make the fig look good
    k = max(1, m//14)

    d = data.T

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axhline(0.5, color="#cccccc")
    try:
        ax.violinplot(d)
    except FloatingPointError:
        ax.violinplot(d, bw_method=0.4)

    ax.hlines(y=d.iloc[:, 0], xmin = np.arange(m)+1-0.2, xmax=np.arange(m)+1+0.2, color='r', lw=2, label=current_plan_name)
    ax.legend(loc='lower right')
    for i in range(m):
        plt.text(i+1, d.iloc[i, 0]-0.04, str(np.round(calc_percentile(d.iloc[i, 0], d.iloc[i]),1))+'%', horizontalalignment='center')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([i for i in range(1, m+1)])

    # Hide enough xlabels to that they don't overlap or look crowded
    if k > 1:
        for i,label in enumerate(ax.xaxis.get_ticklabels()):
            if i%k:
                label.set_visible(False)

    if save: plt.savefig(savetitle, dpi=dpi, bbox_inches='tight')
    plt.clf()

def make_plots(idnum, kind, subdirectory='Data/', figsize=(8,6), dpi=400, file_type='.pdf'):
    """
    Given the id number of a chain run, creates relevant plots (Utah data format only).

    Parameters:
        idnum (int): the Unix timestamp of the second when the chain was created
                    (part of the filename)
        kind (str): the type of chain run
        subdirectory (str): the subdirectory to save the resulting plots to
        figsize (tup): the desired figure size for the plots. Default: (8, 6)
        dpi (int) the desired dots per inch for the plots. Default: 400 dpi
        file_type (str): the desired filetype of the saved plots

    Creates the following plots:
        - Box Plot + Violin Plot of Sen, Gov, and Combined Vote Shares
        - histogram of the following:
            - Average Absolute Partisan Dislocation (Sen, Gov, Combined)
            - Mean Median (Sen, Gov, Combined)
            - Partisan Bias (Sen, Gov, Combined)
            - Partisan Gini (Sen, Gov, Combined)
            - Efficiency Gap (Sen, Gov, Combined)
            - Seats Won  (Sen, Gov, Combined)
            - County splits
            - Mattingly county split score
            - Mean polsby popper
            - Max polsby popper
            - Population standard deviation
            - Population max-min

    Total: 33 plots.
    """
    assert kind in ['flip-uniform', 'flip-mh', 'recom-uniform', 'recom-mh']

    # Extract the data
    data = pd.read_hdf(str(idnum)+'.h5', 'data')
    params = {'figsize':figsize, 'dpi':dpi, 'save':True}
    n = len(data)
    m = int((len(data.columns)-21)/5)

    pp = data.iloc[:, 21:21+m]
    data['Mean Polsby Popper'] = pp.mean(axis=1)
    data['Max Polsby Popper'] = pp.max(axis=1)

    pop = data.iloc[:, 21+m:21+2*m]
    data['Population Standard Deviation, % of Ideal'] = pop.std(axis=1, ddof=0)/pop.mean(axis=1)
    data['Population Max-Min, % of Ideal'] = (pop.max(axis=1) - pop.min(axis=1))/pop.mean(axis=1)

    # Set parameters
    common_file_ending = '-'+str(len(data))+'-'+kind+'-'+str(idnum)+file_type

    boxplots = {'Box Plot Sen 2010':   {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Senate 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'BoxPlotSen2010'+common_file_ending},

            'Box Plot Gov 2010':       {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Gubernatorial 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'BoxPlotGov2010'+common_file_ending},

            'Box Plot Comb 2010':       {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Combined 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'BoxPlotComb2010'+common_file_ending},

            'Violin Plot Sen 2010':    {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Senate 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'ViolinPlotSen2010'+common_file_ending},

            'Violin Plot Gov 2010':    {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Gubernatorial 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'ViolinPlotGov2010'+common_file_ending},

            'Violin Plot Comb 2010':    {'title': 'Distribution of Vote Shares in {}-Plan Ensemble'.format(n),
                                        'ylabel': 'Republican Vote Share (Combined 2010)',
                                        'xlabel': 'Sorted US Congressional Districts',
                                        'savetitle': subdirectory+'ViolinPlotComb2010'+common_file_ending}
               }


    metricplots = {'Avg Abs Partisan Dislocation - SEN': {'title': 'Avg Abs Partisan Dislocation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Avg Abs Partisan Dislocation (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'AvgAbsPDSen2010'+common_file_ending},
                    'Avg Abs Partisan Dislocation - G': {'title': 'Avg Abs Partisan Dislocation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Avg Abs Partisan Dislocation (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'AvgAbsPDGov2010'+common_file_ending},
                    'Avg Abs Partisan Dislocation - COMB': {'title': 'Avg Abs Partisan Dislocation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Avg Abs Partisan Dislocation (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'AvgAbsPDComb2010'+common_file_ending},
                    'Mean Median - SEN': {'title': 'Mean-Median Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean-Median Score (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanMedianSen2010'+common_file_ending},
                    'Mean Median - G': {'title': 'Mean-Median Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean-Median Score (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanMedianGov2010'+common_file_ending},
                    'Mean Median - COMB': {'title': 'Mean-Median Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean-Median Score (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanMedianComb2010'+common_file_ending},
                    'Efficiency Gap - SEN': {'title': 'Efficiency Gap in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Efficiency Gap (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'EfficiencyGapSen2010'+common_file_ending},
                    'Efficiency Gap - G': {'title': 'Efficiency Gap in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Efficiency Gap (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'EfficiencyGapGov2010'+common_file_ending},
                    'Efficiency Gap - COMB': {'title': 'Efficiency Gap in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Efficiency Gap (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'EfficiencyGapComb2010'+common_file_ending},
                    'Partisan Bias - SEN': {'title': 'Partisan Bias Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Bias Score (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanBiasSen2010'+common_file_ending},
                    'Partisan Bias - G': {'title': 'Partisan Bias Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Bias Score (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanBiasGov2010'+common_file_ending},
                    'Partisan Bias - COMB': {'title': 'Partisan Bias Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Bias Score (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanBiasComb2010'+common_file_ending},
                    'Partisan Gini - SEN': {'title': 'Partisan Gini Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Gini Score (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanGiniSen2010'+common_file_ending},
                    'Partisan Gini - G': {'title': 'Partisan Gini Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Gini Score (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanGiniGov2010'+common_file_ending},
                    'Partisan Gini - COMB': {'title': 'Partisan Gini Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Partisan Gini Score (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'PartisanGiniComb2010'+common_file_ending},
                    'Seats Won - SEN': {'title': 'Seats Won in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Seats Won (Senate 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SeatsWonSen2010'+common_file_ending},
                    'Seats Won - G': {'title': 'Seats Won in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Seats Won (Gubernatorial 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SeatsWonGov2010'+common_file_ending},
                    'Seats Won - COMB': {'title': 'Seats Won in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Seats Won (Combined 2010)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SeatsWonComb2010'+common_file_ending},
                    'County Splits' : {'title': 'Split Counties in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Number of Split Counties',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'SplitCounties'+common_file_ending},
                    'Mattingly Splits Score' :  {'title': 'Mattingly Split Counties Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mattingly Split Counties Score',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MattinglySplitCounties'+common_file_ending},
                    'Cut Edges' :  {'title': 'Cut Edges in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Number of Cut Edges',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'CutEdges'+common_file_ending},
                    'Mean Polsby Popper':  {'title': 'Mean Polsby-Popper Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Mean Polsby-Popper Score',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MeanPolsbyPopper'+common_file_ending},
                    'Max Polsby Popper': {'title': 'Max Polsby-Popper Score in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Max Polsby-Popper Score',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MaxPolsbyPopper'+common_file_ending},
                    'Population Standard Deviation, % of Ideal':  {'title': 'Population Deviation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Standard Deviation of District Populations, % of Ideal',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'StdevPop'+common_file_ending},
                    'Population Max-Min, % of Ideal': {'title': 'Population Deviation in a {}-Plan Ensemble'.format(n),
                                                  'xlabel': 'Largest Deviation in District Populations (Max-Min, % of Ideal)',
                                                  'ylabel': 'Number of Plans',
                                                  'savetitle': subdirectory+'MaxMinPop'+common_file_ending},
            }

    print('Finished Importing Data')

    # Box plot: Senate 2010
    key = 'Box Plot Sen 2010'
    vote_share_sen10 = pd.DataFrame(list(data.iloc[:, 21+2*m:21+3*m].values), columns=np.arange(1, m+1))
    make_box_plot(vote_share_sen10, **boxplots[key], **params)

    print('Finished Box Plot 1')

    # Violin plot: Senate 2010
    key = 'Violin Plot Sen 2010'
    make_violin_plot(vote_share_sen10, **boxplots[key], **params)

    print('Finished Violin Plot 1')

    # Box plot: Governor 2010
    key = 'Box Plot Gov 2010'
    vote_share_gov10 = pd.DataFrame(list(data.iloc[:, 21+3*m:21+4*m].values), columns=np.arange(1, m+1))
    make_box_plot(vote_share_gov10, **boxplots[key], **params)

    print('Finished Box Plot 2')

    # Violin plot: Gov 2010
    key = 'Violin Plot Gov 2010'
    make_violin_plot(vote_share_gov10, **boxplots[key], **params)

    print('Finished Violin Plot 2')

    # Box plot: Governor 2010
    key = 'Box Plot Comb 2010'
    vote_share_comb10 = pd.DataFrame(list(data.iloc[:, 21+4*m:21+5*m].values), columns=np.arange(1, m+1))
    make_box_plot(vote_share_comb10, **boxplots[key], **params)

    print('Finished Box Plot 3')

    # Violin plot: Gov 2010
    key = 'Violin Plot Comb 2010'
    make_violin_plot(vote_share_comb10, **boxplots[key], **params)

    print('Finished Violin Plot 3')

    plt.close('all')

    # Construct plots for the various metrics
    for key in metricplots.keys():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        metric = pd.Series(data[key])
        metric.hist(bins=50)
        ax.axvline(x=metric[0], color='r', lw=2, label='2012 plan, '+str(np.round(calc_percentile(metric[0], metric),1))+'%')
        ax.set_title(metricplots[key]['title'])
        ax.set_xlabel(metricplots[key]['xlabel'])
        ax.set_ylabel(metricplots[key]['ylabel'])
        ax.legend(loc='upper right')
        plt.savefig(metricplots[key]['savetitle'], dpi=dpi, bbox_inches='tight')
        plt.clf()

        print('Finished Plot: {}'.format(key))

        plt.close('all')

def precincts_moving_frequency(idnum, subdirectory='Data/', save=False):
    """
    Given the id number of a chain run, generates an array mapping precinct ids to
    the number of recorded times that that precinct changed assignments.

    Parameters:
        idnum (int): the unix timestamp for when the chain was started
        subdirectory (str): the subdirectory to save the result in
        save (bool): whether or not to save the result to a .npy file

    Returns:
        move_frequencies (ndarray): array mapping precinct ids to the number of
        recorded times that the precinct changed assignments
    """
    # Extract the data
    assignments = pd.read_hdf(str(idnum)+'.h5', 'stored_assignments')
    n = len(assignments)

    # Compute the changes
    changes = np.array(assignments.iloc[1:, :]) - np.array(assignments.iloc[:-1, :])
    move_frequencies = np.count_nonzero(changes, axis=1)

    # Save if desired
    if save: np.save(subdirectory+str(i)+'moves.npy', move_frequencies)

    # Result the result if desired
    return np.count_nonzero(changes, axis=1)
