# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, grid)
from gerrychain.metrics import mean_median, partisan_bias, polsby_popper, efficiency_gap, partisan_gini
import pandas as pd
import numpy as np

import pickle

class MetropolisHastings:
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
        mm = np.array(list(partition['population'].values()))

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
