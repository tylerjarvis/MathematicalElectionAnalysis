# Required packages
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, grid)
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
from tqdm.auto import tqdm
import pickle

def plot_districts(grid1, districts=None, layout=None, save=False, savetitle=None):
    """
    Plots the districts of a Grid object.

    Parameters:
       grid1: a Networkx Graph object, built as a GerryChain grid
    """
    # Get a sequence of assignments (rather than the stored dictionary)
    if districts is None:
        districts = np.array([grid1.assignment[node] for node in grid1.graph.nodes()])
    n = len(districts)

    # The Kamada-Kawai method for nx drawing works best in this case
    if layout is None:
        nx.drawing.nx_pylab.draw_kamada_kawai(grid1.graph, node_size=8000/n, node_color=districts)
    else:
        nx.draw(grid1.graph, layout, node_size=8000/n, node_color=districts)
    if save: plt.savefig(savetitle, dpi=300)

def polsby_popper_grid(grid1):
    """
    Computes the Polsby-Popper score for each district in the grid.

    Parameters:
        grid1: a Networkx Graph object, built as a GerryChain grid

    Returns:
        scores (ndarray): The scores of each district in the grid

    """
    # Set parameters
    s = set(grid1.assignment.values())
    n = len(s)
    perimeters = np.zeros(n)
    areas = np.array([sum(grid1.graph.nodes[node]['area'] for node in grid1.graph.nodes if grid1.assignment[node] == i) for i in s])

    # Iterate through each node in the graph
    for node1 in grid1.graph.nodes:
        district = grid1.assignment[node1]
        if grid1.graph.nodes[node1]['boundary_node']:
            # Perimeter along boundary
            perimeters[district] += grid1.graph.nodes[node1]['boundary_perim']
        for node2 in grid1.graph[node1].keys():
            if district != grid1.assignment[node2]:
                # Internal perimeter
                perimeters[district] += grid1.graph[node1][node2]['shared_perim']

    # Compute the Polsby-Popper score for each district
    return 4*np.pi*areas/perimeters**2

polsby_popper_av = lambda grid1: np.mean(polsby_popper_grid(grid1))

def district_populations(grid1):
    """
    Computes the populations the districts in a grid.

    Parameters:
      grid1: a Networkx Graph object, built as a GerryChain grid

    Returns:
      pops (ndarray)
    """
    # Set parameters
    s = set(grid1.assignment.values())
    n = len(s)
    pops = np.zeros(n)

    # Iterate through each node and tally its population in the appropriate index of "populations"
    for node in grid1.graph.nodes:
        pops[grid1.assignment[node]] += grid1.graph.nodes[node]["population"]

    return pops

def population_maxmin(grid1):
    """
    Computes the absolute difference between the populations of the most populous and least populous districts in a grid.

    Parameters:
      grid1: a Networkx Graph object, built as a GerryChain grid
      relative: if True, returns difference as a percentage of average district population

    Returns:
      maxmin (int): The largest population difference. Returns a float (percentage) iff relative is True.
    """
    pops = district_populations(grid1)

    # Perform calculations
    av = np.mean(pops)
    maxmin = np.max(pops) - np.min(pops)

    # Return the difference of the maximum and minimum populations
    return maxmin/av

# This is how to build a cusom acceptance function to work with GerryChain's MarkovChain object

class CustomAccept:
    def __init__(self, weights):
        """
        Initialize an acceptance function with a particular set of weights for
        0) Compactness (Polsby-Popper)
        1) Population Max-min

        Parameters:
            weights (ndarray): Length 2 ndarray specifying weights for the two measures above
                          Larger weights indicate greater importance
            k (float): strictness of the acceptance function
        """
        # User-defined parameter
        self.weights = weights

    def fitness(self, grid1):
        """
        Determines the absolute fitness of a partition in terms of its compactness
        and population max-min, using the given weights and scaling.

        Parameters:
            grid1: a Networkx Graph object, built as a GerryChain grid

        Returns:
            w (float): >= 0, a float indicating the "fitness" of the partition. 0 is best
        """
        # Get raw scores
        if "polsby_popper" in grid1.updaters:
            pp = np.mean(list(grid1["polsby_popper"].values()))
        elif "my_polsby_popper" in grid1.updaters:
            pp = np.mean(grid1["my_polsby_popper"])
        else:
            pp = polsby_popper_av(grid1)

        if "pop_maxmin" in grid1.updaters:
            mm = grid1["pop_maxmin"]
        elif "pops" in grid1.updaters:
            p = grid1["pops"]
            mm = (np.max(p)-np.min(p))/np.mean(p)
        else:
            mm = population_maxmin(grid1)

        # Calculate our weighted coefficient (lower value is better)
        return np.sum(self.weights * np.array([1-pp, mm]))

      # Subtract pp from 1 so that good scores are represented by 0

    def __call__(self, grid1):
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
        return np.exp(self.fitness(grid1.parent)-self.fitness(grid1)) > np.random.random(1)

        # If the new partition is "better" then it will be accepted with probability 1
        # Otherwise, there is some probability that it will be rejected
