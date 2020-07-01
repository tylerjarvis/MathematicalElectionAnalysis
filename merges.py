# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election, grid)
from gerrychain.metrics import mean_median, partisan_bias, polsby_popper, efficiency_gap, partisan_gini
import pandas as pd
import geopandas as gp
import numpy as np
from scipy import optimize as opt
import networkx as nx
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import itertools

from tqdm.auto import tqdm
import pickle

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
    assert len(new_column_names) == len(columns_to_keep)

    # Dissolve the one-neighbor precincts into their containers
    merged = precincts.dissolve(by=dissolve, aggfunc=['sum', 'first'])

    # Drop superfluous columns
    for i, column in enumerate(merged.columns):
        if column not in columns_to_keep:
            merged.drop(column, axis=1, inplace=True)

    # Rename the remaining columns
    merged.columns = new_column_names

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

            for attribute in attributes_to_join:
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

def merge_multipolygons(graph, gdf, preserve_ut_house=False):
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
        if preserve_ut_house:
            neighbors_in_county = [n for n in graph[mp].keys()
                               if graph.nodes[n]['CountyID'] == graph.nodes[mp]['CountyID']
                               and graph.nodes[n]['US_Distric'] == graph.nodes[mp]['US_Distric']
                               and graph.nodes[n]['UT_SEN'] == graph.nodes[mp]['UT_SEN']
                               and graph.nodes[n]['UT_HOUSE'] == graph.nodes[mp]['UT_HOUSE']]
        else:
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
                    and graph.nodes[n]['UT_SEN'] == graph.nodes[node]['UT_SEN']
                    and graph.nodes[n]['UT_HOUSE'] == graph.nodes[node]['UT_HOUSE']
                    and graph.nodes[n]['POP100'] == 0]


            if len(neighbors) == 0:
                # Try using zero-population neighbors
                neighbors = [n for n in graph[node].keys()
                if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']
                and graph.nodes[n]['UT_SEN'] == graph.nodes[node]['UT_SEN']
                and graph.nodes[n]['UT_HOUSE'] == graph.nodes[node]['UT_HOUSE']]

            if len(neighbors) == 0:
                neighbors = [n for n in graph[node].keys()
                if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']
                and graph.nodes[n]['UT_SEN'] == graph.nodes[node]['UT_SEN']
                and graph.nodes[n]['POP100'] == 0]

            if len(neighbors) == 0:
                neighbors = [n for n in graph[node].keys()
                if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']
                and graph.nodes[n]['UT_SEN'] == graph.nodes[node]['UT_SEN']]

            if len(neighbors) == 0:
                neighbors = [n for n in graph[node].keys()
                if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']
                and graph.nodes[n]['POP100'] == 0]

            if len(neighbors) == 0:
                neighbors = [n for n in graph[node].keys()
                if graph.nodes[n]['CountyID'] == graph.nodes[node]['CountyID']
                and graph.nodes[n]['US_Distric'] == graph.nodes[node]['US_Distric']]

            # Maximizing the shared perimeter of the precincts being merged is a decent way to pick good merges
            best = neighbors[np.argmax([graph[node][n]['shared_perim'] for n in neighbors])]
            merges.add(set([node, best]))
            if node == best:
                print('!')




    return merges
