# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph)
import numpy as np

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
