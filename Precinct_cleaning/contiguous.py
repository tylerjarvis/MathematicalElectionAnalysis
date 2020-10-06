from gerrychain import (GeographicPartition, Partition, Graph)
import networkx as nx

def find_buffer(partition, part, district,dist_name):
    """
    Param:  partition: graph of district with assignments
            part: the district number
            district: district that needs to be made contiguous (deep copy of a slice)
            dist_name: Name of the type of district
    returns: edges: list of edges in buffered district
            buffered district
    """
    while not nx.is_connected(partition.subgraphs[part]): # buffer by 1 until connected
        buffer = district.buffer(1)
        district['geometry'] = buffer
        graph = Graph.from_geodataframe(district)
        partition = GeographicPartition(graph, dist_name)
    edges = list(partition.subgraphs[part].edges)
    return edges, district

def get_districts(graph,precincts,dist_name):
    """
    Param: graph: GerryChain adjacency graph
           precincts: geodataframe of the precincts
           dist_name: Name of the district plan to check
    returns: buffered precincts to make dist_name contiguous
            graph with added edges for dist_name
    """
    discontiguous = []
    initial_partition = GeographicPartition(graph, dist_name) # district assignments
    for part in initial_partition.parts:
        if not nx.is_connected(initial_partition.subgraphs[part]): #if part (district) is not contiguous
            discontiguous.append(part) #add to list of disrticts to buffer
    while len(discontiguous) != 0:
        district = precincts.loc[precincts[dist_name] == discontiguous[0]] # get slice of district to buffer
        slice_district =  district.copy(deep = True) #this unlinks it from precincts. Pandas doesn't like reassining geometries in chunks
        slice_graph = Graph.from_geodataframe(district)
        slice_partition = GeographicPartition(slice_graph, dist_name)
        initial_edges = list(slice_partition.subgraphs[discontiguous[0]].edges())
        new_edges, buffered_slice = find_buffer(slice_partition, discontiguous[0], slice_district, dist_name)
        precincts.loc[precincts[dist_name] == discontiguous[0]] = buffered_slice
        for edge in new_edges:
            if edge not in initial_edges:
                graph.add_edge(*edge)
        discontiguous.pop(0)
    return graph, precincts

def make_contiguous(graph,precincts,districts):
    '''
    Param: graph: GerryChain adjacency graph
           precincts: geodataframe of the precincts
           districts: a list of the district names ex: ["US_Distric", "UT_SEN","UT_HOUSE"]
    returns: Geodataframe of new buffered precincts
            Graph with added edges but not updated perim and area
    '''
    for district in districts:
        graph, precincts = get_districts(graph,precincts,district)

    return graph, precincts
