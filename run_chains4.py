# Import necessary packages
from chain import MarkovChain as MC2
from gerrychain import (GeographicPartition, Partition, Graph, proposals, updaters, constraints, accept, Election, grid)
from gerrychain.metrics import mean_median, partisan_bias, polsby_popper, efficiency_gap, partisan_gini
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import json
import time
from functools import partial
import networkx as nx

# Import necessary tools from gerrymandering_tools.py
from metropolis_hastings import *
from partisan_dislocation import *
from counties import *

# Get correct input -----------------------------------------------------------
class Chain:
    """
    A class to assist in the running of Markov chains for gerrymandering
    research. Saves data about the chain in a log file chain_log.txt.
    """

    # Initialization function
    def __init__(self, kind, iters, **kwargs):
        """
        Initializes a run of a chain.

        Parameters:
            kind (str): a kind parameter.
                1) 'flip-uniform' means using the flip proposal with uniform
                acceptance and parameter bounds
                2) 'flip-mh' means using the flip proposal with weighted
                Metropolis-Hastings acceptance function
                3) 'recom-uniform' means using the recom proposal with uniform
                acceptance and parameter bounds
                4) 'recom-mh' means using the recom proposal with weighted
                Metropolis-Hastings acceptance function
        """

        self.id = int(np.round(time.time(), 0))

        # Types of chain runs which are possible
        allowable_kinds = ['flip-uniform', 'flip-mh', 'recom-uniform', 'recom-mh']

        defaults = {'storage_ratio': 100,
                    'checkpoint_ratio': 1000000,
                    'graph': 'graph_combined_vs_2018.json',
                    'state': 'UT',
                    'districts': 'US_Distric',
                    'compactness_ratio': 1.25,
                    'population_wiggle': 0.01,
                    'weights': {'cut_edges': 0.5, 'pop_mattingly': 100},
                    'beta': 1,
                    'population_filename': 'populations_mp_sp.pkl',
                    'pe_gov_filename': 'partisan_environments_mp_sp_G.pkl',
                    'pe_sen_filename': 'partisan_environments_mp_sp_SEN.pkl',
                    'pe_comb_filename': 'partisan_environments_combined.pkl',
                    'pop_col': 'POP100',
                    'starting_assignment': 'current_plan',
                    'partisan_estimators': ['SEN10', 'G10', 'COMB10'],
                    'three_penalty': 1000,
                    'recom_epsilon': 0.02,
                    'recom_node_repeats': 2,
                    'election': 'SEN10',
                    'custom': lambda p: True,
                    'storage_type': 'parquet'
                    }

        # Check kind parameter
        assert kind in allowable_kinds
        self.kind = kind

        # Set parameters
        self.params = defaults.copy()
        self.params.update(kwargs)
        self.params.update({'kind': self.kind, 'id': self.id, 'length': iters})

        # Import the graph
        graph_name = self.params['graph']
        if '.json' in graph_name:
            graph = Graph.from_json(graph_name)
        elif '.pkl' in graph_name:
            graph = pickle.load(open(graph_name, 'rb'))

        # Define our updaters
        my_updaters = {"population": updaters.Tally(self.params['pop_col'], alias="population"),
                       "polsby_popper": polsby_popper,
                       "split_counties": SplitCounties(),
                       'SEN10': Election('SEN10', {"Dem": "SEN_DEM", "Rep": "SEN_REP"}),
                       'G10': Election('G10', {"Dem": "G_DEM", "Rep": "G_REP"}),
                       'COMB10': Election('COMB10', {"Dem":"COMB_DEM", "Rep":"COMB_REP"}),
                       'assignment_array': AssignmentArray()
                      }

        if self.params['starting_assignment'] != 'current_plan':
            self.params['districts'] = 'NEW'

            if type(self.params['starting_assignment']) != np.array:
                assignment = self.params['starting_assignment']
            else:
                a = self.params['starting_assignment']
                assignment = {i: a[i] for i in range(len(a))}
            nx.set_node_attributes(graph, assignment, 'NEW')

        # Define our initial partition
        initial_partition = GeographicPartition(graph, self.params['districts'], my_updaters)

        m = len(initial_partition.assignment.parts)

        # Define tools for partisan environment
        partisan_environments_sen = pickle.load(open(self.params['pe_sen_filename'], 'rb'))
        partisan_environments_gov = pickle.load(open(self.params['pe_gov_filename'], 'rb'))
        partisan_environments_comb = pickle.load(open(self.params['pe_comb_filename'], 'rb'))

        populations = np.array([graph.nodes[n][self.params['pop_col']] for n in graph.nodes], dtype=np.float64)

        # Define recom tools
        ideal_population = sum(initial_partition['population'].values()) / len(initial_partition)

        # We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
        recom_proposal = partial(proposals.recom,
               pop_col=self.params['pop_col'],
               pop_target=ideal_population,
               epsilon=self.params['recom_epsilon'],
               node_repeats=self.params['recom_node_repeats']
              )

        # Break into cases
        if kind == 'flip-uniform':

            # Enforce a compactness bound
            compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]), self.params['compactness_ratio']*len(initial_partition["cut_edges"]))
            population_constraint = constraints.within_percent_of_ideal_population(initial_partition, self.params['population_wiggle'])

            # Construct our Markov Chain
            self.chain = MC2(proposal = proposals.propose_random_flip,
                                constraints=[constraints.single_flip_contiguous, compactness_bound, population_constraint],
                                accept=accept.always_accept,
                                initial_state=initial_partition,
                                total_steps=iters)

        elif kind == 'flip-mh':

            # Construct our acceptance function
            a = MetropolisHastings(self.params['weights'], self.params['beta'], election=self.params['election'], custom=self.params['custom'], environments=self.params['pe_comb_filename'], populations=self.params['population_filename'])

            # Construct our Markov Chain
            self.chain = MC2(proposal = proposals.propose_random_flip,
                                constraints=[constraints.single_flip_contiguous],
                                accept=a,
                                initial_state=initial_partition,
                                total_steps=iters)


        elif kind == 'recom-uniform':

            # Enforce a compactness bound
            compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]), self.params['compactness_ratio']*len(initial_partition["cut_edges"]))
            population_constraint = constraints.within_percent_of_ideal_population(initial_partition, self.params['population_wiggle'])

            # Construct our Markov Chain
            self.chain = MC2(proposal = recom_proposal,
                                constraints=[compactness_bound, population_constraint],
                                accept=accept.always_accept,
                                initial_state=initial_partition,
                                total_steps=iters)

        elif kind == 'recom-mh':

            # Construct our acceptance function
            a = MetropolisHastings(self.params['weights'], self.params['beta'], election=self.params['election'], custom=self.params['custom'], environments=self.params['pe_comb_filename'], populations=self.params['population_filename'])

            # Construct our Markov Chain
            self.chain = MC2(proposal = recom_proposal,
                                constraints=accept.always_accept,
                                accept=a,
                                initial_state=initial_partition,
                                total_steps=iters)


        # Record data
        metric_labels = ["County Splits", "Mattingly Splits Score", "Cut Edges"]

        partisan_metric_labels = ['Avg Abs Partisan Dislocation - SEN', 'Avg Abs Partisan Dislocation - G', 'Avg Abs Partisan Dislocation - COMB',
                                  "Mean Median - SEN", "Mean Median - G", 'Mean Median - COMB',
                                  "Efficiency Gap - SEN", "Efficiency Gap - G", 'Efficiency Gap - COMB',
                                  "Partisan Bias - SEN", "Partisan Bias - G", 'Partisan Bias - COMB',
                                  "Partisan Gini - SEN", "Partisan Gini - G", 'Partisan Gini - COMB',
                                  "Seats Won - SEN", "Seats Won - G", 'Seats Won - COMB']

        district_stats = ['PP', 'POP', 'Sorted SenRep Vote Share ', 'Sorted GRep Vote Share ', 'Sorted CombRep Vote Share ']

        district_labels = [stat+str(num) for stat in district_stats for num in range(1, m+1)]

        # Set parameters
        data = pd.DataFrame(0.0, columns=metric_labels+partisan_metric_labels+district_labels, index=range(iters))
        counter = 0
        stored_assignments = pd.DataFrame(0, columns=range(len(graph)), index=range(int(iters/self.params['storage_ratio'])-1))

        # Iterate through the Markov Chain and store the data in a pd.DataFrame
        for partition in tqdm(self.chain):
            counter += 1

            # Non partisan summary metrics: len 21

            metrics = [county_splits_score(partition["split_counties"], mode="simple", start_at_one=False),
                       county_splits_score(partition["split_counties"], mode="mattingly", three_penalty=self.params['three_penalty'], start_at_one=False),
                       len(partition["cut_edges"])]

            partisan_metrics = [partisan_dislocation_score(partition['SEN10'], partition['assignment_array'], partisan_environments_sen, populations, 'avgabsolute'),
                                partisan_dislocation_score(partition['G10'], partition['assignment_array'], partisan_environments_gov, populations, 'avgabsolute'),
                                partisan_dislocation_score(partition['COMB10'], partition['assignment_array'], partisan_environments_comb, populations, 'avgabsolute'),
                                mean_median(partition['SEN10']), mean_median(partition['G10']), mean_median(partition['COMB10']),
                                efficiency_gap(partition['SEN10']), efficiency_gap(partition['G10']), efficiency_gap(partition['COMB10']),
                                partisan_bias(partition['SEN10']), partisan_bias(partition['G10']), partisan_bias(partition['COMB10']),
                                partisan_gini(partition['SEN10']), partisan_gini(partition['G10']), partisan_gini(partition['COMB10']),
                                partition['SEN10'].wins('Rep'), partition['G10'].wins('Rep'), partition['COMB10'].wins('Rep') ]


            # Measures by district: len 5*m
            pp = list(partition["polsby_popper"].values())
            pop = list(partition["population"].values())
            sen10 = sorted(partition['SEN10'].percents('Rep'))
            gov10 = sorted(partition['G10'].percents('Rep'))
            comb10 = sorted(partition['COMB10'].percents('Rep'))

            #data.loc[counter-1] = np.concatenate((pp, ce, mm, cs, csm, pd_, pd2_, mmd, mmd2, eg, eg2, pb, pb2, pg, pg2, sw, sw2, sd, sd2))
            data.loc[counter-1] = metrics + partisan_metrics + pp + pop + sen10 + gov10 + comb10

            # Store a small amount of the assignments so we can see them
            if (counter % self.params['storage_ratio'] == 0):
                stored_assignments.loc[counter/self.params['storage_ratio']-1] = partition['assignment_array']

        # Format data into appropriate numerical types
        types = {"County Splits":np.uint8, "Mattingly Splits Score":np.float64, "Cut Edges":np.uint16,
                            'Avg Abs Partisan Dislocation - SEN':np.float64, 'Avg Abs Partisan Dislocation - G':np.float64, 'Avg Abs Partisan Dislocation - COMB':np.float64,
                            "Mean Median - SEN":np.float64, "Mean Median - G":np.float64, "Mean Median - COMB":np.float64,
                            "Efficiency Gap - SEN":np.float64, "Efficiency Gap - G":np.float64, "Efficiency Gap - COMB":np.float64,
                            "Partisan Bias - SEN":np.float64, "Partisan Bias - G":np.float64, "Partisan Bias - COMB":np.float64,
                            "Partisan Gini - SEN":np.float64, "Partisan Gini - G":np.float64, "Partisan Gini - COMB":np.float64,
                            "Seats Won - SEN":np.float64, "Seats Won - G":np.float64, "Seats Won - COMB":np.float64}

        types.update({label: np.float64 for label in district_labels})
        types.update({label: np.uint32 for label in ['POP'+str(num) for num in range(1, m+1)]})

        self.data = data.astype(types)
        self.data.columns = self.data.columns.astype(str)

        self.stored_assignments = stored_assignments.astype(np.uint8)
        self.stored_assignments.columns = self.stored_assignments.columns.astype(str)

        # Save data and backup
        if self.params['storage_type'] == 'HDF5':
            with pd.HDFStore(str(self.id)+'.h5') as store:
                store['data'] = self.data.copy()
                store['stored_assignments'] = self.stored_assignments.copy()
        elif self.params['storage_type'] == 'parquet':
            self.data.to_parquet(str(self.id)+'d.parquet.gzip', compression='gzip')
            self.stored_assignments.to_parquet(str(self.id)+'a.parquet.gzip', compression='gzip')

        self.params['custom'] = 'Custom Acceptance Function'

        # Log chain run
        try:
            logdata = pickle.load(open('chain_log.pkl', 'rb'))
            logdata.update({self.id: self.params})
            pickle.dump(logdata, open('chain_log.pkl', 'wb'))

        except FileNotFoundError:
            pickle.dump({self.id: self.params}, open('chain_log.pkl', 'wb'))

        self.data = data
        self.assignments = stored_assignments

class OldChain:

    def __init__(self, id):

        logdata = pickle.load(open('chain_log.pkl', 'rb'))
        self.params = logdata[id]

        self.id = id
        if self.id < 1591297172:
            self.data = pickle.load(open(str(self.id)+'data.pkl', 'rb'))
            self.assignments = pickle.load(open(str(self.id)+'assignments.pkl', 'rb'))
        elif self.id < 1593561600:
            self.data = pd.read_hdf(str(self.id)+'.h5', 'data')
            self.assignments = pd.read_hdf(str(self.id)+'.h5', 'stored_assignments')
        else:
            self.data = pd.read_parquet(str(self.id)+'d.parquet.gzip')
            self.assignments = pd.read_parquet(str(self.id)+'a.parquet.gzip')
        try:
            self.length = self.params['length']
            self.kind = self.params['kind']
        except:
            pass
