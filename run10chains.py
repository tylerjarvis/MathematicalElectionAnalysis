from run_chains4 import *
from gerrychain import Graph

iters = 1000
storage_ratio = 100

g = list()
assignments = list()
for i in range(100000, 1100000, 100000):
    graph = Graph.from_json('AlternativePlans/graph_recom_'+str(i)+'.json')
    g.append(graph)
    assignment = dict()
    for j in graph.nodes:
        assignment[j] = graph.nodes[j]['US_Distric']
    assignments.append(assignment)

d = list()
a = list()
for assignment in assignments:
    i = int(np.round(time.time(), 0))
    c = Chain('flip-mh', iters, starting_assignment=assignment, storage_ratio=storage_ratio)
    d.append(c.data)
    a.append(c.stored_assignments)

d_comb = pd.concat(d, ignore_index=True)
a_comb = pd.concat(a, ignore_index=True)

d_comb.to_parquet('10chains'+str(iters)+'d.parquet.gzip', compression='gzip')
a_comb.to_parquet('10chains'+str(iters)+'a.parquet.gzip', compression='gzip')
