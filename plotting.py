# Import necessary packages
from gerrychain import (GeographicPartition, Partition, Graph)
import pandas as pd
import geopandas as gp
import numpy as np
from scipy import optimize as opt
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString

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
    if save: plt.savefig(savetitle, dpi=dpi)
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

# Analyzing the rejection rate of a chain
def acceptance_series(d):
    """
    Creates an np.array with length d.shape[0], where the ith entry is 1 if
    rows i and i-1 of d are different, and 0 otherwise.

    Parameters:
        d (iterable)

    Returns:
        s (np.array)
    """
    series = np.zeros(d.shape[0], dtype=np.uint8)
    for i in range(1, d.shape[0]):
        if not np.allclose(d.iloc[i, :], d.iloc[i-1, :]):
            series[i] = 1
    return series

def running_mean(x, N):
    """
    Returns a moving average array of the data in x over an N-period interval.

    Parameters:
        x (iterable)

    Returns:
        m (np.array)
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_acceptance_rate(data, period):
    """
    Plot the (period)-moving-average of the acceptance rate of a chain.

    Parameters:
        data (pd.DataFrame)
        period (int): the number of iterations to average over
    """
    s = acceptance_series(data)

    # Position the moving average at the center of the period it is the average over
    plt.plot(np.linspace(period/2, len(s)-period/2, len(s)-period+1), running_mean(s, period))

    plt.title('{}-Iteration Moving Average Acceptance Rate'.format(period))
    plt.ylabel('Acceptance Rate')
    plt.xlabel('Iteration')
    plt.axis([0, len(s), 0, 1])
    plt.show()

def make_plots(idnum, kind, subdirectory='Plots/', figsize=(8,6), dpi=400, file_type='.pdf'):
    """
    Given the id number of a chain run, creates relevant plots (Utah data format only).

    Parameters:
        idnum (int): the Unix timestamp of the second when the chain was created
                    (part of the filename)
                    note: if idnum is string, will use the given name of the file
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
    if type(idnum) == int:
        if idnum < 1593561600:
            data = pd.read_hdf(str(idnum)+'.h5', 'data')
        else:
            data = pd.read_parquet(str(idnum)+'d.parquet.gzip')
    else:
        data = pd.read_hdf(idnum)

    # Set parameters
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


def make_correlation_plots(idnum, kind, subdirectory='Plots/', figsize=(8,6), dpi=400, file_type='.pdf'):
    """
    Produces a set of correlation plots used to analyze how well the partisan gerrymandering metrics
    perform in the case of Utah.

    Parameters:
        idnum (int): the unix timestamp for when the chain was started.
            if passed in as str, the filename of the chain
        subdirectory (str): the subdirectory to save the resulting plots

    Total: 15 plots.
    """
    assert kind in ['flip-uniform', 'flip-mh', 'recom-uniform', 'recom-mh']

    # Extract the data
    if type(idnum) == int:
        if idnum < 1593561600:
            data = pd.read_hdf(str(idnum)+'.h5', 'data')
        else:
            data = pd.read_hdf(idnum)
    else:
        data = pd.read_parquet(idnum)
        idnum = ''




    # Set parameters
    common_file_ending = '-'+str(len(data))+'-'+kind+'-'+str(idnum)+file_type

    # Set parameters
    params = {'figsize':figsize, 'dpi':dpi, 'save':True}
    n = len(data)


    correlationplot_xaxis = {'Avg Abs Partisan Dislocation': {'name': 'Average Absolute Partisan Dislocation', 'savetitle':'AvgAbsPD'},
                             'Efficiency Gap': {'name':'Efficiency Gap', 'savetitle':'EG'},
                             'Mean Median': {'name':'Mean Median Score', 'savetitle':'MM'},
                             'Partisan Bias': {'name':'Partisan Bias Score', 'savetitle':'PB'},
                             'Partisan Gini': {'name':'Partisan Gini Score', 'savetitle':'PG'}}

    correlationplot_yaxis = {' - G': 'Sorted GRep Vote Share 1', ' - SEN':'Sorted SenRep Vote Share 1', ' - COMB':'Sorted CombRep Vote Share 1'}

    # Construct plots for the various metrics
    for key, val in {' - G': ' (Gubernatorial 2010)', ' - SEN':' (Senate 2010)', ' - COMB':' (Combined 2010)'}.items():
        for key1 in correlationplot_xaxis.keys():
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            x = np.array(data[key1+key])
            y = np.array(data[correlationplot_yaxis[key]])

            SStot = np.sum(np.square(y-np.mean(y)))
            p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
            m, c = p[0], p[1]
            SSres = np.sum(residuals)
            R2 = 1-SSres/SStot
            domain = np.linspace(np.min(x), np.max(x), 200)

            plt.scatter(data[key1+key], data[correlationplot_yaxis[key]], s=1, alpha=0.3, label='Data')
            plt.plot(domain, m*domain+c, label='Best Fit, R^2={}, m={}'.format(np.round(R2,2), m), c='orange')
            ax.set_title(correlationplot_xaxis[key1]['name']+' and R Vote Share in Least R District in a {}-Plan Ensemble'.format(n)+val)
            ax.set_xlabel(correlationplot_xaxis[key1]['name'])
            ax.set_ylabel('R Vote Share in Least R District')
            plt.legend(loc='upper right')
            plt.savefig(subdirectory+correlationplot_xaxis[key1]['savetitle']+'Correlation'+common_file_ending, dpi=dpi, bbox_inches='tight')
            plt.clf()

            print('Finished Plot')

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
    if idnum < 1593561600:
        assignments = pd.read_hdf(str(idnum)+'.h5', 'stored_assignments')
    else:
        assignments= pd.read_parquet(str(idnum)+'a.parquet.gzip')
    n = len(assignments)

    # Compute the changes
    changes = np.array(assignments.iloc[1:, :]) - np.array(assignments.iloc[:-1, :])
    move_frequencies = np.count_nonzero(changes, axis=1)

    # Save if desired
    if save: np.save(subdirectory+str(i)+'moves.npy', move_frequencies)

    # Result the result if desired
    return np.count_nonzero(changes, axis=1)
