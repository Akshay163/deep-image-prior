# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 22:51:18 2022

@author: pandi
"""
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Using the data file to create a graph
data = pd.read_csv('data.txt', sep = ' ', header = None)
data.rename(columns = dict(zip(data.columns,['orig','dest'])),inplace = True)
B = nx.DiGraph()
B.add_edges_from([(row['orig'],row['dest']) for index,row in data.iterrows()])

def basic_pagerank(B,k = 100):
    #Finding unique nodes
    nodes = list(B.nodes())
    nodes.sort()
    #Initializing the pagerank algorithm
    pagerank = dict(zip(nodes, [1]*len(nodes)))
    #Variable to test the convergence of the algorithm 
    diff = pagerank.copy()
    for i in range(k):
        for node in nodes:
            tmp_sum = 0
            in_nodes = [x[0] for x in B.in_edges(node)]
            for j in in_nodes:
                tmp_sum += pagerank[j]/len(B.out_edges(j))
            diff[node] = abs(pagerank[node] - tmp_sum)
            pagerank[node] = tmp_sum
        
        if sum(diff.values()) <= 1e-16:
            break
    
    pagerank = pd.DataFrame(pagerank.items(), columns=['nodes', 'pagerank'])
    
    if i != k-1:
        print(f'stops early at iteration {i}')
        pagerank.to_latex('basic_pagerank.tex', index = False)
        return [pagerank, i]
    else:
        pagerank.to_latex('basic_pagerank.tex', index = False)
        return [pagerank, i]
    
def scaled_pagerank(B,k = 100, alpha = 0.85):
    #Finding unique nodes
    nodes = list(B.nodes())
    nodes.sort()
    #Initializing the pagerank algorithm
    pagerank = dict(zip(nodes, [1]*len(nodes)))
    #Variable to test the convergence of the algorithm 
    diff = pagerank.copy()
    for i in range(k):
        for node in nodes:
            tmp_sum = 0
            in_nodes = [x[0] for x in B.in_edges(node)]
            for j in in_nodes:
                tmp_sum += pagerank[j]/len(B.out_edges(j))
            diff[node] = abs(pagerank[node] - tmp_sum)
            pagerank[node] = (1 - alpha)/len(nodes) + alpha*tmp_sum
        
        if sum(diff.values()) <= 1e-16:
            break
    
    pagerank = pd.DataFrame(pagerank.items(), columns=['nodes', 'pagerank'])
    
    if i != k-1:
        print(f'stops early at iteration {i}')
        pagerank.to_latex('scaled_pagerank.tex', index = False)
        return [pagerank, i]
    else:
        pagerank.to_latex('scaled_pagerank.tex', index = False)
        return [pagerank, i]
    