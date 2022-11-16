# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 02:43:32 2022

@author: pandi
"""
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def ER_graph(n,p):
    B = nx.Graph()
    #possible combinations
    all_comb = [i for i in combinations(list(range(1,n+1)),2) if i[0] != i[1]]
    ind = np.random.choice(len(all_comb), int(p*len(all_comb)))
    
    edges = [all_comb[i] for i in ind]
    B.add_nodes_from(list(range(n)))
    B.add_edges_from(edges)
    nx.draw(B, with_labels = 1)

# For n = 50
n = 50
p1 = 1/(n**2)
p2 = 1/n
p3 = np.log(n)/n

for i,j in zip(range(3),[p1,p2,p3]):
    plt.figure(i)
    np.random.seed(i)
    ER_graph(n, j)


n = 150
p1 = 1/(n**2)
p2 = 1/n
p3 = np.log(n)/n

for i,j in zip(range(3),[p1,p2,p3]):
    plt.figure(i+3)
    np.random.seed(i+3)
    ER_graph(n, j)

