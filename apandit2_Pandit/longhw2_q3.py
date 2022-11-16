# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:50:58 2022

@author: pandi
"""
import networkx as nx
import glob
import time
from queue import Queue
import numpy as np
import networkx.algorithms.community as nx_comm
import itertools
import pandas as pd

filenames = [x for y in ['Barabasi', 'WattsStrogatz','ErdosRenyi'] \
            for x in glob.glob('*.txt') if y in x]

def clean_data(file):
    data = pd.read_csv(file, sep=",", lineterminator=")")
    data.columns.values[0] = data.columns.values[0][1:]
    data.iloc[:, 0] = data.iloc[:, 0].str.replace('^\n\(', '')
    data = data.drop([data.shape[0]-1])
    data.rename(columns = dict(zip(data.columns,['orig','dest'])), inplace = True)
    data = data.astype({'dest':'int64',})
    data = data.astype({'orig':'str','dest':'str'})
    return data

def bfs_search(file,B,source):
    graph_list = {}
    for node in B.nodes():
       graph_list[node] = [x[1] for x in B.edges(node)] 
    data = clean_data(file)
    visited = {}
    level = {}
    parent = {}
    bfs_traversal_order = []
    queue = Queue()
    for node in graph_list:
        visited[node] = False
        parent[node] = None
        level[node] = -1
    
    s = source
    visited[s] = True
    level[s] = 0
    queue.put(s)
    
    while not queue.empty():
        u = queue.get()
        bfs_traversal_order.append(u)
        
        for v in graph_list[u]:
            if not visited[v] == True:
                visited[v] = True
                parent[v] = u
                level[v] = level[u] + 1
                queue.put(v)
    print(bfs_traversal_order)

def girvan_newman_algo(filenames, target = 5):
    for file in filenames:
        start = time.time()
        data = clean_data(file)
        B = nx.Graph()
        B.add_edges_from([(row['orig'],row['dest']) for index,row in data.iterrows()])
        modularity = []
        comm_list = [set(itertools.chain.from_iterable(B.edges()))]
        modularity.append(nx_comm.modularity(B, comm_list))
        counter = 0
        edge_counter = [counter]
        community = 1
        while community != target:
            btw_overall = {x:0 for x in B.edges()}
            #Dividing all betweenness centrality values by 2 
            nodes = [x for y in comm_list for x in y]
            comm_list, edge_counter, community,modularity = comm_generator(B, nodes, edge_counter, \
                                counter, community, comm_list, modularity)
            
            for i,node in enumerate(nodes):
            #Perform bfs search with every node in the graph
                bfs = nx.bfs_tree(B, node)
                if len(B.edges(node)) == 0:
                    continue
                elif set(bfs.nodes()) in comm_list:
                    btw_dict = nx.edge_betweenness_centrality(bfs)
                    btw_overall = {k: btw_dict.get(k, 0) + btw_overall.get(k, 0) \
                               for k in set(btw_dict)}

            btw_overall = {k: v / 2 for k, v in btw_overall.items()}
            max_key = max(btw_overall, key = btw_overall.get)
            B.remove_edge(*max_key)
            counter += 1
        print(file.split('.')[0])    
        cols = ['Number of communities','Cumulative no of edges removed', 'Modularity']
        result = pd.DataFrame(dict(zip(cols,[list(range(1,6)),edge_counter,modularity])))
        result.to_latex(file.split('.')[0] + '.tex', index = False)
        end = time.time()
        print(end - start)
    return 

def comm_generator(B,nodes, edge_counter, counter, community, comm_list, modularity):
    ind = 1
    truth = []
    trigger = 0
    while ind != -1:
        for i, node in enumerate(nodes):
                bfs = nx.bfs_tree(B, node)
                truth.append(set(bfs.nodes()) in comm_list)
                if not truth[-1]:
                    trigger = 1
                    break
        if trigger == 1:
            rn_node = np.random.choice(list(bfs.nodes()))
            # print(community,i)
            sublist = set([x for y in comm_list for x in y if rn_node in y])
            comm_list.remove(sublist)
            comm_list.append(set(bfs.nodes())) 
            comm_list.append(set(sublist - set(bfs.nodes())))
            community +=1
            edge_counter.append(counter)
            mod = nx_comm.modularity(B,comm_list)
            modularity.append(mod)
        else:
            ind = -1
    return [comm_list, edge_counter, community, modularity]