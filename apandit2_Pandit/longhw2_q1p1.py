# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:23:15 2022

@author: pandi
"""
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# HITS algorithm

def HITS(file,K = 10):

    data = pd.read_csv(file, sep = ',', header = 0).reset_index()
    data = data.rename(columns = dict(zip(data.columns.tolist(),\
        ['hubs', 'authorities'])))
    data.hubs = data.hubs.astype('str')
    B = nx.Graph()
    
    B.add_nodes_from(data.hubs.unique(), bipartite = 0)
    B.add_nodes_from(data.authorities.unique(), bipartite = 1)
    
    B.add_edges_from([(row['hubs'],row['authorities']) for index,row in data.iterrows()])
    
    X, Y = bipartite.sets(B)
    
    hubs_score = dict(zip(list(X),[1]*len(X)))
    authorities_score = dict(zip(list(Y),[1]*len(Y)))
    
    #Running for K iterations
    
    for k in range(K):    
        # update step
        #authority update rule
        for _,x in enumerate(authorities_score):
            authorities_score[x] = sum([hubs_score[k[1]] for k in list(B.edges(x))])
        
        #hub update rule
        for _,y in enumerate(hubs_score):
            hubs_score[y] = sum([authorities_score[k[1]] for k in list(B.edges(y))])
        
        #normalization step
        authorities_score = dict(zip(list(authorities_score.keys()),list(np.\
        array(list(authorities_score.values()))/sum(authorities_score.values()))))
            
        hubs_score = dict(zip(list(hubs_score.keys()),list(np.\
        array(list(hubs_score.values()))/sum(hubs_score.values()))))
    
    #Sorting values in descending order using dataframe
    
    authorities_score = pd.DataFrame(data = authorities_score.items(),\
    columns = ['authorities','auth_score'])
    authorities_score.sort_values('auth_score', ascending = False, inplace = True)
    authorities_score = authorities_score.reset_index(drop = True)
    hubs_score = pd.DataFrame(data = hubs_score.items(),\
    columns = ['hubs','hub_score'])
    hubs_score.sort_values('hub_score', ascending = False, inplace = True)
    hubs_score = hubs_score.reset_index(drop = True)
    #Creating a datafrma of top3 hubs and authorities with their scores
    top3 = pd.concat([hubs_score.head(3),authorities_score.head(3)], axis = 1)   
    return top3

tdata1_out = HITS('HITS_input1.txt')
tdata2_out = HITS('HITS_input2.txt')

tdata1_out.to_csv('input1_solution.csv',index = False)
tdata2_out.to_csv('input2_solution.csv',index = False)