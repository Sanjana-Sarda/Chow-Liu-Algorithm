import sys
import networkx as nx
import numpy as np
import pandas as pd
from scipy import special as sp
from collections import defaultdict
import time


def prior (sumM0, vars, G):
    n = len(vars)
    r = vars.tolist()
    q = np.ones(n)
    for a in range (n):
        for b in G.predecessors(a+1):
            q[a] = q[a]*r[b-1] 
    alpha = {}
    for a in range (n):
        alpha[a] = np.ones((int(q[a]), r[a])) 
    suma0 = defaultdict(int)
    for key in sumM0.keys():
        suma0[key] = r[key[0]]
    return suma0
    

def statistics (vars, G, D):
    n = len(vars)
    r = vars.tolist()
    q = np.ones(n)
    for a in range (n):
        for b in G.predecessors(a+1):
            q[a] = q[a]*r[b-1]  
    M = defaultdict(int)
    sumM0 = defaultdict(int)
    for index, row in D.iterrows():
        for i in range(n):
            k = row[i]
            parents = list(G.pred[i+1])
            j = 1
            if parents:
                j = row[parents[0]-1]
            ind1 = (i, j, k) 
            M[ind1] +=1
            ind2 = (i, j)
            sumM0[ind2] +=1
    suma0 = prior(sumM0, vars, G)
    return (M, sumM0, suma0)

    
def bayesian_score_component(stats):
    M, sumM0, suma0 = stats
    p = 0
    for key in M.keys():
        p += sp.loggamma(1 + M[key]) 
    for key in suma0.keys():
        p +=  - sp.loggamma(suma0[key] + sumM0[key]) + sp.loggamma(suma0[key])
    return p


def bayesian_score(vars, G, D):
    return bayesian_score_component(statistics(vars, G, D))


def marginal_distribution (D, i):
    c = 1./len(D.index)
    val = defaultdict(float)
    for index, row in D.iterrows():
        val[row[i]] += c
    return val


def marginal_pair_distribution (D, i, j):
    c = 1./len(D.index)
    val = defaultdict(float)
    if (i>j):
        i, j = j, i
    for index, row in D.iterrows():
        ind = (row[i], row[j])
        val[ind] += c
    return val


def mutual_information(D, i, j):
    marg_val_i = marginal_distribution (D, i)
    marg_val_j = marginal_distribution (D, j)
    marg_val_pair = marginal_pair_distribution (D, i, j)
    weight = 0
    for xi, pxi in marg_val_i.items():
        for xj, pxj in marg_val_j.items():
            if (xi, xj) in marg_val_pair:
                pxij = marg_val_pair[(xi, xj)]
                weight = weight + pxij*(np.log(pxij) - (np.log(pxi) + np.log(pxj)))    
    return weight


def make_undirected_gph (D, vars): #chow-liu tree 
    ag = nx.DiGraph()
    n = len(vars)
    for j in range (n):
        for i in range (j): 
            ag.add_edge(i+1, j+1, weight = mutual_information(D, i, j))
    nx.draw(ag, with_labels=True)
    ag = ag.to_undirected()
    ag = nx.minimum_spanning_tree(ag)
    return ag


def make_directed_gph (ag, root):
    dag = nx.DiGraph()
    dag.add_node(root)
    dag.add_edges_from(ag.edges)
    return dag
                

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]-1], idx2names[edge[1]-1]))


def compute(infile, outfile):
    D = pd.read_csv(infile) #dataset
    vars = D.nunique(axis=0)
    start = time.time()
    ag = make_undirected_gph(D, vars)
    write_gph(make_directed_gph(ag, 1), D.columns, outfile)
    print (time.time()-start)
    
    
    
def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
