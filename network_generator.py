import numpy as np
import networkx as nx
import pandas as pd
from networkx import stochastic_block_model
import matplotlib.pyplot as plt

def generate_network(n):
    
    P_norm = pd.read_csv('P_norm.csv', index_col = 0)
    
    P_norm = np.array(P_norm)
    #print(P_norm)
    P_norm /= 60

    # n = 1000
    Asi_population = int(0.08 * n)
    Lat_population = int(0.29 * n)
    Bla_population = int(0.30 * n)
    Whi_population = int(0.33 * n)
    
    #sizes = list([Asi_population, Lat_population, Bla_population, Whi_population])
    sizes = list([Whi_population, Bla_population, Asi_population, Lat_population])
    
    # probs = [[AA , AL, AB, AW ], # Asian
    #          [AL, LL , LB, LW ], # Latino
    #          [AB, LB, BB, BW ],# Black
    #          [AW, LW, BW, WW]] # White
    #
    probs = P_norm
    g = stochastic_block_model(sizes, probs, sparse=True)
    
    # neigh = np.array(list(map(lambda i: len(list(g.neighbors(i))), range(len(g)))))
    # isolated_neighbors = []
    # if len(np.argwhere(neigh == 0)) != 0: isolated_neighbors = np.argwhere(neigh == 0)[:, 0]
    #
    # for i in isolated_neighbors:
    #     g.add_edge(i, np.random.randint(0, n))
    #     g.add_edge(i, np.random.randint(0, n))
    return g

if __name__ == "__main__":

    #g = generate_network(1000, P_norm)
    g = generate_network(1000)
    block_list = np.array( [g.nodes[i]['block'] for i in range(len(g))] )
    
    deg_array = (np.array(  list( dict(nx.degree(g)).values()  )) )
    for i in range(4):
        group = ( block_list == i )
        print( deg_array[group].mean() )
    print(deg_array.mean())
    