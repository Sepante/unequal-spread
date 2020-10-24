import numpy as np
import networkx as nx
import pandas as pd
from networkx import stochastic_block_model
import matplotlib.pyplot as plt

def generate_network(n, on_the_run_tweaks):    
    P_norm_dat = pd.read_csv('P_norm.csv', index_col = 0)
    
    P_norm = np.array(P_norm_dat)
    
    global sizes
    #sizes = list([Asi_population, Lat_population, Bla_population, Whi_population])
    #sizes = list([Whi_population, Bla_population, Asi_population, Lat_population])
    sizes = np.array( pd.read_csv('Population_fraction.csv') )[0]
    sizes = sizes / sizes.sum() * n
    sizes = sizes.astype('int')
    
    probs = P_norm
    
    if on_the_run_tweaks:
        call = 57.5
        cb = 1.08
        ca = 0.83
        cl = 1.04
        probs /= call

        probs[1,:] /= cb
        probs[:,1] /= cb
        
        probs[2,:] /= ca
        probs[:,2] /= ca
        
        probs[3,:] /= cl
        probs[:,3] /= cl
    #print(P_norm_dat)
    #plt.imshow(P_norm_dat)
    #plt.colorbar()
    #plt.show()
    P_norm_dat[:] = probs
    #print(P_norm_dat)
    #plt.imshow(P_norm_dat)
    #plt.colorbar()
    P_norm_dat.to_csv('P_norm_adj.csv')
    print(probs)
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
    g = generate_network(1000, True)
    block_list = np.array( [g.nodes[i]['block'] for i in range(len(g))] )
    
    group_ordering = ['W', 'B', 'A', 'L']
    
    deg_array = (np.array(  list( dict(nx.degree(g)).values()  )) )
    for i in range(4):
        group = ( block_list == i )
        print(group_ordering[i], deg_array[group].mean() )
    print('All ', deg_array.mean())
        