import numpy as np
import random as old_rd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
#from scipy.stats import itemfreq
#from network_generator import generate_network


def simulate(args):
    #results = pd.DataFrame( np.zeros((run_number, social_class_num), int) )
    #results.columns = ['class_'+str(i) for i in range(social_class_num) ]
    sizes, probs, seg_frac, social_class_num, beta\
    , stay_home_reward, infection_reward, learning_rate\
    , transmit_prob, recovery_prob, seg_frac, uniform_reside\
    , random_seed = args
    
    N = sizes.sum()
    
    #seeding the random to obtain different results
    old_rd.seed(random_seed)
    np.random.seed(random_seed)

    
    G = nx.stochastic_block_model(sizes, probs, sparse=True)
    agents = np.zeros((N), dtype=[('health', int), ('future', int),\
                      ('strategy',int), ('social_class',int)] )
    block_list = np.array( [G.nodes[i]['block'] for i in range(len(G))] )
    
    if uniform_reside:
        np.random.shuffle( block_list )
    
    agents['social_class'] = block_list


    
    run_number = 1 #temp
    
    exp_stay_home_reward = tuple(np.exp(beta * stay_home_reward))

    for run in range(run_number):
        rand_string = str(np.random.randint(100000000))
        #print(rand_string)

        infected_num = 1
        survivor_num = 1
        prediction = 1

        #agents = init_agents(agents, N)
        init_agents(agents, N)
        run_time = 20000000 # Just means long enough.
        for t in range(run_time):
            if infected_num >= 1 and survivor_num >= 1: #if there are infectious agents and also if not everyone is infected
                infected_num = infect(G, agents, transmit_prob) #nodes infect their neighbors
                newly_recovered = recover(agents, recovery_prob) #nodes get recovered
                update_infection(agents) #actually change the health statuses (necessary for parallel updating)
                prediction = predict_infected_num(agents, prediction, learning_rate) #predict the number of upcoming infected agents for the next step

                survivor_num = update_strategy(agents, exp_stay_home_reward, prediction * infection_reward, beta) #update strategies (going out and staying in)
                                
                
        #params_titles = ['transmit_prob', 'segregation', 'SES_dispar', 'size_dispar', 'uniform_reside' ]
        params = [transmit_prob, seg_frac, stay_home_reward[0] - stay_home_reward[-1], sizes[0] - sizes[-1], uniform_reside ]
        print ( np.array( params + list( get_results(agents, social_class_num) ) ) )
        return np.array( params + list( get_results(agents, social_class_num) ) )

    #rand_string = str(np.random.randint(100000000))
    #id_string = 'infected_for_each_class, '+ 'Model =' + str(MODEL) + ', p ='+ str(transmit_prob) + ', r =' + str(recovery_prob) + ', ' + rand_string + '.csv'
    
    
    #results.to_csv('Results/' + id_string)
