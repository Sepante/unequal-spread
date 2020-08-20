import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
#from scipy.stats import itemfreq
from network_generator import generate_network
from simulate import simulate
import multiprocessing as mp
from connectivity_calc import connectivity_calc

run_number = 1

N = 1000
social_class_num = 2
seg_frac = 0 #between 1 and zero

#sizes, probs = connectivity_calc(N, social_class_num, seg_frac)


#Spreading Model
#transmit_prob = 0.5
recovery_prob = 0.25


#MODEL = 3 #0: n





#Game Theory Model
learning_rate = 0.5
beta = 1
infection_reward = -2.5


stay_home_reward = np.array([-0.6, -1.9, -1, -1.6]) #W - B - A - L






#block_list = np.array( [G.nodes[i]['block'] for i in range(len(G))] )



#if MODEL == 1: # segregation only
#    stay_home_reward = np.array([-1.312, -1.312, -1.312, -1.312])#(sizes * stay_home_reward).sum()
#    
#elif MODEL == 2: # SES only
#    np.random.shuffle( block_list )
#    
#elif MODEL == 0: # no segregation no SES no NOTHING!
#    stay_home_reward = np.array([-1.312, -1.312, -1.312, -1.312])#(sizes * stay_home_reward).sum()
#    np.random.shuffle( block_list )
#
    
#init

#agents['social_class'] = np.random.randint(0, 3, N)
#exp_stay_home_reward = tuple(np.exp(beta * stay_home_reward))

#agents['social_class'] = block_list

#health : 0 -> suceptible
#health > 0 -> number of days after infection
#health : -1 -> recovered (removed)
#future -> health in the next step (necessary for parallel updating)
#strategy -> going out or not
#social class -> 0, 1, 2 respectively low, medium and high economic class

jobs = []
#seg_frac_seq = np.arange( 0, 1, 0.2 )
#transmit_prob_seq = np.arange( 0.2, 1, 0.1 )
seg_frac_seq = [0.5, 0.6]
transmit_prob_seq = [ 0.1 ]

for seg_frac in seg_frac_seq:
    sizes, probs = connectivity_calc(N, social_class_num, seg_frac)
    for transmit_prob in transmit_prob_seq:
        args = (sizes, probs, seg_frac, social_class_num, beta, stay_home_reward, infection_reward\
                , learning_rate, transmit_prob, recovery_prob, seg_frac)    
        
        for run in range(run_number):
            jobs.append( ( args + (np.random.randint(10000000),) ) )

print('Jobs Done!')
##adding the random seeds.
#jobs = [ ( args + (np.random.randint(10000000),) )  for i in range(run_number)]


params_titles = ['transmit_prob', 'segregation', 'SES_dispar', 'size_dispar' ]
results = pd.DataFrame( np.zeros(( len(jobs) , social_class_num + len(params_titles)), int) )
results.columns = params_titles + ['class_'+str(i) for i in range(social_class_num) ]


with mp.Pool(mp.cpu_count()) as pool:
    p_r = pool.map_async(simulate, jobs)
    res = p_r.get()

results[:] = res

rand_string = str(np.random.randint(100000000))

id_string = 'rewards=' + str(stay_home_reward) + '-infect_rew='  + '-recov =' + str(recovery_prob) + '-' + rand_string + '.csv'


results.to_csv('Results/' + id_string)
