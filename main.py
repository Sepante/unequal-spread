import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
from scipy.stats import itemfreq

#np.random.seed(0)

temp_N = 40
G = nx.erdos_renyi_graph(temp_N, 2.3 / temp_N)

N = len(G) #total number of nodes

run_number = 20
run_time = 20000000
transmit_prob = 0.5
recovery_prob = 0.25
learning_rate = 0.5
beta = 1.3


agents = np.zeros((N), dtype=[('health', int), ('future', int),\
('strategy',int), ('social_class',int)] )

social_class_num = 3

stay_home_reward = np.array([-2.8, -1.5, -1])
infection_reward = -3
exp_stay_home_reward = tuple(np.exp(beta * stay_home_reward))
#exponential 

#init

agents['social_class'] = np.random.randint(0, 3, N)

#health : 0 -> suceptible
#health > 0 -> number of days after infection
#health : -1 -> recovered (removed)
#future -> health in the next step (necessary for parallel updating)
#strategy -> going out or not
#social class -> 0, 1, 2 respectively low, medium and high economic class



results = pd.DataFrame( np.zeros((run_number, social_class_num), int) )
results.columns = ['class_'+str(i) for i in range(social_class_num) ]
if __name__ == "__main__":
    for run in range(run_number):
        infected_num = 1
        survivor_num = 1
        prediction = 1

        #agents = init_agents(agents, N)
        init_agents(agents, N)
        for t in range(run_time):
            if infected_num >= 1 and survivor_num >= 1: #if there are infectious agents and also if not everyone is infected
                infected_num = infect(G, agents, transmit_prob) #nodes infect their neighbors
                newly_recovered = recover(agents, recovery_prob) #nodes get recovered
                update_infection(agents) #actually change the health statuses (necessary for parallel updating)
                prediction = predict_infected_num(agents, prediction, learning_rate) #predict the number of upcoming infected agents for the next step

                survivor_num = update_strategy(agents, exp_stay_home_reward, prediction * infection_reward, beta) #update strategies (going out and staying in)
                                
                
        results.iloc[run] = get_results(agents, social_class_num)

results.to_csv('infected_for_each_class.csv')