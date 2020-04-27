import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functions import *

#np.random.seed(0)

temp_N = 15
G = nx.erdos_renyi_graph(temp_N, 3 / temp_N)

N = len(G) #total number of nodes

run_number = 1
run_time = 10
transmit_prob = 0.25
recovery_prob = 0.25
learning_rate = 0.5
beta = 0.8


agents = np.zeros((N), dtype=[('health', int), ('future', int),\
('strategy',int), ('social_class',int), ('reward',float)] )

social_class_num = 3

stay_home_reward = np.array([-2.8, -1.5, -1])
infection_reward = -3
exp_stay_home_reward = tuple(np.exp(beta * stay_home_reward))
#exponential 

#init

agents['social_class'] = np.random.randint(0, 3, N)

agents['strategy'] = 1 #initially they all choose to go out.
infection_seed = np.random.randint(0, N) #the first infected node
agents['health'][infection_seed] = 1
agents['future'][infection_seed] = 1

#agents['social_class'] = 0

#health : 0 -> suceptible
#health > 0 -> number of days after infection
#health : -1 -> recovered (removed)
#future -> health in the next step (necessary for parallel updating)
#strategy -> going out or not
#social class -> 0, 1, 2 respectively low, medium and high economic class


infected_num = 1
survivor_num = 1
prediction = 1
if __name__ == "__main__":
    for run in range(run_number):
        for t in range(run_time):
            if infected_num >= 1 and survivor_num >= 1: #if there are infectious agents and also if not everyone is infected
                infected_num = infect(G, agents, transmit_prob) #nodes infect their neighbors
                newly_recovered = recover(agents, recovery_prob) #nodes get recovered
                update_infection(agents) #actually change the health statuses (necessary for parallel updating)
                prediction = predict_infected_num(agents, prediction, learning_rate) #predict the number of upcoming infected agents for the next step

                survivor_num = update_strategy(agents, exp_stay_home_reward, prediction * infection_reward, beta) #update strategies (going out and staying in)