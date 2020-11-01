import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def init_agents(agents, N):
    #global agents
    agents['strategy'] = 1 #initially they all choose to go out.
    agents['health'] = 0 #initially they all choose to go out.
    agents['future'] = 0 #initially they all choose to go out.
    infection_seed = np.random.randint(0, N) #the first infected node
    #print(infection_seed)
    agents['health'][infection_seed] = 1
    agents['future'][infection_seed] = 1


def infect(G, agents, transmit_prob):
    health_list = (agents['health'] > 0)
    #strategy_list = (agents['strategy'] == 1)
    out_and_infected = np.all( [health_list, agents['strategy']], axis = 0 )
    infected_agents = np.where( out_and_infected )[0] #select infectious agents
    #print(len(infected_agents))
    for infectious in infected_agents:
        for neighbor in G.neighbors(infectious):
            if (agents['future'][neighbor] == 0 and agents['strategy'][neighbor] == 1): #if susceptible and out
                #print(neighbor)
                if np.random.random() < transmit_prob:
                    agents['future'][neighbor] = 1 #infect
    return np.sum(health_list)


def recover(agents, recovery_prob):
    infected_agents = np.where( (agents['health'] > 0) )[0]
    #print(len(infected_agents))
    future_status = np.random.choice([-1, 1], len(infected_agents), p=[recovery_prob, 1- recovery_prob] )
    
    agents['future'][infected_agents] = future_status
    
    return np.sum(future_status == -1)


def update_infection(agents):
    agents['future'][( agents['health'] == -1 )] = -2
    agents['health'][( agents['health'] == -1 )] = -2
    
    agents['health'] = agents['future']
    
def get_newly_recovered_agents(agents, social_class = -1, compartment_to_look_at = 1):
    if social_class == -1:
        return agents['health'] == compartment_to_look_at
    else:
        return np.all( [ agents['social_class'] == social_class , agents['health'] == compartment_to_look_at ], axis = 0 )
    
    
    
def predict_infected_num(N, agents, t_prediction, learning_rate):
    #print('inside = ', t_prediction)

    newly_infected = np.sum( get_newly_recovered_agents(agents, social_class = -1, compartment_to_look_at = 1) ) / N
    #if newly_infected:
        #print("newly: ", newly_infected)
    
    ## only averages for lowering situation, for increases considers the new value.
#    if newly_infected > t_prediction:
#        new_prediction = newly_infected
#    else:
#        new_prediction = t_prediction * (1-learning_rate ) + newly_infected * learning_rate
#    return new_prediction
    new_prediction = t_prediction * (1-learning_rate ) + newly_infected * learning_rate
    return new_prediction

    
def update_strategy(agents, exp_stay_home_reward, infection_reward_times_infected_num, beta):
    #survivor_num = np.sum( agents['health'] >= 0 )
    #survivor_num = len(agents)
    survivor_num = 1
    
    going_out_mean_reward = infection_reward_times_infected_num / survivor_num
    
    exp_going_out_reward = np.exp(beta * going_out_mean_reward)
    for social_class in range( len(exp_stay_home_reward) ):
        this_class_agents = (agents['social_class'] == social_class)
        
        staying_home_prob = exp_stay_home_reward[social_class]/ (exp_stay_home_reward[social_class] + exp_going_out_reward)
        #print (social_class, staying_home_prob)
        
        agents['strategy'][ this_class_agents ] = np.random.choice([0, 1], np.sum(this_class_agents), p=[staying_home_prob, 1- staying_home_prob] )
    #print(infection_reward_times_infected_num , staying_home_prob)
    #print(staying_home_prob)
    return survivor_num

def get_results(agents, social_class_num):
        finally_infected = agents['health'] < 0
        infected_classes = agents[finally_infected]['social_class']
        #print(infected_classes)
        infected_from_each_class = np.zeros(social_class_num, int)
        for social_class in range( social_class_num ):
            infected_from_each_class[social_class] = np.sum( infected_classes == social_class )
        return infected_from_each_class
    
def get_timed_results(agents, social_class_num):
    
    currently_infected = agents['health'] < 0
    infected_classes = agents[currently_infected]['social_class']
    infected_from_each_class = np.zeros(social_class_num, int)
    for social_class in range( social_class_num ):
        infected_from_each_class[social_class] = np.sum( infected_classes == social_class )
    return infected_from_each_class


