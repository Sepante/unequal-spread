import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib
from matplotlib import gridspec
#matplotlib.rcParams.update({'font.size': 11})
import networkx as nx

from main import *
do_animate = True
graph_type = 'erdos'
#eps = -0.25
pos = {}
if graph_type == 'erdos':
    pos = nx.circular_layout(G)
    #pos = nx.kamada_kawai_layout(G)

    #for i, p in enumerate(pos):
        #if i % 3 == 0:
            #pos[i] = pos[i] * 1.5
            #print('0')
        #elif (i % 3 == 1):
            #pos[i] = pos[i] * 2
            #print('1')
        #else:
            #pos[i] = pos[i] * 0
            #print('last2')
    True
elif graph_type == 'grid':
    #pos = nx.circular_layout(G)
    dist = 1 / L
    for i in G.nodes():
        x = i % L
        y = int( i / L )
        pos[i] = np.array([ x * dist, y * dist ])# + np.array( [eps , eps] )

pos_array = np.array( list( pos.values() ) )

#node_size = 80000

#fig, axes= plt.subplots(1,2, figsize=(5,5))
#fig.subplots_adjust(wspace = 0)
#ax, ay = axes
figure_ratio = 0.75
fig = plt.figure(figsize=(5*(0.9+figure_ratio), 5 ) )
fig.subplots_adjust(wspace = 0)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, figure_ratio]) 
ax = plt.subplot(gs[0])

#ax0.plot(x, y)
ay = plt.subplot(gs[1])
#ay.plot(y, x)
#color_code = [ 'blue', 'red', 'gray' ]
color_code = [ 'dodgerblue', 'tab:red', 'tab:gray' ]
shape_code = [ 's', 'o', '^' ]
def display_city(create_legend = True):
    alpha = 0.8
    for social_class in range( social_class_num ):
        #print('so: ', social_class)
        this_class_agents = (agents['social_class'] == social_class)
        #print('so: ', social_class)
        for health_status in range(-1, 2 ):
            this_health_status = (np.sign( agents['health'] ) == health_status)

            this_subgroup = np.all( [ this_class_agents, this_health_status ], axis = 0 )

            this_subgroup_list = np.where(this_subgroup)[0]
            #print('so: ', social_class)
            if (len(this_subgroup_list)):
                nx.draw_networkx_nodes(G,pos = pos, ax = ax, nodelist=list(this_subgroup_list),  node_color = color_code[health_status]\
               , node_shape = shape_code[social_class], node_size = 150, alpha = 1,\
               edgecolors = 'black')
               #edgecolors = color_code[health_status)
                #nx.draw_networkx_nodes(G,pos = pos, ax = ax, nodelist=np.array([1,2]),  node_color = 'r', node_shape = 'o', alpha = alpha, edgecolors = 'r')
    
    #stayers = list( np.where(agents['strategy'] == 1)[0] )
    
    
    #stayers = np.all ([agents['strategy'] == 0, agents['health'] >= -1], axis = 0)
    stayers = list(np.where(np.all ([agents['strategy'] == 0, agents['health'] >= -1], axis = 0)))[0]
    #print( agents[stayers] )
    
    dead_edges = [edge for edge in G.edges() if( agents[edge[0]]['health'] == -2 or  agents[edge[1]]['health'] == -2) ]
    
    visible_edges = [edge for edge in G.edges()\
     if( edge[0] not in stayers  and edge[1] not in stayers and edge not in dead_edges) ]
    
    invisible_edges = [edge for edge in G.edges() if not(edge in visible_edges or edge in dead_edges) ]
    
    nx.draw_networkx_edges(G,pos = pos, ax = ax, alpha = alpha*0.8, width = 1.2, edge_color = 'black', edgelist = visible_edges)
    nx.draw_networkx_edges(G,pos = pos, ax = ax, alpha = alpha/4, width = 1, style = 'dashed' ,edge_color = 'black', edgelist = invisible_edges)
    
#    stayers = list( np.where(agents['strategy'] == 0)[0] )
    stayers = list(np.where(np.all ([agents['strategy'] == 0, agents['health'] >= 0], axis = 0)))
    stayers_pos = pos_array[stayers]
    x = stayers_pos[:, 0]
    y = stayers_pos[:, 1]

    ax.scatter(x, y, s = 800, facecolors='none', linestyle='dashdot' \
       , edgecolors='green', linewidth = 1.8, alpha = 0.8)

    
    
    ax.axis('off')
    
    ax.set_ylim( np.array( list( pos.values() ) ).min()*1.2  - 0.1, np.array( list( pos.values() ) ).max()*1.2 )
    ax.set_xlim( np.array( list( pos.values() ) ).min()*1.2  - 0.1, np.array( list( pos.values() ) ).max()*1.2 )
    #ax.scatter(x, y, s = 2000, facecolors='none', edgecolors='r', linewidth = 2.8)
    #nx.draw_networkx_labels(G,pos , ax = ax, font_size=16)
    if create_legend:
        grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T
        ay.axis('off')
        legend_elements = [
                       Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label='Susceptible',
                              markerfacecolor='dodgerblue', markersize=10, alpha = alpha),
                              
                       Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label='Infectious',
                              markerfacecolor='tab:red', markersize=10, alpha = alpha),
                              
                       Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label='Removed',
                              markerfacecolor='tab:gray', markersize=10, alpha = alpha),

                              
                       Line2D([0], [0], marker='s', color='w', markeredgecolor='black', label='Low SES',
                              markerfacecolor='white', markersize=10, alpha = alpha),
                       Line2D([0], [0], marker='o', color='w', markeredgecolor='black', label='Middle SES',
                              markerfacecolor='white', markersize=10, alpha = alpha),
                       Line2D([0], [0], marker='^', color='w', markeredgecolor='black', label='High SES',
                              markerfacecolor='white', markersize=10, alpha = alpha),
                       Line2D([0], [0], marker='o', color='w', linewidth = 2, markeredgecolor='green', label='Staying Home',
                              markerfacecolor='white', markersize=14, alpha = alpha, linestyle='dashdot'),
                              Line2D([0], [0], color='k', lw=1.2, label='Active Contact', alpha = alpha * 0.8),                              
                              Line2D([0], [0], color='k', lw=1, label='Inactive Contact', linestyle='--', alpha = alpha * 0.5),
                              
                       #Patch(facecolor='white', linewidth = 2, edgecolor='r', linestyle ='solid',
                         #label='Color Patch'),
                       #matplotlib.patches.Circle((10, 10), edgecolor='r'),
                       #mpatches.Wedge(grid[0], 0.1, 30, 270, ec="none")


    ]
        ay.legend( handles=legend_elements, loc = 'center')
        #ay.legend(handles = legend_elements, bbox_to_anchor=(0.5, 0.5))#,
        #   bbox_transform=plt.gcf().transFigure)
    #plt.show()
    #print( 'healthies = ', np.sum(agents['health'] == 0) )
    return fig

def move_the_removed(newly_removed):
    global moving_out_iter
    if len(newly_removed):
        print('iter: ', moving_out_iter)
        if moving_out_iter <= 10:
            moving_out_iter += 1
            for node in newly_removed:
                pos[node] /= 1.2
                #print(node)
                
                
            display_city()
            return fig
        else:
            update_infection(agents)
            moving_out_iter = 0

        


def display_candidates( movers ):
    movers_list = list( movers )
    movers_pos = pos_array[movers_list]
    x = movers_pos[:, 0]
    y = movers_pos[:, 1]

    ax.scatter(x, y, s = 2000, facecolors='none', edgecolors='r', linewidth = 2.8)
    display_city(create_legend = True)

#display_city()
#G = init_graph()
#init_population(0.5, 0.5, 0.5)
#movers = ()

#display_city()

#print(agents['health'] == 0)
prediction = 1
pred = 1
animation_phases = 3
init_agents(agents, N)
moving_out_iter = 0
def animate(t):
    global moving_out_iter
    global pred
    #print('animate says = ',pred)
    if t == 0:
        pred = 1
    ay.clear()
    ax.clear()
    ax.set_xlim( np.array(ax.get_xlim())*1.1 )
    ax.set_ylim( np.array(ax.get_ylim())*1.1 )

    num_string = str( int( t ) )
    #ay.set_title("$t$ ="+ num_string)
    ay.text(0.4 , 0.8 , "$t$ ="+ num_string )
    print(t)

    
    #################moving
    newly_removed = np.where(agents['health'] ==-1)[0]
    move_the_removed( newly_removed )
    #################
    
    

    display_city()
    if t>0:

        infected_num = infect(G, agents, transmit_prob) #nodes infect their neighbors
        newly_recovered = recover(agents, recovery_prob) #nodes get recovered
        update_infection(agents) #actually change the health statuses (necessary for parallel updating) 
        pred = predict_infected_num(agents, pred, learning_rate) #predict the number of upcoming infected agents for the next step
        survivor_num = update_strategy(agents, exp_stay_home_reward, pred * infection_reward, beta) #update strategies (going out and staying in)
    
    #ax.set_xlim([-1,1])
    return fig

#"""
location = "./"
if do_animate:
#    ani = animation.FuncAnimation(fig, animate, save_count = 40)
#
#    dpi = 100
#    file_name = location + str(time.gmtime()[0:5]) + '.GIF'
#    ani.save( file_name ,dpi=dpi, writer = 'imagemagick')
    #"""
    
    ani = animation.FuncAnimation(fig, animate, save_count = 40)
    dpi = 200
    writer = animation.writers['ffmpeg'](fps = 0.8)
    file_name = str(time.gmtime()[0:5]) + '.mp4'
    ani.save( file_name, dpi=dpi, writer = writer)
