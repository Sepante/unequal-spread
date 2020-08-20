import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


FullLoc = 'Results/'
#file_name = FullLoc + 'Prej-Abs.csv'
file_name = FullLoc + "Prej-Abs-v2-rewards=[-0.6 -1.9 -1.  -1.6]-infect_rew=-recov =0.25-78083249.csv"


print(file_name)

with open(file_name) as f:
    InfResults = pd.read_csv(f, index_col = 0)
InfResults['transmit_prob'] = InfResults['transmit_prob'].round(2)
grouped = InfResults.loc[InfResults['transmit_prob'] == 0.9].groupby(['segregation']).mean()
grouped.plot(style = '--o')

#InfResults.plot(x='transmit_prob', kind = 'line', style = 'o', alpha = 0.1)
#pl.plot()