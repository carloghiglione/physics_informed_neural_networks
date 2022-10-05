# %% Lines for graphical options


# %matplotlib qt
# %matplotlib inline


# %% Import relevant libraries

import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import random


import matplotlib.pyplot as plt
from   matplotlib           import cm
from   mpl_toolkits.mplot3d import Axes3D


# %% Settings

idx_mu    = 0
idx_theta = 3

compare_with_NN = False

# %% Load training data and NN model

# Training data

path_to_tables = os.path.join(os.path.dirname(os.getcwd()), 'Tables')

data_int = pd.read_csv (os.path.join(path_to_tables, 'tab_int.csv'), names = ('x','y','mu','theta','u'))
data_bc  = pd.read_csv (os.path.join(path_to_tables, 'tab_bc.csv' ), names = ('x','y','mu','theta','u'))

# NN model

NN_model = ns.utils.import_ANN('current_model')

with open('current_minmax.json', 'r') as fp:
    minmax_bounds = json.load(fp)

min_train = minmax_bounds['min_train']
max_train = minmax_bounds['max_train']



# %% Extract relevant data

mus    = data_int   ['mu'].unique()
thetas = data_int['theta'].unique()


print(mus)
print(thetas)


data_plot  = data_int.loc[(data_int['mu'] == mus[idx_mu]) & (data_int['theta'] == thetas[idx_theta])].to_numpy()

u_norm = (data_plot[:,4] - min_train) / (max_train - min_train)

# Plot numerical solution and NN estimate

fig = plt.figure(dpi=300)
ax  = fig.add_subplot(111, projection='3d')

ax.scatter(xs = data_plot[:,0], ys = data_plot[:,1], zs = u_norm, c = u_norm, label = 'Numerical solution: mu: ' + str(mus[idx_mu]) + ' theta: ' + str(thetas[idx_theta]))

if(compare_with_NN):
    ax.scatter(xs = data_plot[:,0], ys = data_plot[:,1], zs = NN_model(data_plot[:,0:4]), c = 'red', label = 'NN solution')

ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()