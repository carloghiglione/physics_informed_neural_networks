# %% Import relevant libraries

# %% Inizialization
import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Set seeds for reproducibility

np.random.seed(1)
tf.random.set_seed(1)

# Problem setup:
param_geo_dim = 4
input_dim = param_geo_dim + 3


# NN model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(input_dim,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])



# %% Load data

# Read and manage data from csv files
cwd = os.getcwd()
path_to_table = os.path.join(cwd, 'tabel.csv')

data = pd.read_csv (path_to_table, names = ('x','y','z','geo1','geo2','geo3','geo4','AT'))
AT = data['AT']
AT_norm = (AT - AT.min())/(AT.max() - AT.min())
data = data[['x','y','z','geo1','geo2','geo3','geo4']]
# find the params of the training data
# Nel caso di mesh con curvatura uniforme
geo_params = data['geo1'].unique()
n_params = len(geo_params)
n_nodes = len(data[data["geo1"] == geo_params[0]])


# %% Undersample the data
n_under = 1000                 #number of nodes used for each mesh
data_under = pd.DataFrame()
AT_under = np.array([])

for i in range(n_params):
    idx_under = np.random.choice(range(n_nodes), n_under, replace = False)
    data_under = data_under.append( data[data["geo1"] == geo_params[i]].iloc[idx_under] )
    AT_under = np.append(AT_under, AT_norm[n_nodes*i + idx_under])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data_under['x'].iloc[0:n_under],
           data_under['y'].iloc[0:n_under],
           data_under['z'].iloc[0:n_under],
           color = "green")
plt.show()

# %% Set Training and Test Dataset
# set training and test set size with respect to the mesh nodes

perc_train = 0.8
n_train = int(perc_train*n_params)
n_test = n_params-n_train

# extract training and test data

idx_train = np.sort(np.random.choice(range(n_params), n_train , replace=False))
idx_test_1  = np.delete(range(n_params), idx_train)


train = pd.DataFrame()   # training set
test_1  = pd.DataFrame()   # test set
test_2  = pd.DataFrame()   # test set
AT_train = np.array([])
AT_test_1 = np.array([])
AT_test_2 = np.array([])


#training set
for i in idx_train:
    idx = data_under["geo1"] == geo_params[i]
    train = train.append( data_under[idx]  )
    AT_train = np.append(AT_train, AT_under[idx])

#test set 1 (same nodes of train, different params)
for i in idx_test_1:
    idx = data_under["geo1"] == geo_params[i]
    test_1 = test_1.append( data_under[idx]  )
    AT_test_1 = np.append(AT_test_1, AT_under[idx])

#test set 2 (new nodes of train, all params)
for i in range(n_params):
    idx = np.random.choice(range(n_nodes), int(n_under/2), replace = False)
    test_2 = test_2.append( data[data["geo1"] == geo_params[i]].iloc[idx] )
    AT_test_2 = np.append(AT_test_2, AT_norm[n_nodes*i + idx])

# %% Losses definition

train_tensor = tf.constant(train.to_numpy(), ns.config.get_dtype())
test_1_tensor = tf.constant(test_1.to_numpy(), ns.config.get_dtype())
test_2_tensor = tf.constant(test_2.to_numpy(), ns.config.get_dtype())
AT_train_tensor = tf.constant(AT_train, ns.config.get_dtype())
AT_train_tensor = tf.expand_dims(AT_train_tensor, 1)
AT_test_1_tensor = tf.constant(AT_test_1, ns.config.get_dtype())
AT_test_1_tensor = tf.expand_dims(AT_test_1_tensor, 1)
AT_test_2_tensor = tf.constant(AT_test_2, ns.config.get_dtype())
AT_test_2_tensor = tf.expand_dims(AT_test_2_tensor, 1)

losses = [ ns.LossMeanSquares('fit', lambda: model(train_tensor) - AT_train_tensor)]

loss_test = [ ns.LossMeanSquares('fit_1', lambda: model(test_1_tensor) -  AT_test_1_tensor),
              ns.LossMeanSquares('fit_2', lambda: model(test_2_tensor) -  AT_test_2_tensor)
            ]


# %% Training

pltcb = ns.utils.HistoryPlotCallback()
pb = ns.OptimizationProblem(model.variables, losses, loss_test)


ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-3), num_epochs = 5)
ns.minimize(pb, 'scipy', 'BFGS', num_epochs = 5, options={'gtol': 1e-100})

pltcb.finalize(pb, block = False)


# %% Max AT plot

out_train = model(train_tensor).numpy()
out_test_1 = model(test_1_tensor).numpy()
out_test_2 = model(test_2_tensor).numpy()

AT_max_test_1 = np.array([])
AT_max_test_2 = np.array([])
AT_max_train = np.array([])

for i in idx_train:
    idx = np.where(train["geo1"] == geo_params[i])
    AT_max_train = np.append(AT_max_train, np.max(out_train[idx]))

for i in idx_test_1:
    idx = np.where(test_1["geo1"] == geo_params[i])
    AT_max_test_1 = np.append(AT_max_test_1, np.max(out_test_1[idx]))

for i in range(n_params):
    idx = np.where(test_2["geo1"] == geo_params[i])
    AT_max_test_2 = np.append(AT_max_test_2, np.max(out_test_2[idx]))

AT_max_real = np.array([])
for i in range(n_params):
    idx = np.where(data["geo1"] == geo_params[i])
    AT_max_real = np.append(AT_max_real, np.max(AT.iloc[idx]))

#remove normalization
AT_max_train = AT_max_train*(AT.max() - AT.min()) + AT.min()
AT_max_test_1 = AT_max_test_1*(AT.max() - AT.min()) + AT.min()
AT_max_test_2 = AT_max_test_2*(AT.max() - AT.min()) + AT.min()


fig2 = plt.figure()
plt.plot(geo_params, 0.1*AT_max_real, 'k--s', markersize=5, label="experimental")
plt.plot(geo_params[idx_train], 0.1*AT_max_train, 'b--s', markersize=5, label="train")
plt.plot(geo_params, 0.1*AT_max_test_2, 'gx', markersize=10, markeredgewidth=3, label="test 2")
plt.plot(geo_params[idx_test_1], 0.1*AT_max_test_1, 'rx', markersize=10, markeredgewidth=3, label="test 1")
plt.legend(loc="upper left", fontsize=25)
plt.title("Max Activation Time vs Deformation Parameter", fontdict={'fontsize': 25})
plt.xlabel("Deformation Parameter", fontdict={'fontsize': 20})
plt.ylabel("Max Activation Time [ms]", fontdict={'fontsize': 20})
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
