##############################################################################################################
#
# - mu u_xx - mu u_yy + 10 cos(theta) u_x + 10 sin(theta) u_y = 10 exp(-100 |x - x_0|)   in \Omega = (0, 1)^2
# u(x,y) = 0                                                                             on \partial\Omega
#
##############################################################################################################



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

debug        = False
save_model   = True

domain_dim = 2
params_dim = 2

x_0 = np.array([0.5,0.5])

boundary  = lambda x: 0 * x[:,0]
forcing   = lambda x: 10 * np.exp(-100 * np.sqrt( np.power(x[:,0] - x_0[0], 2) + np.power(x[:,1] - x_0[1], 2) ))


K = 10            # Number of bases functions we decide to keep
train_split = 0.9 # Tentative proportion of training points wrt validation points

epochs_keras =  1000
epochs_scipy = 40000



# %% ##########################################################################################################

### PART 1: Construction of mean function u_0 and pertubation functions \phi_i



# %% Load training data

path_to_tables = os.path.join(os.path.dirname(os.getcwd()), 'tables_mu_theta')

data_int = pd.read_csv (os.path.join(path_to_tables, 'tab_int.csv'), names = ('x','y','mu','theta','u'))
data_bc  = pd.read_csv (os.path.join(path_to_tables, 'tab_bc.csv' ), names = ('x','y','mu','theta','u'))


data_all = pd.concat([data_int,data_bc])


# %% Extract point-param-solution matrix

# Values of mu and theta
mus    = data_all   ['mu'].unique()
thetas = data_all['theta'].unique()


# Physical points

x_all  = data_all.loc[(data_all['mu'] == mus[0]) & (data_all['theta'] == thetas[0])][['x','y']].to_numpy()
x_int  = data_int.loc[(data_int['mu'] == mus[0]) & (data_int['theta'] == thetas[0])][['x','y']].to_numpy()
x_bc   = data_bc.loc [(data_bc ['mu'] == mus[0]) & (data_bc ['theta'] == thetas[0])][['x','y']].to_numpy()

# Dimension of the Matrix
n = x_all.shape[0]
p = len(mus) * len(thetas)



if(debug == True): # Check that the concatenations are correct

    data_all_1  = data_all.loc[(data_all['mu'] == mus[0]) & (data_all['theta'] == thetas[0])].to_numpy()
    data_all_2  = data_all.loc[(data_all['mu'] == mus[1]) & (data_all['theta'] == thetas[1])].to_numpy()
    
    print(np.all(data_all_1[:,0] == data_all_2[:,0]))       # x coincide
    print(np.all(data_all_1[:,1] == data_all_2[:,1]), '\n') # y coincide
    
    u_all_1 = data_all.loc[(data_all['mu'] == mus[0]) & (data_all['theta'] == thetas[0])][['u']].to_numpy()
    u_all_2 = data_all.loc[(data_all['mu'] == mus[1]) & (data_all['theta'] == thetas[1])][['u']].to_numpy()
    
    u_all_12 = np.concatenate((u_all_1,u_all_2), axis = 1)
    p_all_12 = np.concatenate((np.array(([[mus[0],thetas[0]]])),np.array(([[mus[1],thetas[1]]]))), axis = 0)



# Reshaped dataset and parameters
data_all_reshaped = data_all.loc[(data_all['mu'] == mus[0]) & (data_all['theta'] == thetas[0])][['u']].to_numpy()

p_all = np.array(([[mus[0],thetas[0]]]))

for mu in mus:
    for theta in thetas:
        if ((mu != mus[0]) | (theta != thetas[0])):
            
            p_current = np.array(([[mu,theta]]))
            u_current = data_all.loc[(data_all['mu'] == mu) & (data_all['theta'] == theta)][['u']].to_numpy()
            
            p_all = np.concatenate((p_all, p_current), axis = 0)
            data_all_reshaped = np.concatenate((data_all_reshaped, u_current), axis = 1)



if(debug == True): # Check that the rows of data_all_reshaped have the values we expect
    
    print(p_all.shape             == (p,2))
    print(data_all_reshaped.shape == (n,p), '\n')

    for j in range(p):

        print(np.all(data_all.loc[(data_all['mu'] == p_all[j,0]) & (data_all['theta'] == p_all[j,1])]['u'].to_numpy() == data_all_reshaped[:,j]))

    print('\n')



# x_int
# p_all
# data_int_reshaped



S = data_all_reshaped.T

S_means = np.mean(S, axis=0)

U = S

for j in range(n):
    
    U[:,j] = U[:,j] - np.ones(p)*S_means[j]



if(debug == True): # Check that the columns all have 0 mean
    
    print(np.all(np.abs(np.mean(U, axis=0)) < 1e-15), '\n')
    


# %% Compute the covariance matrix C, its eigenvalue and eigenvectors; then extract the K most relevant

# Covariance matrix C
C = (1/(p-1)) * np.matmul(U.T, U)


# Eigenvalues, eigenvectors
e_values, e_vectors = np.linalg.eig(C)


# Eliminate complex eigenvalues and their eigenvectors
real_eigenvalues = (np.imag(e_values) == 0)

e_values_real  = np.real(e_values   [real_eigenvalues])
e_vectors_real =         e_vectors[:,real_eigenvalues]



if(debug): # Check that the remaining eigenvectors are real
    
    print(np.all(np.imag(e_vectors_real) == 0),'\n')



e_vectors_real =  np.real(e_vectors_real)



# Sort eigenvalues and eigenvectors
idx = np.argsort(e_values_real)


e_values_sorted  = e_values_real   [idx]
e_vectors_sorted = e_vectors_real[:,idx]


# Select the K most relevant

relevant_range = range(len(e_values_sorted)-K,len(e_values_sorted))

e_values_final  = e_values_sorted   [relevant_range]
e_vectors_final = e_vectors_sorted[:,relevant_range]

proxy_prop_variability = np.sum(e_values_final) / np.sum(e_values_sorted)



if(debug):
    
    print(C.shape                 == (n,n))
    print(e_vectors_real.shape[1] == e_values_real.shape[0])
    print(e_vectors_sorted.shape  == e_vectors_real.shape)
    print(e_vectors_final.shape   == (n,K))
    print('\n')



# %% Plot cumulative sum of eigenvalues
    
n_values        = np.ones((20,))
cum_variability = np.ones((20,))

for i in range(1,21):
    
    n_values[i-1] = i
    cum_variability[i-1] = np.sum(e_values_sorted[range(len(e_values_sorted)-i,len(e_values_sorted))]) / np.sum(e_values_sorted)
    
fig5 = plt.figure(dpi=300)

plt.scatter(x = n_values, 
            y = cum_variability)

plt.xticks(ticks = n_values, labels = n_values.astype(int))

# %% Plot the data mean, and some of the resulting eigenvectors (plus the mean)

rows = 2
cols = 5

fig1 = plt.figure(dpi=300,figsize=(7,3))

for i in range(cols*rows):
    
    fig1.add_subplot(rows, cols, i+1)
    
    plt.scatter(x = x_all[:,0], 
                y = x_all[:,1], 
                c = e_vectors_final[:,K-1-i],
                s = 1)
    
    plt.plasma()
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Eigenvalue ' + str(i+1))

    plt.axis('off')



fig2 = plt.figure(dpi=300,figsize=(3,3))

plt.scatter(x = x_all[:,0], 
            y = x_all[:,1], 
            c = S_means,
            s = 20)

plt.plasma()
    
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data mean')



# %% ##########################################################################################################

### PART 2: Training of a NN able to approximate u_0 and \phi_i



# %% Initialization

# Set seeds for reproducibility

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# NN model

model = tf.keras.Sequential([
                                tf.keras.layers.Dense(26, input_shape=(2,), activation=tf.nn.tanh),
                                tf.keras.layers.Dense(26, activation=tf.nn.tanh),
                                tf.keras.layers.Dense(K+1)
                            ])



# %% Set dataset

# Overall dataset

x_all_tf      = tf.constant(x_all)                                           # Input
phi_i_and_u_0 = np.concatenate((S_means[:,None], e_vectors_final), axis = 1) # Output

n_int = x_int.shape[0]
n_bc  = x_bc.shape [0]



if(debug):
    
    # x_all is the real concatenation of x_int and x_bc
    print(n_int + n_bc == n)
    print(np.all(x_int == x_all[0:n_int,:]))
    print(np.all(x_bc  == x_all[n_int:n,:]),'\n')

    # phi_i_and_u_0 is actually what we think
    print(np.all(phi_i_and_u_0[:,0]        == S_means))
    print(np.all(phi_i_and_u_0[:,1:(K+1)]  == e_vectors_final), '\n')



# Normalize vector of the means
S_means_norm = np.linalg.norm(S_means)

phi_i_and_u_0[:,0] = phi_i_and_u_0[:,0] / S_means_norm

phi_i_and_u_0 = tf.constant(phi_i_and_u_0)


# Split in internal and boundary points
x_int_tf = x_all_tf[0:n_int,  :]
x_bc_tf  = x_all_tf[n_int:n,  :]

phi_i_and_u_0_int = phi_i_and_u_0[0:n_int,:]
phi_i_and_u_0_bc  = phi_i_and_u_0[n_int:n,:]


# Split internal points in training and validation data

n_int_train = int(np.floor(train_split * n_int))

train_split_int_true = n_int_train/n_int

idx_int = [i for i in range(n_int)]

idx_int_train = np.sort(np.array(random.sample(idx_int, n_int_train)))
idx_int_test  = np.sort(np.setdiff1d(idx_int, idx_int_train))


x_int_train             = tf.gather(x_int_tf,          idx_int_train)
x_int_test              = tf.gather(x_int_tf,          idx_int_test)

phi_i_and_u_0_int_train = tf.gather(phi_i_and_u_0_int, idx_int_train)
phi_i_and_u_0_int_test  = tf.gather(phi_i_and_u_0_int, idx_int_test)


# Split boundary points in training and validation data

n_bc_train = int(np.floor(train_split * n_bc))

train_split_bc_true = n_bc_train/n_bc

idx_bc = [i for i in range(n_bc)]

idx_bc_train = np.sort(np.array(random.sample(idx_bc, n_bc_train)))
idx_bc_test  = np.sort(np.setdiff1d(idx_bc, idx_bc_train))


x_bc_train             = tf.gather(x_bc_tf,          idx_bc_train)
x_bc_test              = tf.gather(x_bc_tf,          idx_bc_test)

phi_i_and_u_0_bc_train = tf.gather(phi_i_and_u_0_bc, idx_bc_train)
phi_i_and_u_0_bc_test  = tf.gather(phi_i_and_u_0_bc, idx_bc_test)



if(debug):
    
    fig6 = plt.figure(dpi=300,figsize=(3,3))
    
    plt.scatter(x = x_bc_train[:,0], y = x_bc_train[:,1], c = 'black',                       s = 10)
    plt.scatter(x = x_bc_test [:,0], y = x_bc_test [:,1], c = 'white', edgecolors = 'black', s = 10)
    
    plt.scatter(x = x_int_train[:,0], y = x_int_train[:,1], c = 'black',                       s = 10)
    plt.scatter(x = x_int_test [:,0], y = x_int_test [:,1], c = 'white', edgecolors = 'black', s = 10)
    


# %% Losses definition


losses      = [ ns.LossMeanSquares('fit', lambda: model(x_int_train) - phi_i_and_u_0_int_train, weight = 1),
                ns.LossMeanSquares('bc' , lambda: model(x_bc_train)  - phi_i_and_u_0_bc_train , weight = 1) ]


losses_test = [ ns.LossMeanSquares('fit', lambda: model(x_int_test)  - phi_i_and_u_0_int_test),
                ns.LossMeanSquares('bc' , lambda: model(x_bc_test)   - phi_i_and_u_0_bc_test ) ]



# %% Training

pltcb = ns.utils.HistoryPlotCallback()
pb = ns.OptimizationProblem(model.variables, losses, losses_test)



if(debug):
    
    epochs_keras = 10
    epochs_scipy = 10
    


ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-3), num_epochs = epochs_keras)
ns.minimize(pb, 'scipy', 'BFGS',                                       num_epochs = epochs_scipy, options={'gtol': 1e-100})


pltcb.finalize(pb, block = False)


if(save_model):
    
    # Create data folder
    data_folder = os.path.join(os.getcwd(), 'POD_1_data')
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Save model
    ns.utils.export_ANN(model, os.path.join(data_folder,'model_phi_i_and_u_0'))
    
    
    # Save relevant data
    relevant_data = {}
    
    relevant_data['S_means_norm'] = float(S_means_norm)
    
    with open(os.path.join(data_folder,'relevant_data.json'), 'w') as fp:
        json.dump(relevant_data, fp)
        


# %% Plot and compare the results

fig4_idx = 3

fig3 = plt.figure(dpi=300)
ax3  = fig3.add_subplot(111, projection='3d')

ax3.scatter(xs = x_all[:,0], ys = x_all[:,1], zs = phi_i_and_u_0[:,0], label = 'Real function of the means')
ax3.scatter(xs = x_all[:,0], ys = x_all[:,1], zs = model(x_all) [:,0], label = 'NN function of the means')

ax3.legend()
plt.show()


fig4 = plt.figure(dpi=300)
ax4  = fig4.add_subplot(111, projection='3d')


ax4.scatter(xs = x_all[:,0], ys = x_all[:,1], zs = phi_i_and_u_0[:,-fig4_idx], label = 'Real eigenvector ' + str(fig4_idx))
ax4.scatter(xs = x_all[:,0], ys = x_all[:,1], zs = model(x_all) [:,-fig4_idx], label = 'NN eigenvector '   + str(fig4_idx))

ax4.legend()
plt.show()