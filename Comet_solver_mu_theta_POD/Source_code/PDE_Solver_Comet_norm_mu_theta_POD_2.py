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


train_split = 0.9 # Tentative proportion of training points wrt validation points

epochs_keras =  500
epochs_scipy = 3000



# %% ##########################################################################################################

### PART 3: Find the modal values of the phi_i functions wrt training numerical functions u, and a bias



# %% Load relevant material

# Load training data
path_to_tables = os.path.join(os.path.dirname(os.getcwd()), 'tables_mu_theta')

data_int = pd.read_csv (os.path.join(path_to_tables, 'tab_int.csv'), names = ('x','y','mu','theta','u'))
data_bc  = pd.read_csv (os.path.join(path_to_tables, 'tab_bc.csv' ), names = ('x','y','mu','theta','u'))


# Load model for basis functions
model_phi_i_and_u_0 = ns.utils.import_ANN(os.path.join('POD_1_data','model_phi_i_and_u_0'))


# Load relevant data
with open(os.path.join('POD_1_data','relevant_data.json'), 'r') as fp:
    relevant_data = json.load(fp)
    

# %% Extract point-param-solution matrix

# Values of mu and theta
mus    = data_int   ['mu'].unique()
thetas = data_int['theta'].unique()


# Internal and boundary points
x_int  = data_int.loc[(data_int['mu'] == mus[0]) & (data_int['theta'] == thetas[0])][['x','y']].to_numpy()
x_bc   = data_bc.loc [(data_bc ['mu'] == mus[0]) & (data_bc ['theta'] == thetas[0])][['x','y']].to_numpy()


# Dimension of the matrices
n_int = x_int.shape[0]
n_bc  = x_bc.shape [0]
p     = len(mus) * len(thetas)


# Dataset for parameters internal points and boundary points
data_int_reshaped = data_int.loc[(data_int['mu'] == mus[0]) & (data_int['theta'] == thetas[0])][['u']].to_numpy()
data_bc_reshaped  = data_bc.loc [(data_bc ['mu'] == mus[0]) & (data_bc ['theta'] == thetas[0])][['u']].to_numpy()

p_all = np.array(([[mus[0],thetas[0]]]))


for mu in mus:
    for theta in thetas:
        if ((mu != mus[0]) | (theta != thetas[0])):
            
            u_int_current = data_int.loc[(data_int['mu'] == mu) & (data_int['theta'] == theta)][['u']].to_numpy()
            u_bc_current  = data_bc.loc [(data_bc ['mu'] == mu) & (data_bc ['theta'] == theta)][['u']].to_numpy()
            p_current = np.array(([[mu,theta]]))
            
            
            p_all = np.concatenate((p_all, p_current), axis = 0)
            data_int_reshaped = np.concatenate((data_int_reshaped, u_int_current), axis = 1)
            data_bc_reshaped  = np.concatenate((data_bc_reshaped,  u_bc_current ), axis = 1)



if(debug == True): # Check that the rows of data_int_reshaped and of data_bc_reshaped have the values we expect
    
    print(p_all.shape             == (p,2))
    print(data_int_reshaped.shape == (n_int,p))
    print(data_bc_reshaped.shape  == (n_bc ,p), '\n')

    for j in range(p):

        print(np.all(data_int.loc[(data_int['mu'] == p_all[j,0]) & (data_int['theta'] == p_all[j,1])]['u'].to_numpy() == data_int_reshaped[:,j]))
        print(np.all(data_bc.loc [(data_bc ['mu'] == p_all[j,0]) & (data_bc ['theta'] == p_all[j,1])]['u'].to_numpy() == data_bc_reshaped [:,j]))

    print('\n')



u_int = data_int_reshaped.T
u_bc  = data_bc_reshaped.T



# %% Initialization

# Set seeds for reproducibility

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
        
# NN model for the modes

K = model_phi_i_and_u_0(tf.constant([[0,0]])).shape[1]-1

model_coeffs = tf.keras.Sequential([
                                    tf.keras.layers.Dense(24, input_shape=(params_dim,), activation=tf.nn.tanh),
                                    tf.keras.layers.Dense(24, activation=tf.nn.tanh),
                                    tf.keras.layers.Dense(K+1)
                                    ])



# %% Split in training and validation data

n_p_train = int(train_split*p)

train_split_true = n_p_train / p

idx = [i for i in range(p)]


idx_train = np.sort(np.array(random.sample(idx, n_p_train)))
idx_test  = np.sort(np.setdiff1d(idx, idx_train))


p_train     =         tf.gather(p_all, idx_train)
p_test      =         tf.gather(p_all, idx_test)


u_int_train =         tf.gather(u_int, idx_train)
u_int_test  =         tf.gather(u_int, idx_test)


u_bc_train  = tf.cast(tf.gather(u_bc,  idx_train), ns.config.get_dtype())
u_bc_test   = tf.cast(tf.gather(u_bc,  idx_test ), ns.config.get_dtype())



if(debug):
    
    fig9 = plt.figure(dpi=300,figsize=(3,6))
    
    plt.scatter(x = p_train[:,0], y = p_train[:,1], c = 'black',                       s = 10)
    plt.scatter(x = p_test [:,0], y = p_test [:,1], c = 'white', edgecolors = 'black', s = 10)
    

    plt.xlabel('mu')
    plt.ylabel('theta')
    plt.title('Training and test parameters')



# %% Losses definition


def u_reconstructed(x, par):

    # Linear combination of the modes
    linar_comb_phi_i = tf.linalg.matmul(model_coeffs(par)[:,1:(K+1)], model_phi_i_and_u_0(x)[:,1:(K+1)], transpose_b = True)

    # Function of the means
    u_0 = tf.tile(tf.transpose(tf.expand_dims(model_phi_i_and_u_0(x)[:,0], axis=1)), [par.shape[0], 1]) * relevant_data['S_means_norm']

    # Bias correction (depending only on the parameters)
    bias = tf.tile(tf.expand_dims(model_coeffs(par)[:,0],axis = 1),[1,x.shape[0]])

    # u reconstructed
    return linar_comb_phi_i + u_0 + bias



def PDE(param):
    
    # Extract inputs and param matrices
    
    param_df = pd.DataFrame(param.numpy(), columns = ('mu','theta'))
    data_df  = data_int[['x','y','mu','theta']].merge(param_df, on = ('mu','theta'), how = 'inner')
    
    x_tf = tf.constant(data_df[['x' ,'y'    ]])
    p_tf = tf.constant(data_df[['mu','theta']])
    
    # Compute force
    
    f = tf.constant(forcing(x_tf)[:,None])
    
    # Compute u, gradient and laplacian
    
    with ns.GradientTape(persistent = True) as tape:
    
        # Watch the input tensor
        
        tape.watch(x_tf)
    
        # Compute the outputs of the 2 NNs
        
        output_NN_1 = model_phi_i_and_u_0(x_tf)
        output_NN_2 = model_coeffs(p_tf)
        
        # Compute the summands
        
        u_0  = output_NN_1[:,0] * relevant_data['S_means_norm']
        bias = output_NN_2[:,0]
        
        sum_phi_i_u_i = tf.math.reduce_sum( tf.math.multiply(output_NN_1[:,1:(K+1)], output_NN_2[:,1:(K+1)]), axis = 1)
        
        # Sum to find u
        
        u = (u_0 + bias + sum_phi_i_u_i)[:,None]
        
        # Gradient and laplacian
        
        grad_u = nse.physics.tens_style.gradient_scalar (tape, u, x_tf)
        lapl_u = nse.physics.tens_style.laplacian_scalar(tape, u, x_tf, domain_dim)[:,None]
        
        
    # Multiply gradient by b
        
    grad_u_per_b = ( grad_u[:,0] * tf.math.cos(p_tf[:,1]) + grad_u[:,1] * tf.math.sin(p_tf[:,1]) )[:,None]
        
    # mu
        
    param_mu = p_tf[:,0][:,None]
    
    # Return residual
        
    return - param_mu * lapl_u + 10 * grad_u_per_b - f
    
    
    
losses      = [ ns.LossMeanSquares('fit', lambda: u_reconstructed(x_int, p_train) - u_int_train, weight = 1),
                ns.LossMeanSquares('bc' , lambda: u_reconstructed(x_bc,  p_train) - u_bc_train , weight = 1),
                ns.LossMeanSquares('PDE', lambda: PDE(p_train)                                 , weight = 1) 
              ]

losses_test = [ ns.LossMeanSquares('fit', lambda: u_reconstructed(x_int, p_test ) - u_int_test ),
                ns.LossMeanSquares('bc' , lambda: u_reconstructed(x_bc,  p_test ) - u_bc_test  ),
                ns.LossMeanSquares('PDE', lambda: PDE(p_test)                                  ) 
              ]



# %% Training

pltcb = ns.utils.HistoryPlotCallback()
pb = ns.OptimizationProblem(model_coeffs.variables, losses, losses_test)



if(debug):
    
    epochs_keras = 10
    epochs_scipy = 10



ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-3), num_epochs = epochs_keras)
ns.minimize(pb, 'scipy', 'BFGS',                                       num_epochs = epochs_scipy, options={'gtol': 1e-100})



if(save_model):
    
    # Create data folder
    data_folder = os.path.join(os.getcwd(), 'POD_2_data')
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Save model
    ns.utils.export_ANN(model_coeffs, os.path.join(data_folder,'model_coeffs'))



pltcb.finalize(pb, block = False)


# %% Plot the results


# %matplotlib qt
# %matplotlib inline


fig7_idx = 10

fig7 = plt.figure(dpi=300)

ax7  = fig7.add_subplot(111, projection='3d')

ax7.scatter(xs = x_int[:,0], ys = x_int[:,1], zs = u_int_test                    [fig7_idx,:], c = 'blue', label = 'True solution')
ax7.scatter(xs = x_bc [:,0], ys = x_bc [:,1], zs = u_bc_test                     [fig7_idx,:], c = 'blue')

ax7.scatter(xs = x_int[:,0], ys = x_int[:,1], zs = u_reconstructed(x_int, p_test)[fig7_idx,:], c = 'red' , label = 'NN solution')
ax7.scatter(xs = x_bc [:,0], ys = x_bc [:,1], zs = u_reconstructed(x_bc,  p_test)[fig7_idx,:], c = 'red' ) 

ax7.set_xlabel('x')
ax7.set_ylabel('y')
ax7.set_zlabel('u');

ax7.legend()
plt.show()



# %% Plot the coefficients as a function of the parameters

fig8_idx = -3

MU, THETA = np.meshgrid(mus, thetas)
COEFF = np.ones(MU.shape)

for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        
        COEFF[i,j] = model_coeffs(tf.constant([[ MU[i,j], THETA[i,j] ]]))[:,fig8_idx]
        
fig8 = plt.figure(dpi=300)

ax8 = plt.axes(projection='3d')
ax8.plot_surface(MU, THETA, COEFF, cmap='jet')

ax8.set_xlabel('mu')
ax8.set_ylabel('theta')
ax8.set_zlabel('c(mu,theta)');