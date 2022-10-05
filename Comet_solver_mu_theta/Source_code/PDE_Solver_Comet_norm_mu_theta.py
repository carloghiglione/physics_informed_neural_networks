###############################################################################################################
# 
# - mu u_xx - mu u_yy + 10 cos(theta) u_x + 10 sin(theta) u_y = 10 exp(-100 |x - x_0|)   in \Omega = (0, 1)^2
# u(x,y) = 0                                                                             on \partial\Omega
#
###############################################################################################################


# %% Import relevant libraries

import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import pandas as pd
import numpy as np
import os

def plot_in_Matlab(x, u_hf, u_ann, exp_name = 'experiment_plot'):

    if (not os.path.exists(exp_name)):
        os.makedirs(exp_name)

    np.savetxt(os.path.join(exp_name,'nodes.txt'), x)

    np.savetxt(os.path.join(exp_name,'high_fidelity_solution.txt'), u_hf)
    np.savetxt(os.path.join(exp_name,'neural_network_solution.txt'), u_ann)


# %% Options

# Problem setup

domain_W1  = 1
domain_W2  = 1

domain_dim = 2
params_dim = 2

x_0 = np.array([0.5,0.5])


boundary  = lambda x: 0 * x[:,0]
forcing   = lambda x: 10 * np.exp(-100 * np.sqrt( np.power(x[:,0] - x_0[0], 2) + np.power(x[:,1] - x_0[1], 2) ))



# %% Inizialization

# Set seeds for reproducibility

np.random.seed(1)
tf.random.set_seed(1)


# NN model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(domain_dim + params_dim,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])


# %% Set training and test dataset

# Read and manage data from csv files

path_to_tables = os.path.join(os.path.dirname(os.getcwd()), 'Tables')
# path_to_tables_Andrea = os.path.join(os.getcwd(), '/Users/andreaboselli/Coding/Python/NAPDE_Project/PINN_Solver_Comet_experiments')

data_int = pd.read_csv (os.path.join(path_to_tables, 'tab_int.csv'), names = ('x','y','mu','theta','u'))
data_bc  = pd.read_csv (os.path.join(path_to_tables, 'tab_bc.csv'),  names = ('x','y','mu','theta','u'))

# find the params of the training data

mu = data_int['mu'].unique()
n_mu = len(mu)
theta = data_int['theta'].unique()
n_theta = len(theta)

# set training and test set size

n_mu_train = 5
n_mu_test  = n_mu - n_mu_train
n_theta_train = 10
n_theta_test  = n_theta - n_theta_train

# extract training and test data
idx_train_mu = np.int64(np.floor(np.linspace(0, n_mu-1, n_mu_train)))
idx_test_mu  = np.delete(range(n_mu), idx_train_mu)
idx_train_theta = np.int64(np.floor(np.linspace(0, n_theta-1, n_theta_train)))
idx_test_theta  = np.delete(range(n_theta), idx_train_theta)

# number of elems in each group 

n_bc_per_param  = len(data_bc)/(n_mu*n_theta)
n_int_per_param = len(data_int)/(n_mu*n_theta)

train_int = pd.DataFrame()   # train set of internal points
train_bc  = pd.DataFrame()   # train set for bc points
test_int  = pd.DataFrame()   # test set for internal points
test_bc   = pd.DataFrame()   # test set for bc points

for i in range(n_mu):
    if i in idx_train_mu:
        for j in range(n_theta):
            if j in idx_train_theta: 
                train_int = train_int.append( data_int.iloc[ np.arange(i*n_theta*n_int_per_param + j*n_int_per_param, i*n_theta*n_int_per_param + (j+1)*n_int_per_param ) ] )
                train_bc = train_bc.append( data_bc.iloc[ np.arange(i*n_theta*n_bc_per_param + j*n_bc_per_param, i*n_theta*n_bc_per_param + (j+1)*n_bc_per_param ) ] )
            else:
                test_int = test_int.append( data_int.iloc[ np.arange(i*n_theta*n_int_per_param + j*n_int_per_param, i*n_theta*n_int_per_param + (j+1)*n_int_per_param ) ] )
                test_bc = test_bc.append( data_bc.iloc[ np.arange(i*n_theta*n_bc_per_param + j*n_bc_per_param, i*n_theta*n_bc_per_param + (j+1)*n_bc_per_param ) ] )
    else:
        for j in range(n_theta):
            test_int = test_int.append( data_int.iloc[ np.arange(i*n_theta*n_int_per_param + j*n_int_per_param, i*n_theta*n_int_per_param + (j+1)*n_int_per_param ) ] )
            test_bc = test_bc.append( data_bc.iloc[ np.arange(i*n_theta*n_bc_per_param + j*n_bc_per_param, i*n_theta*n_bc_per_param + (j+1)*n_bc_per_param ) ] )

# Generalized domain (x,y, mu, theta)

x_PDE    =  tf.constant(np.array(train_int[['x', 'y']]),        ns.config.get_dtype())
p_PDE    =  tf.constant(np.array(train_int[['mu', 'theta']]),             ns.config.get_dtype())
x_p_PDE  =  tf.constant(np.array(train_int[['x', 'y', 'mu', 'theta']]),   ns.config.get_dtype())

x_p_BC   =  tf.constant(np.array(train_bc [['x', 'y', 'mu', 'theta']]),   ns.config.get_dtype())

x_test   =  tf.constant(np.array(test_int [['x', 'y']]),        ns.config.get_dtype())
p_test   =  tf.constant(np.array(test_int [['mu', 'theta']]),             ns.config.get_dtype())
x_p_test =  tf.constant(np.array(test_int [['x', 'y', 'mu', 'theta']]),   ns.config.get_dtype())

x_p_test_BC = tf.constant(np.array(test_bc[['x', 'y', 'mu', 'theta']]),   ns.config.get_dtype())

# Exact solution in training points, test points, and boundaries

u_HF_PDE  = tf.constant(np.array(train_int[['u']]), ns.config.get_dtype())
u_HF_BC   = tf.constant(np.array(train_bc [['u']]), ns.config.get_dtype())

u_HF_test    = tf.constant(np.array(test_int [['u']]), ns.config.get_dtype())
u_HF_test_BC = tf.constant(np.array(test_bc [['u']]), ns.config.get_dtype())


# %% Normalization

max_train = tf.math.reduce_max(tf.concat([u_HF_PDE, u_HF_BC], 0))
min_train = tf.math.reduce_min(tf.concat([u_HF_PDE, u_HF_BC], 0))

u_HF_PDE_norm = ( u_HF_PDE - min_train )/( max_train - min_train )
u_HF_BC_norm = ( u_HF_BC - min_train )/( max_train - min_train )

u_HF_norm_test = ( u_HF_test - min_train )/( max_train - min_train )
u_HF_BC_norm_test = ( u_HF_test_BC - min_train )/( max_train - min_train )



# %% Define normalized PDE equation

def PDE(x, param):
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x)
        
        # Domain, output and forcing
         
        x_and_p = tf.concat((x,param), axis = 1)
        u = model(x_and_p)
        f = forcing(x_and_p)[:,None]
        
        # Gradient and laplacian
        
        grad_u = nse.physics.tens_style.gradient_scalar(tape, u, x)
        lapl_u = nse.physics.tens_style.laplacian_scalar(tape, u, x, domain_dim)[:,None]
        
        # Convective summand
        
        b = tf.concat((tf.constant(np.cos(param[:,1]))[:,None],
                       tf.constant(np.sin(param[:,1]))[:,None]), axis = 1)
        
        grad_u_per_b = (grad_u[:,0]*b[:,0] + grad_u[:,1]*b[:,1])[:,None]
        
        # Diffusion parameter
        
        mu = param[:,0][:,None]

    # Residual of the equation

    return - mu * (max_train - min_train) * lapl_u + 10 * (max_train - min_train) * grad_u_per_b - f



# %% Losses definition

losses = [ ns.LossMeanSquares('fit', lambda: model(x_p_PDE) - u_HF_PDE_norm, weight = 1),
           ns.LossMeanSquares('PDE', lambda: PDE(x_PDE, p_PDE), weight = 1),
           ns.LossMeanSquares('BC',  lambda: model(x_p_BC) - u_HF_BC_norm, weight = 1)
         ]

loss_test = [ ns.LossMeanSquares('fit', lambda: model(x_p_test) -  u_HF_norm_test),
              ns.LossMeanSquares('PDE', lambda: PDE(x_test, p_test)),
              ns.LossMeanSquares('BC', lambda: model(x_p_test_BC) - u_HF_BC_norm_test)
            ]


# %% Training

pltcb = ns.utils.HistoryPlotCallback()
pb = ns.OptimizationProblem(model.variables, losses, loss_test)


ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-3), num_epochs = 250)
ns.minimize(pb, 'scipy', 'BFGS', num_epochs = 10000, options={'gtol': 1e-100})

pltcb.finalize(pb, block = False)

#model.save("many_data_model")
#reconstructed_model = tf.keras.models.load_model("many_data_model")

# %% Post-processing

#  %matplotlib qt
#  %matplotlib inline

# x_p_final    = tf.constant(np.array(test_int.reset_index().loc[range(int(n_int_per_param))][['x','y','mu','theta']]), ns.config.get_dtype())
# u_final      = tf.constant(np.array(test_int.reset_index().loc[range(int(n_int_per_param))][['u']]),                  ns.config.get_dtype())
# u_final_norm = ( u_final - min_train )/( max_train - min_train )

# plot_in_Matlab(x_p_final[:,0:2],
#                u_final_norm,
#                model(x_p_final),
#                'PINN_solver_comet_mu_theta_experiment')