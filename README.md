# Neural Networks-Based Reduced Order Modeling of PDEs

#### Numerical Analysis for Partial Differential Equations, Course Project 2020-2021

#### Andrea Boselli, Carlo Ghiglione, Leonardo Perelli
#### Tutors: Stefano Pagani, Francesco Regazzoni

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## Overview

It is a well know result that Neural Networks can asymptotically approximate any continous real function.
Following a <a href="https://arxiv.org/abs/1910.03193">paper</a> authored by George Em Karniadakis et al., we implement and test the ability of deep Neural Networks to approximate parametric maps underlying the solution of PDEs. In particular, given a PDE depending on some parameters, we implement a system being able to accurately approximate the solution of the equation.

Provided that high fidelity numerical solutions of a given problem are available, and that selection of parameters and hyperparameters and training of the architecture are properly performed (off-line phase), the Neural Network is able to provide the approximations of the solutions of the problem in an extremely fast way (on-line phase), compared to numerical solvers based on Finite Element Methods. This finds application in all the areas that require to quickly compute solutions of problems that depend on unknown values of parameters.

With this work, we explore the possibility of implementing such architectures, studying their characteristics and limitations. To do so, we first focus on a case study problem, a linear Advection-Diffusion equation that features as parameters the diffusion and transport coefficient.

Next, we focus on the Monodomain equation, contributing to model cardiac electrical activity, under different configurations of the PDE domain. We cope with this variability parametrising the domain as a function of some scalar variables, and considering such variables as parameters of the problem.

For the practical implementation, we work on Python, in particular using Nisaba, a machine learning library built on top of Tensorflow.
