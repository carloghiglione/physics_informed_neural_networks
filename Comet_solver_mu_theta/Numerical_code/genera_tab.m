clear
clc

n_params_theta = 20;
n_params_mu = 11;
params_theta = linspace(0, 2*pi, n_params_theta);
params_mu = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5];

tab_int_full = [];
tab_bc_full = [];

for j = 1:n_params_mu
    param_mu = params_mu(j);
    
for i = 1:n_params_theta
    param_theta = params_theta(i);
    
    [errors,solutions,femregion,Dati] = C_main2D('Test1', 5, param_theta, param_mu);

    grid = femregion.coord;
    n_tot = length(grid);
    bc_points = grid(femregion.boundary_points,:);
    grid(femregion.boundary_points,:) = [];

    uh = full(solutions.uh);

    uh_bc = uh(femregion.boundary_points,:);
    uh(femregion.boundary_points,:) = [];
    n_int = length(uh);
    n_bc = length(uh_bc);
    
    tab_int = [grid, param_mu*ones(n_int,1), param_theta*ones(n_int,1), uh];
    tab_bc = [bc_points, param_mu*ones(n_bc,1), param_theta*ones(n_bc,1), uh_bc];
    
    tab_int_full = [tab_int_full; tab_int];
    tab_bc_full = [tab_bc_full; tab_bc];
      
end
end

csvwrite("tab_int.csv", tab_int_full);
csvwrite("tab_bc.csv", tab_bc_full);