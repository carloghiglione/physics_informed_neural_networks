[errors,solutions,femregion,Dati] = C_main2D('Test1',6, pi/4, 1)

grid = femregion.coord;
n_tot = length(grid);
bc_points = grid(femregion.boundary_points,:);
grid(femregion.boundary_points,:) = [];

uh = full(solutions.uh);

uh_bc = uh(femregion.boundary_points,:);
uh(femregion.boundary_points,:) = [];
n_int = length(uh);
n_bc = length(uh_bc);

param = 1;

tab_int = [grid, param*ones(n_int,1), uh];

tab_bc = [bc_points, param*ones(n_bc,1), uh_bc];


