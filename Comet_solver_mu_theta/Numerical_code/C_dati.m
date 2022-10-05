%=======================================================================================================
% This contain all the information for running main
% TEMPLATE OF THE STRUCT DATI
%=======================================================================================================
%
%  DATI= struct( 'name',              % set the name of the test 
%                'Domain',            % set the domain [x1,x2;y1,y2]
%                'exact_sol',         % set the exact solution
%                'force',             % set the forcing term
%                'grad_exact_1',      % set the first componenet of the gradient of the exact solution
%                'grad_exact_2',      % set the second componenet of the gradient of the exact solution
%                'fem',               % set finite element space
%                'nqn_1D',            % number of quadrature nodes for integrals over lines
%                'nqn_2D',            % number of quadrature nodes for integrals over elements
%                'MeshType',          % set the elements of the mesh  'TS'
%                'refinement_vector', % set the level of refinement for the grid
%                'visual_graph',      % if you want to display the graphical results ['Y','N']
%                'print_out',         % if you want to print out the results ['Y','N']
%                'plot_errors'        % you want to plot the computed errors ['Y','N']
% 
%========================================================================================================

function [Dati]=C_dati(test)

if test=='Test1'
Dati = struct( 'name',             test,...
               ... % Test name
               'name_method',       'SD',... 
               ... % Discretization method ('CG','SUPG','SD')   
               'domain',           [0,1;0,1],...                          
               ... % Domain bounds       
               'mu',               1, ...
               ... % Diffusive term ...
               'sigma',            0, ...
               ... % Reactive term ...
               'beta',             0, ...
               ... % Advection vector ...
               'exact_sol',        '0.*x + 0.*y',...      
               ... % Definition of exact solution
               'force',            '10 * exp(-100 * sqrt((x-0.5).^2 + (y-0.5).^2))',...  
               ... % Forcing term
               'grad_exact_1',     '0.*x + 0.*y',...    
               ... % Definition of exact gradient (x comp) 
               'grad_exact_2',     '0.*x + 0.*y',...   
               ... % Definition of exact gradient (y comp)
               'fem',              'P1',...         
               ... % P1-fe
               'nqn_1D',            4,...           
               ... % Number of quad. points per edge
               'nqn_2D',            3,...           
               ... % Number of quad. points per triangle
               'MeshType',         'TS', ...        
               ... % Triangular Structured mesh
               'refinement_vector', [2,3,4,5],...   
               ... % Refinement levels for  the error analysis
               'visual_graph',      'N',...
               ... % Visualization of the solution
               'plot_errors',       'N' ...
               ...% Compute Errors 
               );



end



