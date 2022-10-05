function [errors_table,rates]=C_convergence_test(test_name)
%% [errors_table,rates]=C_convergence_test(test_name)
%==========================================================================
% Error analysis varying the mesh size h 
%==========================================================================
% Example of usage: [errors_table,rates] = C_convergence_test('Test1')
%
%    INPUT:
%          test_name    : (string)  test case name, see C_dati.m
%
%    OUTPUT:
%          errors_table : (struct) containing the computed errors
%          rates        : (struct) containing the computed rates


warning off;
addpath Assembly
addpath BoundaryConditions
addpath Errors
addpath MeshGeneration
addpath FESpace
addpath Postprocessing


Dati=C_dati(test_name);

refinement_vector=Dati.refinement_vector;
num_test=length(refinement_vector);

for k=1:num_test
    [errors,~,femregion,~] = C_main2D(test_name,refinement_vector(k));
    Error_L2(k)=errors.Error_L2;
    Error_SEMI_H1(k)=errors.Error_SEMI_H1;
    Error_H1(k)=errors.Error_H1;
    Error_inf(k)=errors.Error_inf;
    ne(k)=femregion.ne;    
    h(k)=femregion.h;
    fprintf('==========================================\n');    
    fprintf('End test %i\n',k);
    fprintf('==========================================\n');
end
p=femregion.degree;

%ERROR TABLE
errors_table=struct('ne',ne,...
                   'h',h,...
                   'Error_L2', Error_L2,...
                   'Error_SEMI_H1', Error_SEMI_H1,...
                   'Error_H1', Error_H1,...
                   'Error_inf',Error_inf);


               
%TABLE
rate_L2=log10(Error_L2(2:num_test)./Error_L2(1:num_test-1))./log10(h(2:num_test)./h(1:num_test-1));
rate_SEMI_H1=log10( Error_SEMI_H1(2:num_test)./ Error_SEMI_H1(1:num_test-1))./log10(h(2:num_test)./h(1:num_test-1));
rate_H1=log10( Error_H1(2:num_test)./ Error_H1(1:num_test-1))./log10(h(2:num_test)./h(1:num_test-1));
rate_inf=log10( Error_inf(2:num_test)./ Error_inf(1:num_test-1))./log10(h(2:num_test)./h(1:num_test-1));

rates=struct('rate_L2',rate_L2,...
             'rate_SEMI_H1',rate_SEMI_H1,...
             'rate_H1',rate_H1,...
             'rate_inf',rate_inf);
         

% ERROR PLOTS         
figure()
loglog(h,Error_L2,'-o','Linewidth',2);
hold on
loglog(h,Error_H1,'-o','Linewidth',2);
loglog(h,Error_H1(1)/h(1)*h,'k-','Linewidth',2);
loglog(h,Error_L2(1)/h(1)^2*h.^2,'k--','Linewidth',2);

xlabel('h');
legend('err L^2', 'err H^1', 'h', 'h^2')
 