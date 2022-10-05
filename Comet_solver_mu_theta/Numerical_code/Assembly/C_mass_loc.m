function [M_loc]=C_mass_loc(dphiq,w_2D,nln,BJ)
%% [M_loc]=C_mass_loc(dphiq,w_2D,nln,BJ)
%==========================================================================
% Build the local mass matrix for the term (uv)
%==========================================================================
%    called in C_matrix2D.m
%
%    INPUT:
%          dphiq       : (array real) evaluation of the basis function on
%                        quadrature nodes
%          w_2D        : (array real) quadrature weights
%          nln         : (integer) number of local unknowns
%          BJ          : (array real) Jacobian of the map 
%
%    OUTPUT:
%          M_loc       :  (array real) Local mass matrix

M_loc=zeros(nln,nln);

for i=1:nln
    for j=1:nln
        for k=1:length(w_2D)
            Binv = inv(BJ(:,:,k));      % inverse
            Jdet = det(BJ(:,:,k));      % determinant 
            M_loc(i,j) = M_loc(i,j) + (Jdet.*w_2D(k)) .* dphiq(1,k,i).* dphiq(1,k,j);
        end
    end
end



                                              
                                              

