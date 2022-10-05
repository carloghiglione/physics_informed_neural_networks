function [K_loc]=C_lap_loc(Grad,w_2D,nln,BJ)
%% [K_loc]=C_lap_loc(Grad,w_2D,nln,BJ)
%==========================================================================
% Build the local stiffness matrix for the term grad(u)grad(v)
%==========================================================================
%    called in C_matrix2D.m
%
%    INPUT:
%          Grad        : (array real) evaluation of the gradient on
%                        quadrature nodes
%          w_2D        : (array real) quadrature weights
%          nln         : (integer) number of local unknowns
%          BJ          : (array real) Jacobian of the map 
%
%    OUTPUT:
%          K_loc       :  (array real) Local stiffness matrix


K_loc=zeros(nln,nln);

%% General implementation -- to be used with general finite element spaces
for i=1:nln
    for j=1:nln
        for k=1:length(w_2D)
            Binv = inv(BJ(:,:,k));   % inverse
            Jdet = det(BJ(:,:,k));   % determinant 
            K_loc(i,j) = K_loc(i,j) + (Jdet.*w_2D(k)) .* ( (Grad(k,:,i) * Binv) * (Grad(k,:,j) * Binv )');
        end
    end
end

%% Equivalent implementation -- valid only for linear finite element space
% for i=1:nln
%     for j=1:nln
%             Binv = inv(BJ(:,:,1));   % inverse
%             Jdet = det(BJ(:,:,1));   % determinant 
%             K_loc(i,j) = K_loc(i,j) + 0.5 * Jdet *  Grad(1,:,i) * Binv * Binv' * Grad(1,:,j)';
%     end
% end


                                              
                                              

