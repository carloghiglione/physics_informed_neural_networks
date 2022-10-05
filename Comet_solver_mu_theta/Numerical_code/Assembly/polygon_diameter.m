function [h] = polygon_diameter(coord)

% the longest edge

coord_tmp=[coord; coord(1,:)];
for k=1:size(coord,1)
    length_edges(k)=hypot(coord_tmp(k+1,1)-coord_tmp(k,1), coord_tmp(k+1,2)-coord_tmp(k,2));
end

h=max(length_edges);