function diameter_map = matlab_diameter_map(file_path)
close all;
warning('off');
img = imread(file_path);
[rows, columns, numberOfColorChannels] = size(img)

if numberOfColorChannels == 3
    img = rgb2gray(img);
end

img = logical(img);

% skeleton
vessel_skeleton = bwskel(img, 'MinBranchLength', 0);

% contour
vessel_contour = bwmorph(img, 'remove');

% dist
[vessel_edtimage, vessel_distance_position_map] = bwdist(~img, 'euclidean');

diameter_map = vessel_edtimage .* vessel_skeleton