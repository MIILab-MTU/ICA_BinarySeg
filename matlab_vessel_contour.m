function vessel_contour = matlab_vessel_contour(file_path)
close all;
warning('off');
img = imread(file_path);
[rows, columns, numberOfColorChannels] = size(img)

if numberOfColorChannels == 3
    img = rgb2gray(img);
end

img = logical(img);
% contour
vessel_contour = bwmorph(img, 'remove');
