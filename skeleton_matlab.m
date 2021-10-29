function [img]=skeleton_matlab(file_path)
close all;
warning('off');
img = imread(file_path);
[rows, columns, numberOfColorChannels] = size(img)
if numberOfColorChannels == 3
    img = rgb2gray(img);
end
img = logical(img);
img = bwskel(img, 'MinBranchLength', 0);