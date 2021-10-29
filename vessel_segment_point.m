function [rj, cj, re, ce] = vessel_segment_point(file_path)
    img = imread(file_path);
    [rows, columns, numberOfColorChannels] = size(img)
    [rj, cj, re, ce] = findendsjunctions(img, 0)

