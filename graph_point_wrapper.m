function [rj, cj, re, ce, skeleton_img] = graph_point_wrapper(file_path)
    skeleton_img = skeleton_matlab(file_path)
    [rj, cj, re, ce] = findendsjunctions(skeleton_img, 0)
    
