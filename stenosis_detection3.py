import matlab.engine
import argparse
import json
import numpy as np
import os
import math
import cv2
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pydicom
import pickle
import argparse

from scipy.interpolate import interp1d
from tqdm import tqdm
from skimage import measure

from glob import glob


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Node(object):
    def __init__(self, degree, x, y):
        self.degree = degree
        self.x = x
        self.y = y

    def __str__(self):
        return "degree = {}, x = {}, y = {}".format(self.degree, self.x, self.y)


class VesselSegment(object):
    def __init__(self, node1, node2, vessel_centerline):
        self.node1 = node1
        self.node2 = node2
        self.vessel_centerline = vessel_centerline
        self.vessel_centerline_dist = None
        self.vessel_mask = None
        self.vessel_class = None

        self.stenosis_point = None
        self.stenosis_percent = None
        self.matched = False


def skeleton_by_matlab(file_path, eng, show=False, save_path=None):
    """
    calculate the skeleton by matlab function
    :param file_path: file path to binary image_x
    :return:
    """
    ret = eng.skeleton_matlab(file_path)
    skeleton_image = np.array(ret, dtype=np.int32)
    image_size = skeleton_image.shape[0]
    if show:
        fig, ax = plt.subplots()
        ax.imshow(skeleton_image, cmap=plt.cm.gray)
        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.savefig(save_path)
        plt.close()

    return skeleton_image


def remove_isolated_segment(skeleton_image, show=False, save_path=None):
    labeling = measure.label(skeleton_image)
    regions = measure.regionprops(labeling)
    largest_region = None
    image_size = skeleton_image.shape[0]
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    bin_image = np.zeros_like(skeleton_image)
    for coord in largest_region.coords:
        bin_image[coord[0], coord[1]] = 1

    if show:
        fig, ax = plt.subplots()
        ax.imshow(bin_image, cmap=plt.cm.gray)
        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)
        plt.savefig(save_path)
        plt.close()

    return bin_image


def calculate_end_and_joint_points_by_matlab(vessel_file_path, eng, show=True, save_path=None):
    """
    generate the joint points and end points according to matlab code
    :param vessel_file_path:
    :param eng:
    :param show:
    :return:
        points: each point is a tuple, and the first element is the column, the second is the row
    """
    ret = eng.graph_point_wrapper(vessel_file_path, nargout=5)
    rj, cj, re, ce, skeleton_image = np.array(ret[0], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[1], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[2], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[3], dtype=np.int32).flatten() - 1, \
                                     np.array(ret[4], dtype=np.int32)

    image_size = skeleton_image.shape[0]
    end_points = []
    joint_points = []

    points = []

    for i in range(len(rj)):
        joint_points.append((rj[i], cj[i]))
    for i in range(len(ce)):
        end_points.append((re[i], ce[i]))

    # for each point, the first element is the column, the second is the row

    for point in joint_points:
        adj_matrix = skeleton_image[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2]
        degree = np.sum(adj_matrix == 1) - skeleton_image[point[0], point[1]]
        points.append(Node(degree, point[0], point[1]))
        # print("joint point {}, degree = {}".format(point, degree))

    for point in end_points:
        adj_matrix = skeleton_image[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2]
        degree = np.sum(adj_matrix == 1) - skeleton_image[point[0], point[1]]
        points.append(Node(degree, point[0], point[1]))
        # print("end point {}, degree = {}".format(point, degree))

    if show:
        vessel_image = cv2.imread(vessel_file_path)

        fig, ax = plt.subplots()
        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)

        plt.imshow(vessel_image, cmap='gray')
        plt.imshow(skeleton_image, alpha=0.6, cmap='gray')
        plt.scatter(cj, rj, c='r', linewidth=1, marker='*')
        plt.scatter(ce, re, c='g', linewidth=1, marker='+')
        # plt.show()
        plt.savefig(save_path)
        plt.close()
    return points


def graph_generation_by_connection(nodes, skeleton_img, eng=None, show=False, save_path=None, tmp_dir=None):
    vessel_objects = []
    image_size = skeleton_img.shape[0]
    for node in nodes:
        # skeleton_img[node.x, node.y] = 0  # OLD
        for i in range(node.x - 1, node.x + 2):
            for j in range(node.y - 1, node.y + 2):
                if i < skeleton_img.shape[0] and j < skeleton_img.shape[1]:
                    skeleton_img[i, j] = 0

    # labeling = measure.label(skeleton_img, neighbors=8, connectivity=1)
    labeling = measure.label(skeleton_img)

    for idx, label in enumerate(np.unique(labeling)[1:]):
        vessel_segment = np.zeros_like(labeling)
        vessel_segment[labeling == label] = 1

        # find end points by matlab
        # cv2.imwrite("tmp.png", vessel_segment)
        # find end points by matlab
        cv2.imwrite(os.path.join(tmp_dir, "tmp.png"), vessel_segment)
        ret = eng.vessel_segment_point(os.path.join(tmp_dir, "tmp.png"), nargout=4)
        rj, cj, re, ce = np.array(ret[0], dtype=np.int32).flatten() - 1, \
                         np.array(ret[1], dtype=np.int32).flatten() - 1, \
                         np.array(ret[2], dtype=np.int32).flatten() - 1, \
                         np.array(ret[3], dtype=np.int32).flatten() - 1
        if re.shape[0] == 2:
            # print("rj {}, cj {}, re {}, ce {}".format(rj, cj, re, ce))
            edge_points = []
            for i in range(2):
                near_dist = np.inf
                near_node = None
                for node in nodes:
                    end_point_x, end_point_y = re[i], ce[i]
                    dist = np.sqrt((end_point_x - node.x) ** 2 + (end_point_y - node.y) ** 2)
                    if dist < near_dist:
                        near_dist = dist
                        near_node = node
                edge_points.append(near_node)
            vessel_objects.append(VesselSegment(edge_points[0], edge_points[1], vessel_segment))

        else:
            print("rj {}, cj {}, re {}, ce {}".format(rj, cj, re, ce))
            plt.imshow(vessel_segment, cmap="gray")
            plt.savefig(os.path.join(tmp_dir, "tmpa.png"))
            plt.close()

    if show:
        fig, ax = plt.subplots()
        plt.axis('off')
        fig.set_size_inches(image_size / 100, image_size / 100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
        plt.margins(0, 0)

        plt.imshow(skeleton_img, cmap='gray')
        plt.scatter([n.y for n in nodes if n.degree > 2], [n.x for n in nodes if n.degree > 2],
                    c='r', linewidth=1, marker='*')
        plt.scatter([n.y for n in nodes if n.degree == 1], [n.x for n in nodes if n.degree == 1],
                    c='g', linewidth=1, marker='+')
        plt.scatter([n.y for n in nodes if n.degree == 2], [n.x for n in nodes if n.degree == 2],
                    c='b', linewidth=1, marker='o')
        plt.savefig(save_path)
        plt.close()

    return vessel_objects


def visualize_graph(vessel_infos, original_image, save_path):
    image_size = original_image.shape[0]
    fig, ax = plt.subplots()
    ax.imshow(original_image, cmap="gray")

    for vessel_info in vessel_infos:
        if vessel_info.node1.degree == 1:
            plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=1, marker='+')
        elif vessel_info.node1.degree == 2:
            plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='b', linewidth=1, marker='o')
        else:
            plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='r', linewidth=1, marker='*')

        if vessel_info.node2.degree == 1:
            plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='g', linewidth=1, marker='+')
        elif vessel_info.node2.degree == 2:
            plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='b', linewidth=1, marker='o')
        else:
            plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='r', linewidth=1, marker='*')

        plt.plot([vessel_info.node1.y, vessel_info.node2.y], [vessel_info.node1.x, vessel_info.node2.x], '-',
                 color="#ffffff")

    plt.axis('off')
    fig.set_size_inches(image_size / 100, image_size / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close()
    # plt.show()


def visualize_centerline(vessel_infos, original_image, save_path, with_ori=True, with_joint=True):
    image_size = original_image.shape[0]
    fig, ax = plt.subplots()
    # ax.imshow(original_image, cmap="gray")
    if with_ori:
        plt.imshow(original_image, cmap='gray')

    plt.axis('off')
    fig.set_size_inches(image_size / 100, image_size / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    centerline_img = np.zeros_like(original_image)

    for vessel_info in vessel_infos:
        if vessel_info.node1.degree == 1:
            if with_joint:
                # plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=2, marker='+', s=100)
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='limegreen', linewidth=2, marker='*', s=100)
        elif vessel_info.node1.degree == 2:
            plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='b', linewidth=1, marker='o')
        else:
            if with_joint:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='r', linewidth=2, marker='*', s=100)

        if vessel_info.node2.degree == 1:
            if with_joint:
                # plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=2, marker='+', s=100)
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='limegreen', linewidth=2, marker='*', s=100)
        elif vessel_info.node2.degree == 2:
            plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='b', linewidth=1, marker='o')
        else:
            if with_joint:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='r', linewidth=2, marker='*', s=100)

        centerline_img[vessel_info.vessel_centerline > 0] = 1.

    if with_ori:
        plt.imshow(centerline_img, alpha=0.6, cmap='gray')
    else:
        plt.imshow(centerline_img, cmap='gray')
    plt.savefig(save_path)
    plt.close()


def visualize_segmentation_and_stenosis_all(vessel_objs_in_gt,
                                            vessel_objs_in_pred,
                                            original_image,
                                            save_path):
    image_size = original_image.shape[0]
    fig, ax = plt.subplots()
    # ax.imshow(original_image, cmap="gray")

    plt.axis('off')
    fig.set_size_inches(image_size / 100, image_size / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    # draw points
    # draw points
    for vessel_info in vessel_objs_in_pred:
        if vessel_info.stenosis_point:
            plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='lime', edgecolors='lime',
                        linewidth=2, marker='*', s=100)

    for vessel_info in vessel_objs_in_gt:
        if vessel_info.stenosis_point:
            plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='fuchsia', edgecolors='fuchsia',
                        linewidth=2, marker='*', s=100)

    # draw circles
    for vessel_info in vessel_objs_in_gt:
        if vessel_info.stenosis_point:
            if vessel_info.matched:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='r', edgecolors='deepskyblue',
                            linewidth=1, marker='o', s=200)
            else:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='w', edgecolors='white',
                            linewidth=1, marker='o', s=200)  # FN
    for vessel_info in vessel_objs_in_pred:
        if vessel_info.stenosis_point:
            if not vessel_info.matched:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='r', edgecolors='white',
                            linewidth=1, marker='o', s=200)  # FP

    plt.imshow(original_image, cmap='gray')
    plt.savefig(save_path)
    plt.close()


def visualize_segmentation_and_stenosis(filtered_vessel_objs,
                                        original_image,
                                        segmentation_file_path,
                                        eng,
                                        save_path,
                                        with_joint=True):
    ret = eng.matlab_vessel_contour(segmentation_file_path)
    vessel_contour = np.array(ret, dtype=np.float32)

    image_size = original_image.shape[0]
    fig, ax = plt.subplots()
    # ax.imshow(original_image, cmap="gray")

    plt.axis('off')
    fig.set_size_inches(image_size / 100, image_size / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    centerline_img = np.zeros_like(original_image)
    for vessel_info in filtered_vessel_objs:
        if vessel_info.node1.degree == 1:
            if with_joint:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='g', linewidth=1, marker='+')
        elif vessel_info.node1.degree == 2:
            plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='b', linewidth=1, marker='o')
        else:
            if with_joint:
                plt.scatter(vessel_info.node1.y, vessel_info.node1.x, c='r', linewidth=1, marker='*')

        if vessel_info.node2.degree == 1:
            if with_joint:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='g', linewidth=1, marker='+')
        elif vessel_info.node2.degree == 2:
            plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='b', linewidth=1, marker='o')
        else:
            if with_joint:
                plt.scatter(vessel_info.node2.y, vessel_info.node2.x, c='r', linewidth=1, marker='*')

        centerline_img[vessel_info.vessel_centerline > 0] = 1

    for vessel_info in filtered_vessel_objs:
        if vessel_info.stenosis_point:
            if vessel_info.stenosis_percent > 0.75:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='limegreen', edgecolors='limegreen',
                            linewidth=2, marker='o', s=100)
            elif vessel_info.stenosis_percent <= 0.75 and vessel_info.stenosis_percent > 0.5:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='cyan', edgecolors='cyan',
                            linewidth=2, marker='o', s=100)
            elif vessel_info.stenosis_percent <= 0.5 and vessel_info.stenosis_percent > 0.3:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='yellow', edgecolors='yellow',
                            linewidth=2, marker='o', s=100)
            else:
                plt.scatter(vessel_info.stenosis_point[1], vessel_info.stenosis_point[0], c='red', edgecolors='red',
                            linewidth=2, marker='o', s=100)

    plt.imshow(original_image, cmap='gray')
    plt.imshow(vessel_contour, alpha=0.3, cmap='gray')
    plt.imshow(centerline_img, alpha=0.3, cmap='gray')
    plt.savefig(save_path)
    plt.close()


def diameter_map(binary_file_path, eng):
    ret = eng.matlab_diameter_map(binary_file_path)
    dist_map = np.array(ret, dtype=np.float32)
    return dist_map


############################## fucntions for filtering vessel #################################
def get_spacing(patient_name, dicom_spacing_file_path="./data_nj/spacing.txt"):
    patient_id = patient_name[:patient_name.find("_")]
    for line in open(dicom_spacing_file_path, "rt").readlines():
        if line.split(",")[0] == patient_id:
            return float(line.split(",")[1])


def filter_vessel(vessel_objs, dist_map, patient_name, mean_dist_threshold, spacing_file_path):
    v_objs = []
    pixel_spacing = get_spacing(patient_name, spacing_file_path)

    for idx, vessel_obj in enumerate(vessel_objs):
        segment_dist_map = vessel_obj.vessel_centerline * dist_map
        # assign distance map to each vessel object
        vessel_obj.vessel_centerline_dist = segment_dist_map
        # average_dist = np.mean(segment_dist_map[segment_dist_map > 0])
        max_dist = np.max(segment_dist_map[segment_dist_map > 0])

        if max_dist * pixel_spacing > mean_dist_threshold:
            v_objs.append(vessel_obj)
    return v_objs


def nearest_points(point, centerline):
    search_points = np.where(centerline > 0)
    min_dist = np.inf
    target_point = (np.inf, np.inf)
    for i in range(len(search_points[0])):
        # print(np.sum(np.square(np.array(point) - np.array((search_points[0][i], search_points[1][i])))))
        if np.sum(np.square(np.array(point) - np.array((search_points[0][i], search_points[1][i])))) < min_dist:
            min_dist = np.sum(np.square(np.array(point) - np.array((search_points[0][i], search_points[1][i]))))
            target_point = (search_points[0][i], search_points[1][i])
    return target_point


def traverse_segment(vessel_obj, segment_dist_map):
    current_search_point = (vessel_obj.node1.x, vessel_obj.node1.y)
    selected_points = []
    centerline_map = np.zeros_like(segment_dist_map)
    centerline_map[segment_dist_map > 0] = 1

    for i in range(np.count_nonzero(segment_dist_map)):
        selected_point = nearest_points(current_search_point, centerline_map)
        # print(selected_point)
        selected_points.append(selected_point)

        current_search_point = selected_point
        centerline_map[selected_point[0], selected_point[1]] = 0

    diameters = []
    for point in selected_points:
        diameters.append(segment_dist_map[point[0], point[1]])

    return selected_points, diameters


#################################### functions for stenosis detection #####################################
def stenosis_detection(diameters, radius=3, stenosis_threshold=0.7):
    stenosis_point_idxs = []
    stenosis_percents = []

    l_min = np.diff(np.sign(np.diff(diameters))) == -2  # logical vector for the local min value
    min_idxs = np.where(l_min == True)
    for min_idx in min_idxs[0]:
        # print(f"{min_idx}  len(diameters) = {len(diameters)}")
        if (min_idx > radius) and (min_idx < (len(diameters) - radius)):
            current_point_diam = diameters[min_idx]
            contextual_diams = diameters[min_idx - radius: min_idx + radius + 1]
            contextual_diams.pop(radius)
            contextual_diams.pop(radius - 1);
            contextual_diams.pop(radius - 2);
            contextual_diams.pop(radius - 3);
            contextual_diams.pop(radius + 1);
            contextual_diams.pop(radius + 2);
            contextual_diams.pop(radius + 3);

            contextual_mean = np.mean(contextual_diams)
            if current_point_diam / contextual_mean < stenosis_threshold:
                stenosis_point_idxs.append(min_idx)
                stenosis_percents.append(current_point_diam / contextual_mean)

    if len(stenosis_point_idxs) > 1:
        stenosis_percents = []
        for point_idx in stenosis_point_idxs:
            current_point_diam = diameters[point_idx]
            contextual_diams = diameters[point_idx - radius: point_idx + radius + 1]
            contextual_diams.pop(radius)
            contextual_diams.pop(radius - 1);
            contextual_diams.pop(radius - 2);
            contextual_diams.pop(radius - 3);
            contextual_diams.pop(radius + 1);
            contextual_diams.pop(radius + 2);
            contextual_diams.pop(radius + 3);

            contextual_mean = np.mean(contextual_diams)
            stenosis_percents.append(current_point_diam / contextual_mean)

    return stenosis_point_idxs, stenosis_percents


def stenosis_detection2(diameters, radius=3, stenosis_threshold=0.7):
    stenosis_point_idxs = []
    stenosis_percents = []

    diameters = np.array(diameters)
    for i in range(len(diameters)):
        if (i > radius) and (i < (len(diameters) - radius)):
            current_point_diam = diameters[i]
            contextual_diams = np.append(diameters[i - radius: radius - 3], diameters[i + 3: i + radius + 1])
            contextual_mean = np.mean(contextual_diams)

            if current_point_diam / contextual_mean < stenosis_threshold:
                stenosis_point_idxs.append(i)
                stenosis_percents.append(current_point_diam / contextual_mean)

    if len(stenosis_point_idxs) > 1:
        stenosis_percents = []
        for point_idx in stenosis_point_idxs:
            current_point_diam = diameters[point_idx]
            contextual_diams = np.append(diameters[point_idx - radius: radius],
                                         diameters[point_idx + 1: point_idx + radius + 1])

            contextual_mean = np.mean(contextual_diams)
            stenosis_percents.append(current_point_diam / contextual_mean)

    return stenosis_point_idxs, stenosis_percents


def stenosis_detection3(diameters, radius=3, stenosis_threshold=0.7):
    stenosis_point_idxs = []
    stenosis_percents = []

    diameters = np.array(diameters)
    if len(diameters) > radius * 2:
        diam_max = np.max(diameters[radius: len(diameters) - radius])
        diam_min = np.min(diameters[radius: len(diameters) - radius])
        if diam_min / diam_max < stenosis_threshold:
            stenosis_percents.append(diam_min / diam_max)
            stenosis_point_idxs.append(np.argmin(diameters[radius: len(diameters) - radius]) + radius)

    return stenosis_point_idxs, stenosis_percents


def find_stenosis_points(ori_img_path, binary_img_path, eng, save_dir, patient_name, detection_method, spacing_file_path,
                         mean_dist_threshold=2.5, detection_radius=3, stenosis_threshold=0.9):
    # step: calculate the skeleton image
    # skeleton_image = skeleton_by_matlab(gt_img_path, eng, show=True,
    #                                     save_path=os.path.join(save_dir, "{}_step1_skeleton.png".format(patient_name)))
    # skeleton_image = remove_isolated_segment(skeleton_image, show=True,
    #                                          save_path=os.path.join(save_dir, "{}_step1_connected_skeleton.png".format(
    #                                              patient_name)))
    skeleton_image = skeleton_by_matlab(binary_img_path, eng, show=False,
                                        save_path=os.path.join(save_dir, "{}_step1_skeleton.png".format(patient_name)))
    skeleton_image = remove_isolated_segment(skeleton_image, show=False,
                                             save_path=os.path.join(save_dir, "{}_step1_connected_skeleton.png".format(
                                                 patient_name)))

    # step 2: calculate end points and joint points
    nodes = calculate_end_and_joint_points_by_matlab(binary_img_path, eng, show=True,
                                                     save_path=os.path.join(save_dir,
                                                                            "{}_step2_skeleton_with_points.png".format(
                                                                                patient_name)))

    original_vessel_objects = graph_generation_by_connection(nodes, skeleton_image, eng, show=True,
                                                             save_path=os.path.join(save_dir,
                                                                                    "{}_step2_skeleton_with_all_points.png".format(
                                                                                        patient_name)),
                                                             tmp_dir=save_dir)

    # visualize_graph(vessel_objects, cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
    #                 save_path=os.path.join(save_dir, "{}_step2_generated_graph.png".format(patient_name)))
    visualize_centerline(original_vessel_objects, cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                         save_path=os.path.join(save_dir, "{}_step2_generated_centerline.png".format(patient_name)))
    visualize_centerline(original_vessel_objects, cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                         save_path=os.path.join(save_dir,
                                                "{}_step2_generated_centerline_pure.png".format(patient_name)),
                         with_ori=False, with_joint=False)

    # step3: calculate the distance map
    dist_map = diameter_map(binary_img_path, eng)

    # step4: filter vessel objects: remove the vessel segment with the average diameter less than threshold
    filtered_vessel_objects = filter_vessel(original_vessel_objects, dist_map, patient_name[:patient_name.rfind("_y")],
                                            mean_dist_threshold=mean_dist_threshold, spacing_file_path=spacing_file_path)

    visualize_centerline(filtered_vessel_objects, cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                         save_path=os.path.join(save_dir, "{}_step4_filtered_centerline.png".format(patient_name)))
    visualize_centerline(filtered_vessel_objects, cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                         save_path=os.path.join(save_dir, "{}_step4_filtered_centerline_pure.png".format(patient_name)),
                         with_ori=False, with_joint=False)

    # step5: find stenosis points
    stenosis_points = []
    stenosis_percents = []

    for i, v_obj in enumerate(filtered_vessel_objects):
        segment_dist_map = v_obj.vessel_centerline * dist_map
        traversed_points, diams = traverse_segment(v_obj, segment_dist_map)
        # print(traversed_points)

        plt.imshow(segment_dist_map)
        plt.savefig("tmp.png")
        plt.close()

        if detection_method == "local_min":
            stenosis_point_idxs, stenosis_ps = stenosis_detection(diams, radius=detection_radius,
                                                                  stenosis_threshold=stenosis_threshold)
        elif detection_method == "local_mean":
            stenosis_point_idxs, stenosis_ps = stenosis_detection2(diams, radius=detection_radius,
                                                                   stenosis_threshold=stenosis_threshold)
        elif detection_method == "min_to_max":
            stenosis_point_idxs, stenosis_ps = stenosis_detection3(diams, radius=detection_radius,
                                                                   stenosis_threshold=stenosis_threshold)
        else:
            raise NotImplemented(f"stenosis detection method {detection_method} is not supported!")

        if len(stenosis_point_idxs) > 0:
            segment_points = []
            segment_percents = []

            for i, stenosis_point_idx in enumerate(stenosis_point_idxs):
                # detected stenosis points
                # print(f"{traversed_points[stenosis_point_idx][0]}-{traversed_points[stenosis_point_idx][1]}")
                stenosis_points.append(
                    (traversed_points[stenosis_point_idx][0], traversed_points[stenosis_point_idx][1]))
                stenosis_percents.append(stenosis_ps[i])

                segment_points.append(
                    (traversed_points[stenosis_point_idx][0], traversed_points[stenosis_point_idx][1]))
                segment_percents.append(stenosis_ps[i])

            min_percent_point_idx = np.argmin(segment_percents)
            v_obj.stenosis_point = (segment_points[min_percent_point_idx][0], segment_points[min_percent_point_idx][1])
            v_obj.stenosis_percent = segment_percents[min_percent_point_idx]

    # visualize
    visualize_segmentation_and_stenosis(filtered_vessel_objects,
                                        cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                                        binary_img_path, eng,
                                        save_path=os.path.join(save_dir, "{}_step5_detected_stenosis_points.png".format(
                                            patient_name)))
    visualize_segmentation_and_stenosis(filtered_vessel_objects,
                                        cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                                        binary_img_path, eng,
                                        save_path=os.path.join(save_dir,
                                                               "{}_step5_detected_stenosis_points_pure.png".format(
                                                                   patient_name)),
                                        with_joint=False)

    return stenosis_points, stenosis_percents, original_vessel_objects, filtered_vessel_objects


def distance_between_nodes(node1: Node, node2: Node):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


####################### stenosis point compare based on correspoing vessel segment #####################################
def stenosis_evaluation(ori_img_path, gt_img_path, pred_img_path, eng, save_dir, detection_method, spacing_file_path,
                        mean_dist_threshold=2.5, detection_radius=3, stenosis_threshold=0.9):
    patient_name = gt_img_path[gt_img_path.rfind("/") + 1: gt_img_path.rfind(".png")]

    stenosis_points_gt, stenosis_percents_gt, origianl_vessel_objects_gt, filtered_vessel_objects_gt = find_stenosis_points(
        ori_img_path, gt_img_path, eng, save_dir,
        "{}_gt".format(patient_name), detection_method,
        mean_dist_threshold=mean_dist_threshold,
        detection_radius=detection_radius,
        stenosis_threshold=stenosis_threshold,
    spacing_file_path=spacing_file_path)
    stenosis_points_pred, stenosis_percents_pred, origianl_vessel_objects_pred, filtered_vessel_objects_pred = find_stenosis_points(
        ori_img_path, pred_img_path, eng, save_dir,
        "{}_pred".format(patient_name), detection_method,
        mean_dist_threshold=mean_dist_threshold,
        detection_radius=detection_radius,
        stenosis_threshold=stenosis_threshold,
    spacing_file_path=spacing_file_path)

    # stenosis point matching based on vessel segment
    with open(os.path.join(save_dir, "stenosis_match.txt"), "wt") as f:
        f.write("index,p_gt_x,p_gt_y,p_pred_x,p_pred_y,stenosis_gt,stenosis_pred,type\n")
        count = 0
        # 1. match stenotic point on the same vessel segment
        for i in range(len(filtered_vessel_objects_gt)):
            if filtered_vessel_objects_gt[i].stenosis_point and filtered_vessel_objects_gt[i].matched == False:
                # current vessel segment in gt has stenotic point
                segment_match_flag = 0
                # find candidacy segments
                candidacy_segment_index_in_pred = []
                for j in range(len(filtered_vessel_objects_pred)):
                    # 1. match vessel segment
                    if distance_between_nodes(filtered_vessel_objects_gt[i].node1,
                                              filtered_vessel_objects_pred[j].node1) < detection_radius and \
                            distance_between_nodes(filtered_vessel_objects_gt[i].node2,
                                                   filtered_vessel_objects_pred[j].node2) < detection_radius:
                        if filtered_vessel_objects_pred[j].stenosis_point:
                            segment_match_flag = 1
                            candidacy_segment_index_in_pred.append(j)
                    elif distance_between_nodes(filtered_vessel_objects_gt[i].node1,
                                                filtered_vessel_objects_pred[j].node2) < detection_radius and \
                            distance_between_nodes(filtered_vessel_objects_gt[i].node2,
                                                   filtered_vessel_objects_pred[j].node1) < detection_radius:
                        if filtered_vessel_objects_pred[j].stenosis_point:
                            segment_match_flag = 1
                            candidacy_segment_index_in_pred.append(j)

                if segment_match_flag == 1:
                    # find the most similary vessel
                    centerline_len_diffs = []
                    for j in candidacy_segment_index_in_pred:
                        centerline_len_diffs.append(np.abs(
                            np.sum(filtered_vessel_objects_gt[i].vessel_centerline > 0) - np.sum(
                                filtered_vessel_objects_pred[j].vessel_centerline > 0)))

                    # 2. matched segment

                    # match
                    f.write("%d,%f,%f,%f,%f,%f,%f,match1(TP)\n" % (count,
                                                                   filtered_vessel_objects_gt[i].stenosis_point[0],
                                                                   filtered_vessel_objects_gt[i].stenosis_point[1],
                                                                   filtered_vessel_objects_pred[candidacy_segment_index_in_pred[np.argmin(centerline_len_diffs)]].stenosis_point[0],
                                                                   filtered_vessel_objects_pred[candidacy_segment_index_in_pred[np.argmin(centerline_len_diffs)]].stenosis_point[1],
                                                                   filtered_vessel_objects_gt[i].stenosis_percent,
                                                                   filtered_vessel_objects_pred[candidacy_segment_index_in_pred[np.argmin(centerline_len_diffs)]].stenosis_percent))
                    f.flush()
                    filtered_vessel_objects_gt[i].matched = True
                    filtered_vessel_objects_pred[candidacy_segment_index_in_pred[np.argmin(centerline_len_diffs)]].matched = True

        # 2. match stenotic point based on the nearest point
        for i in range(len(filtered_vessel_objects_gt)):
            if filtered_vessel_objects_gt[i].stenosis_point and filtered_vessel_objects_gt[i].matched == False:
                for j in range(len(filtered_vessel_objects_pred)):
                    if filtered_vessel_objects_pred[j].stenosis_point:
                        point_radius = detection_radius * 1.5
                        if np.sqrt((filtered_vessel_objects_gt[i].stenosis_point[0] -
                                    filtered_vessel_objects_pred[j].stenosis_point[0]) ** 2 + \
                                   (filtered_vessel_objects_gt[i].stenosis_point[1] -
                                    filtered_vessel_objects_pred[j].stenosis_point[1]) ** 2) < point_radius:
                            f.write("%d,%f,%f,%f,%f,%f,%f,match2(TP)\n" % (count,
                                                                           filtered_vessel_objects_gt[i].stenosis_point[0],
                                                                           filtered_vessel_objects_gt[i].stenosis_point[1],
                                                                           filtered_vessel_objects_pred[j].stenosis_point[0],
                                                                           filtered_vessel_objects_pred[j].stenosis_point[1],
                                                                           filtered_vessel_objects_gt[i].stenosis_percent,
                                                                           filtered_vessel_objects_pred[j].stenosis_percent))
                            f.flush()
                            filtered_vessel_objects_gt[i].matched = True
                            filtered_vessel_objects_pred[j].matched = True

        ###################################### visualization ############################################################
        visualize_segmentation_and_stenosis_all(filtered_vessel_objects_gt, filtered_vessel_objects_pred,
                                                cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE),
                                                save_path=os.path.join(save_dir,
                                                                       "{}_stenosis_compare.png".format(patient_name)))

        # 3. mismatched stenotic point
        for i in range(len(filtered_vessel_objects_gt)):
            if filtered_vessel_objects_gt[i].stenosis_point and filtered_vessel_objects_gt[i].matched == False:
                # NEW TN TO FN
                f.write("%d,%f,%f,%f,%f,%f,%f,miss(FN)\n" % (count,
                                                             filtered_vessel_objects_gt[i].stenosis_point[0],
                                                             filtered_vessel_objects_gt[i].stenosis_point[1],
                                                             0.,
                                                             0.,
                                                             filtered_vessel_objects_gt[i].stenosis_percent,
                                                             1.))  # miss point as FN
                f.flush()
                filtered_vessel_objects_gt[i].matched = True

        # 4. FP samples
        for j in range(len(filtered_vessel_objects_pred)):
            if filtered_vessel_objects_pred[j].stenosis_point and filtered_vessel_objects_pred[j].matched == False:
                f.write("%d,%f,%f,%f,%f,%f,%f,type3(FP)\n" % (count,
                                                              0.,
                                                              0.,
                                                              filtered_vessel_objects_pred[j].stenosis_point[0],
                                                              filtered_vessel_objects_pred[j].stenosis_point[1],
                                                              1.,
                                                              filtered_vessel_objects_pred[j].stenosis_percent))

################################# stenosis evaluation (stenosis point in vessel segment matching) ##########################\###########################3
def read_file(file_path):
    stenosis_points = []
    stenosis_percents = []

    with open(file_path, "rt") as f:
        lines = f.readlines()
        if len(lines) > 0:
            for i in range(len(lines)):
                if i == 0:
                    continue
                else:
                    point_x = float(lines[i].split(",")[1])
                    point_y = float(lines[i].split(",")[2])
                    percent = float(lines[i].split(",")[3])
                    stenosis_points.append((point_x, point_y))
                    stenosis_percents.append(percent)

    return stenosis_points, stenosis_percents


def nearest_point_evaluation(point, point_list, radius):
    if len(point_list) > 0:
        distances = []
        for i in range(len(point_list)):
            distances.append(
                np.sqrt(np.power(point_list[i][0] - point[0], 2) + np.power(point_list[i][1] - point[1], 2)))
        min_dist_idx = np.argmin(distances)
        if np.sqrt(
                np.power(point_list[min_dist_idx][0] - point[0], 2) + np.power(point_list[min_dist_idx][1] - point[1],
                                                                               2)) < radius:
            return min_dist_idx
        else:
            return -1
    else:
        return -1


def generate_report(process_dir, search_radius):
    patient_folders = os.listdir(process_dir)

    total_stenosis_points = 0.
    total_detected_points = 0.
    stenosis_gts = []
    stenosis_preds = []

    # patients_in_test = []
    # for patient in glob("./vessel_nj_xnet_cv4/0701/*_x.png"):
    #     patient_name = patient[patient.rfind("/") + 1: patient.rfind("_x.png")]
    #     patients_in_test.append(patient_name)

    for patient_folder in patient_folders:
        # if os.path.isdir(os.path.join(process_dir, patient_folder)) and patient_folder in patients_in_test:
        if os.path.isdir(os.path.join(process_dir, patient_folder)):
            stenosis_points_in_pred, stenosis_percents_in_pred = read_file(
                os.path.join(process_dir, patient_folder, "stenosis_pred.txt"))
            stenosis_points_in_gt, stenosis_percents_in_gt = read_file(
                os.path.join(process_dir, patient_folder, "stenosis_gt.txt"))

            detected_stenosis = 0
            stenosis_mse = []

            if len(stenosis_points_in_gt) > 0:
                for i in range(len(stenosis_points_in_gt)):
                    point_idx = nearest_point_evaluation(stenosis_points_in_gt[i], stenosis_points_in_pred,
                                                         search_radius)
                    if point_idx != -1:
                        detected_stenosis += 1
                        stenosis_mse.append(
                            np.power(stenosis_percents_in_gt[i] - stenosis_percents_in_pred[point_idx], 2))

                        stenosis_gts.append(stenosis_percents_in_gt[i])
                        stenosis_preds.append(stenosis_percents_in_pred[point_idx])
                    else:
                        stenosis_mse.append(np.power(stenosis_percents_in_gt[i], 1))

                        stenosis_gts.append(stenosis_percents_in_gt[i])
                        stenosis_preds.append(1)

                print(f"{patient_folder}, "
                      f"# stenosis point in gt {len(stenosis_points_in_gt)}, "
                      f"# detected = {detected_stenosis},"
                      f" stenosis percent mse = {np.mean(stenosis_mse)}")
            else:
                print(f"{patient_folder}, # stenosis point in gt = 0")

            total_stenosis_points += len(stenosis_points_in_gt)
            total_detected_points += detected_stenosis

    print(f"total_stenosis points = {total_stenosis_points}, detected = {total_detected_points},"
          f" ratio = {total_detected_points / total_stenosis_points},"
          f" percents MSE = {np.mean(np.square(np.array(stenosis_gts) - np.array(stenosis_preds)))}")

    with open(os.path.join(process_dir, "evaluation_radius_{}.txt".format(search_radius)), "wt+") as f:
        f.write(f"total_stenosis points = {total_stenosis_points}, detected = {total_detected_points},"
                f" ratio = {total_detected_points / total_stenosis_points},"
                f" percents MSE = {np.mean(np.square(np.array(stenosis_gts) - np.array(stenosis_preds)))}")


def generate_report_on_segment(process_dir):
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    patient_folders = os.listdir(process_dir)

    total_stenosis = 0.
    total_detected = 0.
    stenosis_gts = []
    stenosis_preds = []

    # patients_in_test = []
    # for patient in glob("./vessel_nj_xnet_cv4/0701/*_x.png"):.
    #     patient_name = patient[patient.rfind("/") + 1: patient.rfind("_x.png")]
    #     patients_in_test.append(patient_name)

    for patient_folder in patient_folders:
        # if os.path.isdir(os.path.join(process_dir, patient_folder)) and patient_folder in patients_in_test:
        if os.path.isdir(os.path.join(process_dir, patient_folder)):
            lines = open(os.path.join(process_dir, patient_folder, "stenosis_match.txt"), "rt").readlines()
            if len(lines) > 1:
                for i in range(1, len(lines)):
                    total_stenosis += 1
                    if lines[i].split(",")[7].startswith("match"):
                        total_detected += 1
                    stenosis_gts.append(float(lines[i].split(",")[5]))
                    stenosis_preds.append(float(lines[i].split(",")[6]))

    print(f"total_stenosis points = {total_stenosis},"
          f" detected = {total_detected},"
          f" ratio = {total_detected / total_stenosis},"
          f" percents MSE = {np.mean(np.square(np.array(stenosis_gts) - np.array(stenosis_preds)))}")

    with open(os.path.join(process_dir, "evaluation_on_segment.txt"), "wt") as f:
        f.write(f"total_stenosis points = {total_stenosis},"
                f" detected = {total_detected},"
                f" ratio = {total_detected / total_stenosis},"
                f" percents MSE = {np.mean(np.square(np.array(stenosis_gts) - np.array(stenosis_preds)))}")


def generate_report_on_tpfn_classified(process_dir, only_test):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr

    patient_folders = os.listdir(process_dir)

    if only_test:
        patient_folders = []
        for patient in glob("./vessel_nj_xnet_cv4/0701/*_x.png"):
            patient_name = patient[patient.rfind("/") + 1: patient.rfind("_x.png")]
            patient_folders.append(patient_name)

    # 1-24%, 25-49 %, 50- 69%, >= 70%
    tp = {"all": 0, "minimal": 0, "mild": 0, "moderate": 0, "severe": 0.}
    fn = {"all": 0, "minimal": 0, "mild": 0, "moderate": 0, "severe": 0.}
    fp = {"all": 0, "minimal": 0, "mild": 0, "moderate": 0, "severe": 0.}
    tn_percents_gt = {"all": [], "minimal": [], "mild": [], "moderate": [], "severe": []}
    tn_percents_pred = {"all": [], "minimal": [], "mild": [], "moderate": [], "severe": []}

    for patient_folder in patient_folders:
        if os.path.isdir(os.path.join(process_dir, patient_folder)):
            lines = open(os.path.join(process_dir, patient_folder, "stenosis_match.txt"), "rt").readlines()
            if len(lines) > 1:
                for i in range(1, len(lines)):
                    if lines[i].split(",")[7].strip().endswith("(TP)"):
                        if float(lines[i].split(",")[5].strip()) > 0.75:
                            tp["minimal"] += 1
                            tn_percents_gt["minimal"].append(float(lines[i].split(",")[5].strip()))
                            tn_percents_pred["minimal"].append(float(lines[i].split(",")[6].strip()))
                        elif 0.5 < float(lines[i].split(",")[5].strip()) <= 0.75:
                            tp["mild"] += 1
                            tn_percents_gt["mild"].append(float(lines[i].split(",")[5].strip()))
                            tn_percents_pred["mild"].append(float(lines[i].split(",")[6].strip()))
                        elif 0.3 < float(lines[i].split(",")[5].strip()) <= 0.5:
                            tp["moderate"] += 1
                            tn_percents_gt["moderate"].append(float(lines[i].split(",")[5].strip()))
                            tn_percents_pred["moderate"].append(float(lines[i].split(",")[6].strip()))
                        else:
                            tp["severe"] += 1
                            tn_percents_gt["severe"].append(float(lines[i].split(",")[5].strip()))
                            tn_percents_pred["severe"].append(float(lines[i].split(",")[6].strip()))
                        tp["all"] += 1
                        tn_percents_gt["all"].append(float(lines[i].split(",")[5].strip()))
                        tn_percents_pred["all"].append(float(lines[i].split(",")[6].strip()))
                    elif lines[i].split(",")[7].strip().endswith("(FN)"):
                        if float(lines[i].split(",")[5].strip()) > 0.75:
                            fn["minimal"] += 1
                        elif 0.5 < float(lines[i].split(",")[5].strip()) <= 0.75:
                            fn["mild"] += 1
                        elif 0.3 < float(lines[i].split(",")[5].strip()) <= 0.5:
                            fn["moderate"] += 1
                        else:
                            fn["severe"] += 1
                        fn["all"] += 1
                    elif lines[i].split(",")[7].strip().endswith("(FP)"):
                        if float(lines[i].split(",")[6].strip()) > 0.75:
                            fp["minimal"] += 1
                        elif 0.5 < float(lines[i].split(",")[6].strip()) <= 0.75:
                            fp["mild"] += 1
                        elif 0.3 < float(lines[i].split(",")[6].strip()) <= 0.5:
                            fp["moderate"] += 1
                        else:
                            fp["severe"] += 1
                        fp["all"] += 1
                    else:
                        raise (ValueError("not a supported type"))

    print("done")
    if only_test:
        output_file_name = os.path.join(process_dir, "evaluation_on_tfpn_class_test.txt")
    else:
        output_file_name = os.path.join(process_dir, "evaluation_on_tfpn_class.txt")

    re_dict = []
    with open(output_file_name, "wt") as f:
        for c in ["all", "minimal", "mild", "moderate", "severe"]:
            armse = np.sqrt(np.mean([d ** 2 for d in np.array(tn_percents_gt[c]) - np.array(tn_percents_pred[c])]))
            rrmse = np.sqrt(np.mean([d ** 2 for d in
                                     (np.array(tn_percents_gt[c]) - np.array(tn_percents_pred[c])) / np.array(
                                         tn_percents_gt[c])]))
            tpr = tp[c] / (tp[c] + fn[c])
            ppv = tp[c] / (tp[c] + fp[c])
            f.write(f"\n------------------{c}-----------------\n")
            f.write(f"TP = {tp[c]}, FN = {fn[c]}, FP = {fp[c]}\n")
            f.write(
                f" R2 score = {r2_score(np.array(tn_percents_gt[c]), np.array(tn_percents_pred[c]), multioutput='variance_weighted')},\n"
                f" Pearson correlation: {pearsonr(np.array(tn_percents_gt[c]), np.array(tn_percents_pred[c]))[0]},\n"
                f" ARMSE = {armse},\n"
                f" RRMSE = {rrmse},\n"
                f" TPR = {tpr},\n"
                f" PPV = {ppv},\n")

            re_dict.append({"class": c, "armse": armse, "rrmse": rrmse, "TPR": tpr, "PPV": ppv})

    return re_dict


'''
python3 stenosis_detection2.py --root_path=sources/seg_all/xnet/all \
    --mean_dist_threshold=0.9 \
    --detection_radius=10 \
    --stenosis_threshold=0.9 \
    --detection_method=min_to_max \
    --save_dir=sources/stenosis_detection1.9/xnet \
    --report_only=false \
    --report_on_test=false

python3 stenosis_detection2.py --root_path=sources/seg_all/x/all \
    --mean_dist_threshold=0.9 \
    --detection_radius=10 \
    --stenosis_threshold=0.9 \
    --detection_method=min_to_max \
    --save_dir=sources/stenosis_detection1.9/x \
    --report_only=false \
    --report_on_test=false
    
python3 stenosis_detection2.py --root_path=sources/seg_all/u/all \
    --mean_dist_threshold=0.9 \
    --detection_radius=10 \
    --stenosis_threshold=0.9 \
    --detection_method=min_to_max \
    --save_dir=sources/stenosis_detection1.9/u \
    --report_only=false \
    --report_on_test=false

'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--root_path', default="./data_for_stenosis_detection/mimsunet", type=str)
    parser.add_argument('--root_path', default="./example_images", type=str)
    parser.add_argument('--mean_dist_threshold', type=float, default=0.9, help="minial mean of vessel diameters (this may be fixed)")
    parser.add_argument('--detection_radius', type=int, default=10, help="")
    parser.add_argument('--stenosis_threshold', type=float, default=0.9)
    parser.add_argument('--detection_method', type=str, default="local_min", choices=["local_min", "local_mean", "min_to_max"])
    parser.add_argument('--save_dir', type=str, default="./figure_example/save")
    parser.add_argument('--report_only', type=str2bool, default=False)
    parser.add_argument('--report_on_test', type=str2bool, default=False)
    parser.add_argument('--spacing_file_path', type=str, default='./spacing_file/spacing.txt')
    args = parser.parse_args()

    if args.report_only:
        generate_report_on_segment(args.save_dir)
        generate_report_on_tpfn_classified(args.save_dir, args.report_on_test)
    else:
        # python3 stenosis_detection.py --mean_dist_threshold=1.5 --detection_radius=5 --combine_gap=5 --detection_method=local_mean --save_dir=trail1
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        with open(os.path.join(args.save_dir, "config.txt"), "wt") as f:
            json.dump(vars(args), f, indent=4)

        eng = matlab.engine.start_matlab()
        patient_image_x_files = glob(args.root_path + "/*x.png")
        patient_image_x_files = sorted(patient_image_x_files)

        for patient_image_x_file in patient_image_x_files:
            patient_name = patient_image_x_file[
                           patient_image_x_file.rfind("/") + 1: patient_image_x_file.rfind("_x.png")]
            print(f"processing patient {patient_name}")

            original_file_path = "./{}/{}_x.png".format(args.root_path, patient_name)
            binary_file_path = "./{}/{}_y.png".format(args.root_path, patient_name)
            segment_file_path = "./{}/{}_ybin.png".format(args.root_path, patient_name)

            save_dir = "{}/{}".format(args.save_dir, patient_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            stenosis_evaluation(original_file_path, binary_file_path, segment_file_path, eng, save_dir,
                                args.detection_method, args.spacing_file_path, args.mean_dist_threshold, args.detection_radius,
                                args.stenosis_threshold)

        generate_report_on_segment(args.save_dir)
        generate_report_on_tpfn_classified(args.save_dir, args.report_on_test)