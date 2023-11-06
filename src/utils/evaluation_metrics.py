import numpy as np
import os
import pandas as pd
import surface_distance


def calculate_surface_dsc(mask_pred, mask_gt):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, 3)

    return surface_dsc


def calculate_surface_dsc_2mm(mask_pred, mask_gt):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, 2)
    return surface_dsc

def calculate_surface_dscmm(mask_pred, mask_gt):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, 1)
    return surface_dsc

def calculate_hausoff_dist(mask_pred, mask_gt):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    surface_dsc = surface_distance.compute_robust_hausdorff(
            surface_distances, 100)
    return surface_dsc

def calculate_volume(voxel_array, voxel_size):
    voxel =  np.sum(voxel_array)*np.array(voxel_size)
    return voxel[0]*voxel[1]*voxel[2]