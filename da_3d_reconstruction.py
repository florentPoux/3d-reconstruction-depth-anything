# -*- coding: utf-8 -*-
"""
3D Reconstruction from Video with Depth-Anything-3
===============================================================

This tutorial shows how to:
1. Generate depth maps and camera poses from images using Depth-Anything-3
2. Create 3D point clouds and clean noisy geometry
3. Register multi-frame point clouds with ICP
4. Segment ground and cluster objects (unsupervised)
5. Generate voxel meshes
6. Export Gaussian Splatting PLY and GLB scenes
7. Export all reconstruction data for downstream tasks

Dependencies:
- torch, depth_anything_3
- numpy, open3d, scipy, matplotlib

Author: Florent Poux — 3D Geodata Academy
Website: https://learngeodata.eu
License: MIT
"""

#%% Initialization: Environment Setup

import time
import glob
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from depth_anything_3.api import DepthAnything3

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def setup_paths(data_folder="SAMPLE_SCENE"):
    """Create project paths for data, results, and models."""
    paths = {
        'data': f"data/{data_folder}",
        'results': f"results/{data_folder}",
        'masks': f"results/{data_folder}/masks"
    }
    os.makedirs(paths['results'], exist_ok=True)
    os.makedirs(paths['masks'], exist_ok=True)
    return paths

# ──────────────────────────────────────────────
# SET YOUR SCENE NAME HERE
# ──────────────────────────────────────────────
SCENE = "MY_SCENE"
paths = setup_paths(SCENE)
print(f"Project paths created: {paths}")

#%% Initialization: Visualization Functions

def visualize_depth_and_confidence(images, depths, confidences, sample_idx=0):
    """Show RGB image, depth map, and confidence map side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(images[sample_idx])
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    axes[1].imshow(depths[sample_idx], cmap='turbo')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')

    axes[2].imshow(confidences[sample_idx], cmap='viridis')
    axes[2].set_title('Confidence Map')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_point_cloud_open3d(points, colors=None, window_name="Point Cloud"):
    """Display 3D point cloud with Open3D viewer."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=window_name, width=1920, height=1080)

#%% Step 1: Load DA3 Model

def load_da3_model(model_name="depth-anything/DA3NESTED-GIANT-LARGE"):
    """Initialize Depth-Anything-3 model on available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)

    return model, device

model, device = load_da3_model()
print("DA3 model loaded successfully")

#%% Step 2: Load Images from Folder

def load_images_from_folder(data_path, extensions=['*.jpg', '*.png', '*.jpeg']):
    """Scan folder and load all images with supported extensions."""
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(glob.glob(os.path.join(data_path, ext))))

    print(f"Found {len(image_files)} images in {data_path}")
    return image_files

image_files = load_images_from_folder(paths['data'])

#%% Step 3: Run DA3 Inference for Depth and Poses

def run_da3_inference(model, image_files, process_res_method="upper_bound_resize"):
    """Run Depth-Anything-3 to get depth maps, camera poses, and intrinsics."""
    prediction = model.inference(
        image=image_files,
        infer_gs=True,
        process_res_method=process_res_method
    )

    print(f"Depth maps shape: {prediction.depth.shape}")
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"Confidence shape: {prediction.conf.shape}")

    return prediction

prediction = run_da3_inference(model, image_files)
visualize_depth_and_confidence(
    prediction.processed_images,
    prediction.depth,
    prediction.conf,
    sample_idx=0
)

#%% Step 4: Generate 3D Point Cloud from Depth Maps

def depth_to_point_cloud(depth_map, rgb_image, intrinsics, extrinsics, conf_map=None, conf_thresh=0.5):
    """Back-project depth map to 3D points using camera parameters."""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filter by confidence if provided
    if conf_map is not None:
        valid_mask = conf_map > conf_thresh
        u, v, depth_map, rgb_image = u[valid_mask], v[valid_mask], depth_map[valid_mask], rgb_image[valid_mask]
    else:
        u, v, depth_map = u.flatten(), v.flatten(), depth_map.flatten()
        rgb_image = rgb_image.reshape(-1, 3)

    # Back-project to camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    points_cam = np.stack([x, y, z], axis=-1)

    # Transform to world coordinates using extrinsics (w2c format)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (points_cam - t) @ R  # Inverse transform

    colors = rgb_image.astype(np.float32) / 255.0

    return points_world, colors

i = 1
pcd_single, pcd_single_colors = depth_to_point_cloud(prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            0.5)

visualize_point_cloud_open3d(pcd_single, pcd_single_colors, window_name="Single Frame Point Cloud")

#%% Step 4b: Merge All Frames

def merge_point_clouds(prediction, conf_thresh=0.5):
    """Combine all frames into single point cloud, also returning per-frame clouds."""
    all_points = []
    all_colors = []

    n_frames = len(prediction.depth)

    for i in range(n_frames):
        points, colors = depth_to_point_cloud(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            conf_thresh
        )
        all_points.append(points)
        all_colors.append(colors)

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    print(f"Merged point cloud: {len(merged_points)} points")
    return merged_points, merged_colors, all_points, all_colors

points_3d, colors_3d, per_frame_points, per_frame_colors = merge_point_clouds(prediction, conf_thresh=0.4)

#%% Step 5: Cleaning the 3D Geometry

def clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=2.0):
    """Cleans a point cloud using SOR via Scipy cKDTree."""

    # 1. Build KD-Tree
    tree = cKDTree(points_3d)

    # 2. Query neighbors
    distances, _ = tree.query(points_3d, k=nb_neighbors + 1, workers=-1)

    # Exclude the first column (distance to self, which is 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # 3. Calculate statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    # 4. Generate Mask
    distance_threshold = global_mean + (std_ratio * global_std)
    mask = mean_distances < distance_threshold

    return points_3d[mask], colors_3d[mask]

start = time.time()
clean_pts_sci, clean_cols_sci = clean_point_cloud_scipy(points_3d, colors_3d, nb_neighbors=20, std_ratio=1)
end = time.time()
print(f"\n[Scipy] Cleaned shape: {clean_pts_sci.shape}")
print(f"[Scipy] Time taken: {end - start:.4f} seconds")

visualize_point_cloud_open3d(clean_pts_sci, clean_cols_sci, window_name="Full Scene Point Cloud")

#%% Step 6: Interactive ROI Selection

def interactive_crop(points, colors=None):
    """Interactively crop the point cloud using Open3D's crop-box mode.

    Workflow:
      1. Press 'K' to lock the viewpoint
      2. Click-drag a rectangle over the region of interest
      3. Press 'C' to confirm the crop
      4. Press 'Q' to close the window

    Returns (roi_min, roi_max) or (None, None) if no crop was made.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    print("=== Interactive ROI Selection (Crop Box) ===")
    print("  1. Press 'K' to lock the viewpoint")
    print("  2. Click-drag a rectangle to select the region")
    print("  3. Press 'C' to crop")
    print("  4. Press 'Q' to close")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Crop ROI (K > drag > C > Q)")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    cropped = vis.get_cropped_geometry()
    if cropped is None or len(cropped.points) == 0:
        print("No crop performed -- using full cloud")
        return None, None

    cropped_pts = np.asarray(cropped.points)
    roi_min = cropped_pts.min(axis=0)
    roi_max = cropped_pts.max(axis=0)
    print(f"ROI min: {roi_min}")
    print(f"ROI max: {roi_max}")

    mask = np.all((points >= roi_min) & (points <= roi_max), axis=1)
    print(f"Points inside ROI: {mask.sum()} / {len(points)}")
    visualize_point_cloud_open3d(
        points[mask], colors[mask] if colors is not None else None,
        window_name="ROI Preview",
    )
    return roi_min, roi_max

roi_min, roi_max = interactive_crop(clean_pts_sci, clean_cols_sci)

#%% Step 7: Two-Frame Registration Preview

def extract_center_zone_points(depth_map, rgb_image, intrinsics, extrinsics,
                               conf_map, conf_thresh=0.7, center_ratio=0.6):
    """Back-project only high-confidence points from the image center."""
    h, w = depth_map.shape
    margin_h = int(h * (1 - center_ratio) / 2)
    margin_w = int(w * (1 - center_ratio) / 2)
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[margin_h:h - margin_h, margin_w:w - margin_w] = True
    valid = center_mask & (conf_map > conf_thresh)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_v, v_v, d = u[valid], v[valid], depth_map[valid]

    x = (u_v - cx) * d / fx
    y = (v_v - cy) * d / fy

    pts_cam = np.stack([x, y, d], axis=-1)
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    pts_world = (pts_cam - t) @ R
    cols = rgb_image[valid].astype(np.float32) / 255.0
    return pts_world, cols

def make_registration_pcd(points, voxel_size):
    """Build a downsampled Open3D PointCloud with normals for ICP."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd

def icp_refine(source, target, voxel_size):
    """Point-to-plane ICP from identity — small correction only."""
    return o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel_size * 0.5,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

def extract_registration_cloud(prediction, frame_idx, conf_thresh, center_ratio,
                                roi_min=None, roi_max=None):
    """Center-zone points for one frame, optionally cropped to ROI."""
    pts, _ = extract_center_zone_points(
        prediction.depth[frame_idx], prediction.processed_images[frame_idx],
        prediction.intrinsics[frame_idx], prediction.extrinsics[frame_idx],
        prediction.conf[frame_idx], conf_thresh, center_ratio,
    )
    if roi_min is not None and roi_max is not None:
        pts = pts[np.all((pts >= roi_min) & (pts <= roi_max), axis=1)]
    return pts

def preview_two_frame_registration(prediction, frame_a=0, frame_b=1,
                                    conf_thresh=0.7, center_ratio=0.6,
                                    roi_min=None, roi_max=None):
    """Visualize two-frame overlap before/after ICP and print deviation stats."""
    pts_a = extract_registration_cloud(prediction, frame_a, conf_thresh, center_ratio, roi_min, roi_max)
    pts_b = extract_registration_cloud(prediction, frame_b, conf_thresh, center_ratio, roi_min, roi_max)
    print(f"Two-frame preview: frame {frame_a} ({len(pts_a)} pts) vs frame {frame_b} ({len(pts_b)} pts)")

    cols_a = np.broadcast_to([1.0, 0.3, 0.3], (len(pts_a), 3)).copy()
    cols_b = np.broadcast_to([0.3, 0.3, 1.0], (len(pts_b), 3)).copy()

    combined = np.vstack([pts_a, pts_b])
    reg_voxel = float(combined.ptp(axis=0).max()) / 200.0

    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(combined)
    pcd_before.colors = o3d.utility.Vector3dVector(np.vstack([cols_a, cols_b]))

    source_pcd = make_registration_pcd(pts_b, reg_voxel)
    target_pcd = make_registration_pcd(pts_a, reg_voxel)
    icp_result = icp_refine(source_pcd, target_pcd, reg_voxel)

    T = icp_result.transformation
    pts_b_h = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    pts_b_aligned = (T @ pts_b_h.T).T[:, :3]

    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(np.vstack([pts_a, pts_b_aligned]))
    pcd_after.colors = o3d.utility.Vector3dVector(np.vstack([cols_a, cols_b]))

    tree_a = cKDTree(pts_a)
    d_before, _ = tree_a.query(pts_b)
    d_after, _ = tree_a.query(pts_b_aligned)
    shift = np.linalg.norm(T[:3, 3])

    print(f"  Before — mean: {d_before.mean():.4f}, median: {np.median(d_before):.4f}")
    print(f"  After  — mean: {d_after.mean():.4f}, median: {np.median(d_after):.4f}")
    print(f"  ICP fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.6f}, shift: {shift:.4f}")

    o3d.visualization.draw_geometries([pcd_before],
        window_name=f"Before ICP (Red=frame {frame_a}, Blue=frame {frame_b})")
    o3d.visualization.draw_geometries([pcd_after],
        window_name=f"After ICP (Red=frame {frame_a}, Blue=frame {frame_b})")
    return T

if len(per_frame_points) >= 2:
    preview_T = preview_two_frame_registration(
        prediction, conf_thresh=0.9, center_ratio=0.5,
        roi_min=roi_min, roi_max=roi_max,
    )

#%% Step 8: Full Multi-Frame Registration

def register_frames(prediction, per_frame_points, per_frame_colors,
                    conf_thresh=0.7, center_ratio=0.6, max_shift_pct=0.05,
                    roi_min=None, roi_max=None):
    """Refine per-frame alignment with ICP on center-zone high-confidence points.

    If roi_min/roi_max are provided, ICP runs only on points inside the ROI.
    The resulting transform is still applied to ALL points in each frame.
    """
    n_frames = len(per_frame_points)
    if n_frames < 2:
        return np.vstack(per_frame_points), np.vstack(per_frame_colors)

    all_pts = np.vstack(per_frame_points)
    bbox_extent = all_pts.max(axis=0) - all_pts.min(axis=0)
    reg_voxel = float(np.max(bbox_extent)) / 200.0
    max_translation = float(np.linalg.norm(bbox_extent)) * max_shift_pct

    print(f"Registration voxel size: {reg_voxel:.4f}")
    print(f"Max allowed translation: {max_translation:.4f}")
    if roi_min is not None:
        print(f"ROI crop: {roi_min} -> {roi_max}")

    target_pts = extract_registration_cloud(prediction, 0, conf_thresh, center_ratio, roi_min, roi_max)
    target_pcd = make_registration_pcd(target_pts, reg_voxel)

    registered_points = [per_frame_points[0]]
    registered_colors = [per_frame_colors[0]]

    for i in range(1, n_frames):
        source_pts = extract_registration_cloud(prediction, i, conf_thresh, center_ratio, roi_min, roi_max)
        source_pcd = make_registration_pcd(source_pts, reg_voxel)

        icp_result = icp_refine(source_pcd, target_pcd, reg_voxel)

        T = icp_result.transformation
        shift = np.linalg.norm(T[:3, 3])

        if shift > max_translation:
            print(f"  Frame {i}: shift {shift:.4f} exceeds limit — kept identity")
            T = np.eye(4)
        else:
            print(f"  Frame {i} -> 0 | fitness: {icp_result.fitness:.4f} | "
                  f"RMSE: {icp_result.inlier_rmse:.6f} | shift: {shift:.4f}")

        pts_h = np.hstack([per_frame_points[i], np.ones((len(per_frame_points[i]), 1))])
        registered_points.append((T @ pts_h.T).T[:, :3])
        registered_colors.append(per_frame_colors[i])

    merged_points = np.vstack(registered_points)
    merged_colors = np.vstack(registered_colors)
    print(f"Registered point cloud: {len(merged_points)} points")
    return merged_points, merged_colors

start = time.time()
reg_points, reg_colors = register_frames(
    prediction, per_frame_points, per_frame_colors,
    conf_thresh=0.9, center_ratio=0.5, max_shift_pct=0.05,
    roi_min=roi_min, roi_max=roi_max,
)
end = time.time()
print(f"Registration time: {end - start:.4f} seconds")

clean_pts_sci, clean_cols_sci = clean_point_cloud_scipy(reg_points, reg_colors, nb_neighbors=20, std_ratio=1)
print(f"Post-registration cleaned shape: {clean_pts_sci.shape}")
visualize_point_cloud_open3d(clean_pts_sci, clean_cols_sci, window_name="Registered + Cleaned")

#%% Step 9: Multi-Plane Segmentation (NumPy RANSAC + SVD)

def _fit_plane_numpy(points, distance_thresh, n_iter=1000, batch=256, rng=None):
    """Vectorized RANSAC plane fit + least-squares SVD refinement."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(points)
    if n < 3:
        return None, None, np.zeros(n, dtype=bool)

    # --- vectorized candidate generation ---
    idx = rng.randint(0, n, size=(n_iter, 3))
    normals = np.cross(points[idx[:, 1]] - points[idx[:, 0]],
                       points[idx[:, 2]] - points[idx[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    ok = (norms > 1e-10).ravel()
    normals = normals[ok] / norms[ok]
    d = -(normals * points[idx[ok, 0]]).sum(axis=1)
    K = len(normals)

    # --- batched scoring (keeps peak RAM manageable) ---
    best_count, best_k = 0, 0
    for b in range(0, K, batch):
        nb, db = normals[b:b + batch], d[b:b + batch]
        counts = (np.abs(points @ nb.T + db) < distance_thresh).sum(axis=0)
        loc = int(counts.argmax())
        if counts[loc] > best_count:
            best_count, best_k = int(counts[loc]), b + loc

    # --- SVD refinement on inliers ---
    mask = np.abs(points @ normals[best_k] + d[best_k]) < distance_thresh
    centroid = points[mask].mean(axis=0)
    _, _, vh = np.linalg.svd(points[mask] - centroid, full_matrices=False)
    normal_r = vh[2]
    d_r = -normal_r @ centroid
    mask = np.abs(points @ normal_r + d_r) < distance_thresh

    return normal_r, d_r, mask


def segment_planes(points, n_planes=3, distance_thresh=0.05,
                   n_iterations=2000, min_plane_points=100):
    """Find up to n_planes via iterative NumPy RANSAC + SVD refinement."""
    n = len(points)
    plane_labels = np.zeros(n, dtype=np.int32)
    plane_models = []
    remaining = np.ones(n, dtype=bool)
    rng = np.random.RandomState(42)

    for i in range(n_planes):
        rem_idx = np.where(remaining)[0]
        if len(rem_idx) < min_plane_points:
            print(f"  Plane {i+1}: too few remaining points "
                  f"({len(rem_idx)}), stopping")
            break

        normal, d_val, local_mask = _fit_plane_numpy(
            points[rem_idx], distance_thresh, n_iterations, rng=rng)

        if normal is None or local_mask.sum() < min_plane_points:
            print(f"  Plane {i+1}: insufficient inliers, stopping")
            break

        hits = rem_idx[local_mask]
        plane_labels[hits] = i + 1
        remaining[hits] = False
        plane_models.append(np.array([*normal, d_val]))

        print(f"  Plane {i+1}: {normal[0]:.3f}x + {normal[1]:.3f}y + "
              f"{normal[2]:.3f}z + {d_val:.3f} = 0  "
              f"({len(hits)} pts, {100 * len(hits) / n:.1f}%)")

    plane_mask = plane_labels > 0
    print(f"Total plane points: {plane_mask.sum()} / {n} "
          f"({100 * plane_mask.mean():.1f}%)")
    return plane_mask, plane_labels, plane_models

start = time.time()
ground_mask, plane_labels, plane_models = segment_planes(
    clean_pts_sci, n_planes=2, distance_thresh=0.01,
    n_iterations=100, min_plane_points=100)
end = time.time()
print(f"Plane segmentation time: {end - start:.4f} seconds")

# Visualize: each plane gets an earth-tone color
_earth = np.array([[0.55, 0.35, 0.17],   # brown
                    [0.65, 0.50, 0.30],   # tan
                    [0.45, 0.40, 0.25]])  # olive-brown
plane_vis_colors = clean_cols_sci.copy()
for pid in range(1, plane_labels.max() + 1):
    plane_vis_colors[plane_labels == pid] = _earth[(pid - 1) % len(_earth)]
visualize_point_cloud_open3d(clean_pts_sci, plane_vis_colors,
                              "Detected Planes (earth tones) vs Objects")

#%% Step 10: Object Clustering (Vectorized Euclidean Clustering)

def cluster_objects(points, voxel_size=0.15, min_points=10):
    """Vectorized Euclidean clustering via voxel connected components."""
    bbox_min = points.min(axis=0)
    ijk = np.floor((points - bbox_min) / voxel_size).astype(np.int64)
    dims = ijk.max(axis=0) + 1

    # Encode 3D grid indices to sorted 1D keys
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    keys = (ijk * strides).sum(axis=1)

    unique_keys, pt_to_vox, vox_counts = np.unique(
        keys, return_inverse=True, return_counts=True)
    n_vox = len(unique_keys)

    # Decode voxel keys back to ijk
    vox_ijk = np.column_stack([
        unique_keys // strides[0],
        (unique_keys % strides[0]) // strides[1],
        unique_keys % strides[1],
    ])

    # 26-connected neighbor offsets
    offsets = np.array(
        [[di, dj, dk]
         for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
         if (di, dj, dk) != (0, 0, 0)], dtype=np.int64)

    # Build sparse adjacency via vectorized searchsorted
    rows_list, cols_list = [], []
    for off in offsets:
        nb = vox_ijk + off
        valid = np.all((nb >= 0) & (nb < dims), axis=1)
        if not valid.any():
            continue
        nb_keys = (nb[valid] * strides).sum(axis=1)
        pos = np.searchsorted(unique_keys, nb_keys)
        pos = np.clip(pos, 0, n_vox - 1)
        found = unique_keys[pos] == nb_keys
        src = np.where(valid)[0][found]
        rows_list.append(src)
        cols_list.append(pos[found])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    adj = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_vox, n_vox))

    # Connected components on the voxel graph
    _, vox_labels = connected_components(adj, directed=False)
    pt_labels = vox_labels[pt_to_vox]

    # Filter small clusters, sort by descending size
    cl_ids, cl_counts = np.unique(pt_labels, return_counts=True)
    big = cl_counts >= min_points
    if not big.any():
        return np.zeros(len(points), dtype=np.int32)

    valid_ids = cl_ids[big]
    valid_counts = cl_counts[big]
    order = np.argsort(-valid_counts)

    remap = np.zeros(vox_labels.max() + 1, dtype=np.int32)
    for new_id, idx in enumerate(order, 1):
        remap[valid_ids[idx]] = new_id

    labels = remap[pt_labels]
    n_clusters = len(order)
    n_noise = (labels == 0).sum()
    print(f"Clusters: {n_clusters}, Noise points: {n_noise} "
          f"({100 * n_noise / len(points):.1f}%)")
    return labels

non_ground_pts = clean_pts_sci[~ground_mask]
non_ground_cols = clean_cols_sci[~ground_mask]

start = time.time()
cluster_labels = cluster_objects(non_ground_pts, voxel_size=0.005, min_points=10)
end = time.time()
print(f"Clustering time: {end - start:.4f} seconds")

n_cl = max(cluster_labels.max() + 1, 2)
rng_cl = np.random.RandomState(42)
cl_cmap = rng_cl.rand(n_cl, 3) * 0.7 + 0.3
cl_cmap[0] = [0.3, 0.3, 0.3]
visualize_point_cloud_open3d(non_ground_pts, cl_cmap[cluster_labels],
                              "Object Clusters")

#%% Step 11: Label Merge + Boundary Refinement

def merge_seg_labels(plane_labels, cluster_labels, n_points):
    """Combine plane labels and cluster labels into a unified array.

    Planes keep their IDs (1..n_planes).  Cluster IDs are offset by
    n_planes so there is no collision.  Label 0 = unlabeled / noise.
    """
    n_planes = int(plane_labels.max())
    labels = plane_labels.copy()
    non_plane = plane_labels == 0
    offset_clusters = cluster_labels.copy()
    offset_clusters[offset_clusters > 0] += n_planes
    labels[non_plane] = offset_clusters
    return labels

def refine_labels_knn(points, labels, k=15):
    """Smooth noisy label boundaries via KNN majority vote (vectorized)."""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1, workers=-1)
    neighbor_labels = labels[indices[:, 1:]]

    n_classes = labels.max() + 1
    offsets = (np.arange(len(labels), dtype=np.int64) * n_classes)
    flat = (neighbor_labels + offsets[:, np.newaxis]).ravel()
    votes = np.bincount(flat, minlength=len(labels) * n_classes)
    refined = votes.reshape(len(labels), n_classes).argmax(axis=1).astype(np.int32)

    changed = (refined != labels).sum()
    print(f"Boundary refinement: {changed} labels changed ({100 * changed / len(labels):.1f}%)")
    return refined

def make_seg_colormap(labels, plane_labels):
    """Planes = earth tones, clusters = distinct colors, noise = gray."""
    n_planes = int(plane_labels.max())
    n_labels = max(labels.max() + 1, 2)
    rng = np.random.RandomState(42)
    cmap = rng.rand(n_labels, 3) * 0.7 + 0.3
    cmap[0] = [0.3, 0.3, 0.3]  # noise
    earth_tones = [[0.55, 0.35, 0.17],   # brown
                   [0.65, 0.50, 0.30],   # tan
                   [0.45, 0.40, 0.25]]   # olive-brown
    for i in range(min(n_planes, len(earth_tones))):
        cmap[i + 1] = earth_tones[i]
    return cmap[labels]

n_planes_found = int(plane_labels.max())
seg_labels = merge_seg_labels(plane_labels, cluster_labels, len(clean_pts_sci))
print(f"Merged labels: {n_planes_found} planes + "
      f"{seg_labels.max() - n_planes_found} clusters")
visualize_point_cloud_open3d(clean_pts_sci, make_seg_colormap(seg_labels, plane_labels),
                              "Before Refinement")

start = time.time()
seg_labels = refine_labels_knn(clean_pts_sci, seg_labels, k=15)
end = time.time()
print(f"Refinement time: {end - start:.4f} seconds")
visualize_point_cloud_open3d(clean_pts_sci, make_seg_colormap(seg_labels, plane_labels),
                              "After Refinement")

#%% Step 12: Voxel Mesh Generation

def compute_auto_voxel_size(points, target_voxels=200_000):
    """Compute voxel size from bounding box volume and target voxel budget."""
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    volume = np.prod(bbox_max - bbox_min)
    voxel_size = (volume / target_voxels) ** (1 / 3)
    return voxel_size

def voxelize_point_cloud(points, colors, voxel_size=None, labels=None, target_voxels=200_000):
    """Assign points to a structured voxel grid and compute per-voxel average color.

    Voxel centers are snapped to a regular grid: bbox_min + (idx + 0.5) * voxel_size,
    guaranteeing uniform, non-overlapping cubes.
    """
    if voxel_size is None:
        voxel_size = compute_auto_voxel_size(points, target_voxels)

    bbox_min = points.min(axis=0)
    voxel_indices = np.floor((points - bbox_min) / voxel_size).astype(int)

    # Find unique occupied grid cells
    unique_indices, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)

    # Structured grid centers (no overlap possible)
    voxel_centers = bbox_min + (unique_indices + 0.5) * voxel_size

    n_voxels = len(unique_indices)

    # Compute per-voxel average color
    voxel_colors = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(voxel_colors, inverse, colors)
    voxel_colors /= counts[:, np.newaxis]
    voxel_colors = np.clip(voxel_colors, 0, 1)

    # Compute per-voxel majority label if provided
    voxel_labels = None
    if labels is not None:
        n_classes = labels.max() + 1
        label_counts = np.zeros((n_voxels, n_classes), dtype=int)
        np.add.at(label_counts, inverse, np.eye(n_classes, dtype=int)[labels])
        voxel_labels = label_counts.argmax(axis=1)

    return voxel_centers, voxel_colors, voxel_size, voxel_labels

def create_voxel_cube_mesh(voxel_centers, voxel_colors, voxel_size):
    """Build a triangle mesh of cubes (one per voxel) — fully vectorized, no Python loops."""
    # Unit cube: 8 vertices centered at origin
    unit_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64) - 0.5  # center at origin

    # 12 triangles (2 per face), winding order for outward normals
    unit_tris = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ], dtype=np.int32)

    n = len(voxel_centers)

    # Broadcast vertices: (N, 8, 3)
    all_vertices = (unit_verts * voxel_size) + voxel_centers[:, np.newaxis, :]
    all_vertices = all_vertices.reshape(-1, 3)

    # Broadcast triangles with per-voxel vertex offset
    offsets = (np.arange(n) * 8)[:, np.newaxis, np.newaxis]
    all_triangles = (unit_tris + offsets).reshape(-1, 3)

    # Per-vertex colors (each cube's 8 vertices share the voxel color)
    all_colors = np.repeat(voxel_colors, 8, axis=0)

    # Build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
    mesh.compute_vertex_normals()

    return mesh

# Auto voxel size, create mesh from cleaned point cloud (with segmentation labels)
voxel_centers, voxel_colors, auto_voxel_size, voxel_labels = voxelize_point_cloud(
    clean_pts_sci, clean_cols_sci, 0.01, labels=seg_labels)

print(f"Auto voxel size: {auto_voxel_size:.4f}, Voxels: {len(voxel_centers)}")

# Voxel mesh with RGB colors
voxel_mesh = create_voxel_cube_mesh(voxel_centers, voxel_colors, auto_voxel_size)
o3d.visualization.draw_geometries([voxel_mesh], window_name="Voxel Mesh (RGB)")

# Voxel mesh with segmentation-label colors
n_vlabels = max(voxel_labels.max() + 1, 2)
rng_v = np.random.RandomState(42)
voxel_label_cmap = rng_v.rand(n_vlabels, 3) * 0.7 + 0.3
voxel_label_cmap[0] = [0.3, 0.3, 0.3]  # noise
_vox_earth = [[0.55, 0.35, 0.17], [0.65, 0.50, 0.30], [0.45, 0.40, 0.25]]
for _pi in range(min(n_planes_found, len(_vox_earth))):
    voxel_label_cmap[_pi + 1] = _vox_earth[_pi]
voxel_label_colors = voxel_label_cmap[voxel_labels]
voxel_mesh_seg = create_voxel_cube_mesh(voxel_centers, voxel_label_colors, auto_voxel_size)
o3d.visualization.draw_geometries([voxel_mesh_seg], window_name="Voxel Mesh (Segmentation Labels)")


#%% Step 13: DA3 Gaussian Splatting & GLB Export

from depth_anything_3.utils.export.gs import export_to_gs_ply
from depth_anything_3.utils.export.glb import export_to_glb

export_dir = paths['results']

# Gaussian Splatting PLY — one splat cloud per view
export_to_gs_ply(prediction, export_dir)
print(f"GS PLY exported to {export_dir}/gs_ply/")

# GLB scene — point cloud with camera frustums, viewable in any 3D viewer/browser
glb_path = export_to_glb(prediction, export_dir)
print(f"GLB scene exported: {glb_path}")

# Optional: render a Gaussian Splatting fly-through video
# from depth_anything_3.utils.export.gs import export_to_gs_video
# export_to_gs_video(prediction, export_dir, trj_mode="extend", video_quality="high")

#%% Step 14: Export Reconstruction Data

def save_reconstruction_ply(points, colors, filepath, seg_labels=None, ground_mask=None):
    """Export point cloud to binary PLY with all scalar fields (vectorized write)."""
    n = len(points)
    props = [
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
    ]
    dtypes = [('x','f4'),('y','f4'),('z','f4'),('r','u1'),('g','u1'),('b','u1')]

    if seg_labels is not None:
        props.append("property int seg_label")
        dtypes.append(('seg_label', 'i4'))
    if ground_mask is not None:
        props.append("property uchar is_ground")
        dtypes.append(('is_ground', 'u1'))

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "\n".join(props) + "\nend_header\n"
    )

    arr = np.empty(n, dtype=dtypes)
    arr['x'], arr['y'], arr['z'] = points[:, 0], points[:, 1], points[:, 2]
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    arr['r'], arr['g'], arr['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if seg_labels is not None:
        arr['seg_label'] = seg_labels
    if ground_mask is not None:
        arr['is_ground'] = ground_mask.astype(np.uint8)

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)
    print(f"Saved {n} points to {filepath}")

# 1. Point cloud PLY with all scalar fields
ply_path = os.path.join(paths['results'], "reconstruction.ply")
save_reconstruction_ply(clean_pts_sci, clean_cols_sci, ply_path,
                        seg_labels=seg_labels, ground_mask=ground_mask)

# 2. Voxel meshes
voxel_rgb_path = os.path.join(paths['results'], "voxel_mesh_rgb.ply")
o3d.io.write_triangle_mesh(voxel_rgb_path, voxel_mesh)
print(f"Voxel mesh (RGB) saved to {voxel_rgb_path}")

voxel_seg_path = os.path.join(paths['results'], "voxel_mesh_seg.ply")
o3d.io.write_triangle_mesh(voxel_seg_path, voxel_mesh_seg)
print(f"Voxel mesh (seg) saved to {voxel_seg_path}")

# 3. All arrays for downstream workflows
npz_path = os.path.join(paths['results'], "reconstruction_data.npz")
np.savez_compressed(
    npz_path,
    points_3d=points_3d,
    colors_3d=colors_3d,
    depth=prediction.depth,
    conf=prediction.conf,
    intrinsics=prediction.intrinsics,
    extrinsics=prediction.extrinsics,
    processed_images=prediction.processed_images,
)
print(f"Reconstruction data saved to {npz_path}")
print(f"\nExport complete!")
print(f"Learn more at https://learngeodata.eu")
