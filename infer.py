"""
UniCal Test Script - Depth and Force Map Generation

Simplified test script that generates:
- Depth maps (grayscale images)
- Force maps (heatmap visualizations)

No metric computation, focusing on fast inference and visualization.

Usage:
    python test.py --load_weights_folder weights_xx --data_root /path/to/data
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import open3d as o3d
from networks import DWResNetEncoder, DWDepthDecoder, DWForceDecoder, DWMaskDecoder


# =====================================================================
# Utility Functions
# =====================================================================

def create_output_dirs(base_dir: str, save_mask: bool = False, enable_open3d: bool = True) -> tuple:
    """
    Create output directories for depth and force maps.
    
    Args:
        base_dir: Base output directory
        save_mask: Whether to save mask predictions
        enable_open3d: Whether to enable Open3D visualization
    
    Returns:
        Tuple of (depth_npy_dir, o3d_dir, force_dir, mask_dir) paths
    """
    depth_npy_dir = os.path.join(base_dir, "depth_pred_npy")
    force_dir = os.path.join(base_dir, "force_map")
    
    os.makedirs(depth_npy_dir, exist_ok=True)
    os.makedirs(force_dir, exist_ok=True)
    
    o3d_dir = None
    if enable_open3d:
        o3d_dir = os.path.join(base_dir, "open3d_view")
        os.makedirs(o3d_dir, exist_ok=True)
    
    mask_dir = None
    if save_mask:
        mask_dir = os.path.join(base_dir, "mask_pred")
        os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  Depth (npy): {depth_npy_dir}")
    if enable_open3d:
        print(f"  Open3D views: {o3d_dir}")
    print(f"  Force maps: {force_dir}")
    if save_mask:
        print(f"  Mask: {mask_dir}")
    
    return depth_npy_dir, o3d_dir, force_dir, mask_dir


def load_model_weights(model, weight_path: str, model_name: str, device) -> bool:
    """
    Safely load model weights.
    
    Args:
        model: PyTorch model
        weight_path: Path to weight file
        model_name: Model name for logging
        device: torch device
    
    Returns:
        True if loaded successfully, False otherwise
    """
    if not os.path.exists(weight_path):
        print(f"⚠️  {model_name} weights not found: {weight_path}")
        return False
    
    state_dict = torch.load(weight_path, map_location=device)
    
    # Filter out incompatible keys
    model_keys = set(model.state_dict().keys())
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    model.load_state_dict(filtered_dict, strict=False)
    print(f"✅ Loaded {model_name}: {len(filtered_dict)} parameters")
    
    return True


def save_depth_npy(depth_tensor: torch.Tensor, save_path: str):
    """
    Save depth map as numpy array (.npy).
    
    This saves the raw depth prediction for later normalization.
    
    Args:
        depth_tensor: Depth prediction (H, W)
        save_path: Output file path (.npy)
    """
    # Convert to numpy
    depth_np = depth_tensor.detach().cpu().numpy()
    
    # Save as .npy
    np.save(save_path, depth_np)


def save_depth_png(depth_tensor: torch.Tensor, save_path: str, colormap: str = 'gray', depth_range: tuple = None):
    """
    Save depth map as 16-bit PNG visualization (no banding artifacts).

    Args:
        depth_tensor: Depth prediction (H, W)
        save_path: Output file path (.png)
        colormap: Matplotlib colormap name (default: 'gray' for 16-bit grayscale). 
                  Use 'gray' for 16-bit grayscale (65536 levels, no banding).
                  Other colormaps will use 8-bit RGB.
        depth_range: Optional (min, max) range for fixed normalization. If None, uses per-image min-max.
    """
    # Convert to numpy
    depth_np = depth_tensor.detach().cpu().numpy()

    # Normalization to [0, 1]
    if depth_range is not None:
        # Fixed-range normalization for cross-sensor consistency
        depth_min, depth_max = depth_range
        depth_norm = np.clip((depth_np - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
    else:
        # Per-image min-max normalization
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max - depth_min > 1e-8:
            depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_np)

    if colormap == 'gray':
        # Save as 16-bit grayscale (65536 levels, eliminates banding)
        depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        img = Image.fromarray(depth_uint16, mode='I;16')
        img.save(save_path)
    else:
        # Apply colormap (8-bit RGB)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        cmap = plt.get_cmap(colormap)
        colored = cmap(depth_norm)
        colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored_uint8, mode='RGB')
        img.save(save_path)


def save_force_map(force_tensor: torch.Tensor, save_path: str, colormap: str = 'jet'):
    """
    Save force map as heatmap PNG.
    
    Args:
        force_tensor: Force prediction (H, W)
        save_path: Output file path
        colormap: Matplotlib colormap name
    """
    # Convert to numpy and clip negative values
    force_np = force_tensor.detach().cpu().numpy()
    force_np = np.clip(force_np, 0, None)
    
    # Normalize
    vmax = force_np.max() + 1e-8
    force_norm = np.clip(force_np / vmax, 0, 1)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(force_norm)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save as RGB image
    img = Image.fromarray(colored_uint8, mode='RGB')
    img.save(save_path)


def save_mask_map(mask_tensor: torch.Tensor, save_path: str):
    """
    Save binary mask as black/white PNG.
    
    Args:
        mask_tensor: Binary mask prediction (H, W)
        save_path: Output file path
    """
    # Convert to numpy
    mask_np = mask_tensor.detach().cpu().numpy()
    
    # Convert to uint8 (0 or 255)
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    # Save as grayscale image
    img = Image.fromarray(mask_uint8, mode='L')
    img.save(save_path)
  


class Open3DVisualizer:
    """Open3D Visualizer for depth point clouds (from test_demo)"""
    def __init__(self, points, point_cloud_width=640, point_cloud_height=480, visible=True):
        self.window_open = False
        try:
            self.vis = o3d.visualization.Visualizer()
            # Create Open3D window
            if visible:
                self.vis.create_window(window_name='3D Point Cloud', width=960, height=480, left=1280, top=100)
            else:
                self.vis.create_window(visible=False)  # offscreen for saving only
            self.window_open = True

            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.vis.add_geometry(self.pcd)

            self.colors = np.zeros([points.shape[0], 3])

            self.ctr = self.vis.get_view_control()
            self.ctr.change_field_of_view(-25)
            self.ctr.convert_to_pinhole_camera_parameters()
             
            # Set zoom first
            self.ctr.set_zoom(0.4)

            # Set view direction (after zoom)
            self.ctr.set_front([0, 0.7, -0.9])
            self.ctr.set_up([0.0, -1.0, 0.0])
            self.ctr.set_lookat([
                point_cloud_width // 2,
                point_cloud_height // 2,
                0.0
            ])

            # Update renderer
            self.vis.update_renderer()

        except Exception as e:
            print(f"⚠️  Open3D initialization failed: {e}")
            self.window_open = False

    def update(self, points):
        """Update point cloud with new depth data"""
        if not self.window_open:
            return False
            
        try:
            points[:, 2] = np.clip(points[:, 2], 0, 50)
            
            z_values = points[:, 2]
            z_norm = z_values / 50.0
            z_norm = np.clip(z_norm, 0, 1)
            
            # Continuous colormap mapping (YlOrRd_r: low=red, high=yellow)
            colors = plt.get_cmap('YlOrRd_r')(z_norm)[:, :3]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd)
            
            if not self.vis.poll_events():
                self.window_open = False
                return False
                
            self.vis.update_renderer()
            return True
        except Exception as e:
            print(f"⚠️  Open3D update error: {e}")
            self.window_open = False
            return False
    
    def capture_image(self, filename):
        """Capture screenshot of the Open3D window"""
        if self.window_open:
            try:
                self.vis.capture_screen_image(filename, do_render=True)
                return True
            except Exception as e:
                print(f"⚠️  Error capturing Open3D screenshot: {e}")
                return False
        return False
    
    def close(self):
        """Close the visualizer window"""
        if self.window_open:
            try:
                self.vis.destroy_window()
                self.window_open = False
            except Exception as e:
                print(f"⚠️  Error closing Open3D window: {e}")
                pass


def depth_to_point_cloud(depth_np: np.ndarray, depth_range: tuple, scale: float = 50.0, downsample: int = 2, edge_margin: int = 30, mask_np: np.ndarray = None, sensor_type: str = "default"):
    """Convert depth (H, W) to (N, 3) point cloud in image coordinates.
    
    Uses baseline (95th percentile) normalization to preserve absolute depth relationships,
    matching the demo's approach for consistent 3D visualization.
    
    For 9dtact sensor, applies mask-based zone coloring:
    - Non-contact areas: yellow gradient [0.6, 1.0] (flat surface)
    - Contact areas: full colormap [0, 1.0] (red-orange-yellow)

    Args:
        depth_np: depth map (H, W) in physical units (e.g., [0, 10] mm)
        depth_range: (min, max) depth range in physical units
        scale: scaling factor for Z axis
        downsample: downsample factor for grid
        edge_margin: pixels to exclude from edges (default: 30)
        mask_np: optional contact mask (H, W) for 9dtact zone-based coloring
        sensor_type: sensor type ('9dtact' or 'default')
    """
    h, w = depth_np.shape
    
    # Create mesh grid
    X_grid = np.arange(0, w)
    Y_grid = np.arange(0, h)
    X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
    
    # Create edge mask to filter out unreliable edge regions
    valid_mask = np.ones((h, w), dtype=bool)
    valid_mask[:edge_margin, :] = False
    valid_mask[-edge_margin:, :] = False
    valid_mask[:, :edge_margin] = False
    valid_mask[:, -edge_margin:] = False
    
    # Downsample
    X_mesh_ds = X_mesh[::downsample, ::downsample]
    Y_mesh_ds = Y_mesh[::downsample, ::downsample]
    valid_mask_ds = valid_mask[::downsample, ::downsample]
    Z = depth_np[::downsample, ::downsample].astype(np.float32)

    # Use baseline-based normalization (matching demo's approach)
    # This preserves absolute depth relationships across images
    Z_valid = Z[valid_mask_ds]
    if len(Z_valid) > 0:
        # Use 95th percentile as baseline (most points are on the flat surface)
        baseline_depth = np.percentile(Z_valid, 95)
        depth_min_valid = Z_valid.min()
        
        if baseline_depth - depth_min_valid > 1e-8:
            # Normalize based on baseline, not min-max
            # This keeps shallow indentations different from deep indentations
            raw_norm = (Z - depth_min_valid) / (baseline_depth - depth_min_valid + 1e-8)
            
            # 9dtact-specific: mask-based zone coloring
            #if sensor_type == "gelsight" and mask_np is not None:
            # Ensure mask matches depth dimensions before downsampling
            if mask_np.shape != (h, w):
                # Resize mask to match depth dimensions using nearest interpolation
                mask_resized = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask_np
            
            # Downsample mask to match point cloud (after ensuring size match)
            mask_ds = mask_resized[::downsample, ::downsample].astype(np.float32)
            
            # Apply Gaussian blur for smooth transitions (using scipy if available)
            try:
                from scipy.ndimage import gaussian_filter
                mask_smooth = gaussian_filter(mask_ds, sigma=2.5)
            except ImportError:
                # Fallback: simple blur or no blur
                mask_smooth = mask_ds
            
            # Zone 1: Non-contact areas -> yellow gradient [0.6, 1.0]
            depth_norm_base = 0.4 + 0.6 * np.clip(raw_norm, 0, 1)
            
            # Zone 2: Contact areas -> full colormap [0, 1.0]
            depth_norm_press = np.clip(raw_norm, 0, 1)
            
            # Blend based on smooth mask
            Z_norm = (1 - mask_smooth) * depth_norm_base + mask_smooth * depth_norm_press
            
        else:
            Z_norm = np.ones_like(Z) * 0.8
    else:
        Z_norm = np.ones_like(Z) * 0.8

    Z_vis = Z_norm * scale
    
    # Only keep valid points (filter out edges)
    X_valid = X_mesh_ds[valid_mask_ds]
    Y_valid = Y_mesh_ds[valid_mask_ds]
    Z_valid = Z_vis[valid_mask_ds]
    
    points = np.stack([X_valid, Y_valid, Z_valid], axis=-1)
    return points



# =====================================================================
# Main Test Function
# =====================================================================

@torch.no_grad()
def test(opt):
    """
    Main test function for depth and force map generation.
    
    Args:
        opt: Command-line options
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"UniCal Test - Depth & Force Map Generation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Sensor type: {opt.sensor_type}")
    print(f"Weights: {opt.load_weights_folder}")
    print(f"Input: {opt.image_path}")
    print(f"Output: {opt.output_dir}")
    print(f"Input size: {opt.height}x{opt.width}")
    if opt.sensor_type == "9dtact":
        print(f"9dtact mask-based zone coloring: ENABLED")
    print(f"{'='*60}\n")
    
    # -------------------------------------------------------------------------
    # 1. Create output directories
    # -------------------------------------------------------------------------
    depth_npy_dir, o3d_dir, force_dir, mask_dir = create_output_dirs(
        opt.output_dir,
        save_mask=opt.save_mask,
        enable_open3d=opt.enable_open3d
    )
    
    # -------------------------------------------------------------------------
    # 2. Load models
    # -------------------------------------------------------------------------
    print("Loading models...")

    # Encoder for depth/force
    encoder = DWResNetEncoder(
        num_layers=18,
        num_input_images=1
    ).to(device)

    # Encoder for mask (use aligned weights)
    mask_encoder = DWResNetEncoder(
        num_layers=18,
        num_input_images=1
    ).to(device)

    # Mask decoder (for contact area prediction, aligned weights)
    mask_decoder = DWMaskDecoder(
        num_ch_enc=mask_encoder.num_ch_enc,
        scales=range(4)
    ).to(device)

    # Depth decoder
    depth_decoder = DWDepthDecoder(
        num_ch_enc=encoder.num_ch_enc,
        scales=range(4),
        num_output_channels=1,
        use_skips=True
    ).to(device)

    # Force decoder
    force_decoder = DWForceDecoder(
        num_ch_enc=encoder.num_ch_enc,
        scales=range(4),
        num_output_channels=1,
        use_skips=True
    ).to(device)

    # Load weights
    weights_folder = opt.load_weights_folder

    encoder_loaded = load_model_weights(
        encoder,
        os.path.join(weights_folder, "encoder.pth"),
        "Encoder",
        device
    )

    # Mask encoder/decoder use weights_align
    mask_weights_folder = "weights_align"

    mask_encoder_loaded = load_model_weights(
        mask_encoder,
        os.path.join(mask_weights_folder, "mask_encoder.pth"),
        "Mask Encoder (aligned)",
        device
    )

    mask_loaded = load_model_weights(
        mask_decoder,
        os.path.join(mask_weights_folder, "mask_decoder.pth"),
        "Mask Decoder (aligned)",
        device
    )

    depth_loaded = load_model_weights(
        depth_decoder,
        os.path.join(weights_folder, "depth.pth"),
        "Depth Decoder",
        device
    )

    force_loaded = load_model_weights(
        force_decoder,
        os.path.join(weights_folder, "force.pth"),
        "Force Decoder",
        device
    )

    if not (encoder_loaded and depth_loaded):
        raise RuntimeError("Failed to load encoder or depth decoder weights!")

    if not (mask_encoder_loaded and mask_loaded):
        print("\n⚠️  Mask encoder/decoder aligned weights not fully loaded. Mask quality may be affected.\n")

    if not force_loaded:
        print("⚠️  Force decoder weights not loaded. Force maps may not be accurate.")

    # Set to eval mode
    encoder.eval()
    mask_encoder.eval()
    mask_decoder.eval()
    depth_decoder.eval()
    force_decoder.eval()

    print(f"\n✅ Models loaded and set to eval mode\n")
    
    # -------------------------------------------------------------------------
    # 3. Load images
    # -------------------------------------------------------------------------
    print("Loading images...")
    
    input_path = Path(opt.image_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    
    # Determine if input is file or directory
    if input_path.is_file():
        # Single image file
        if input_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        image_files = [input_path]
        print(f"Processing single image: {input_path.name}")
    
    elif input_path.is_dir():
        # Directory of images
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(sorted(input_path.glob(f'*{ext}')))
            image_files.extend(sorted(input_path.glob(f'*{ext.upper()}')))
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
        
        if not image_files:
            raise ValueError(f"No images found in directory: {input_path}")
        
        print(f"Found {len(image_files)} images in directory")
    
    else:
        raise ValueError(f"Invalid path: {input_path} (not a file or directory)")
    
    print()
    
    # -------------------------------------------------------------------------
    # 4. Inference loop
    # -------------------------------------------------------------------------
    print("Starting inference...\n")
    
    # Initialize Open3D visualizer (default enabled)
    visualizer_3d = None
    if opt.enable_open3d:
        # Create initial point cloud for window
        X = np.arange(0, opt.width)
        Y = np.arange(0, opt.height)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros_like(X, dtype=np.float32)
        initial_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        
        try:
            visualizer_3d = Open3DVisualizer(initial_points, opt.width, opt.height, visible=True)
            if visualizer_3d.window_open:
                print(f"✅ Open3D window opened for real-time display")
            else:
                print("⚠️  Failed to open Open3D window, will save images only")
                visualizer_3d = None
        except Exception as e:
            print(f"⚠️  Failed to initialize Open3D: {e}")
            visualizer_3d = None

    sample_counter = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image 
            img_pil = Image.open(image_path).convert('RGB')

            # Convert to numpy (H, W, C)
            img_np = np.array(img_pil)

            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)


            # Forward pass - shared depth/force encoder
            features = encoder(img_tensor)

            # Forward pass - separate mask encoder
            mask_features = mask_encoder(img_tensor)

            # Mask prediction using aligned network
            mask_outputs = mask_decoder(mask_features)
            mask_pred = mask_outputs[("mask", 0)]
            mask_prob = torch.sigmoid(mask_pred)
            # Upsample mask to full resolution (height, width)
            mask_upsampled = F.interpolate(mask_prob, size=(opt.height, opt.width), mode="bilinear", align_corners=False)
            mask_binary = mask_upsampled > 0.5

            # Depth prediction
            depth_outputs = depth_decoder(features)
            depth_pred = depth_outputs[("depth", 0)]
            depth_pred_clamped = depth_pred.clamp(min=1e-3, max=1.0)
            
            # De-normalize depth from [0, 1] to real physical range [depth_min, depth_max] (in mm)
            # This matches the demo's approach: depth = model_output * (z_max - z_min) + z_min
            depth_denorm = depth_pred_clamped * (opt.depth_max - opt.depth_min) + opt.depth_min

            # Force prediction (still conditioned on mask)
            force_outputs = force_decoder(features, mask_binary.float())
            force_scale = 3
            force_key = ("force_map", force_scale)
            force_pred = force_outputs.get(force_key, None)


            # Generate filename (use original filename without extension)
            sample_name = image_path.stem

            # Save depth as numpy array (de-normalized physical depth)
            depth_npy_path = os.path.join(depth_npy_dir, f"{sample_name}.npy")
            save_depth_npy(depth_denorm[0, 0], depth_npy_path)

            # Open3D point cloud visualization and screenshot (default enabled)
            if opt.enable_open3d and o3d_dir is not None:
                # Use de-normalized depth for point cloud generation
                depth_np = depth_denorm[0, 0].detach().cpu().numpy()
                
                # Pass mask for 9dtact zone-based coloring
                mask_for_pc = mask_binary[0, 0].detach().cpu().numpy()
                
                points = depth_to_point_cloud(
                    depth_np, 
                    depth_range=(opt.depth_min, opt.depth_max),
                    scale=opt.pointcloud_scale, 
                    downsample=2,
                    mask_np=mask_for_pc,
                    sensor_type=opt.sensor_type
                )
                
                # Update existing visualizer or create temporary one for saving
                if visualizer_3d is not None:
                    # Update persistent window
                    visualizer_3d.update(points)
                    # Save screenshot
                    o3d_img_path = os.path.join(o3d_dir, f"{sample_name}_open3d.png")
                    visualizer_3d.capture_image(o3d_img_path)
                else:
                    # Create temporary offscreen visualizer just for saving image
                    try:
                        temp_vis = Open3DVisualizer(points, opt.width, opt.height, visible=False)
                        if temp_vis.window_open:
                            # Apply coloring
                            z_values = points[:, 2]
                            z_norm = z_values / 50.0
                            z_norm = np.clip(z_norm, 0, 1)
                            colors = plt.get_cmap('YlOrRd_r')(z_norm)[:, :3]
                            temp_vis.pcd.colors = o3d.utility.Vector3dVector(colors)
                            temp_vis.vis.update_geometry(temp_vis.pcd)
                            temp_vis.vis.poll_events()
                            temp_vis.vis.update_renderer()
                            
                            # Save screenshot
                            o3d_img_path = os.path.join(o3d_dir, f"{sample_name}_open3d.png")
                            temp_vis.capture_image(o3d_img_path)
                            temp_vis.close()
                    except Exception as e:
                        print(f"⚠️  Failed to save Open3D image for {sample_name}: {e}")

            # Save mask (if enabled)
            if opt.save_mask and mask_dir is not None:
                mask_save_path = os.path.join(mask_dir, f"{sample_name}_mask.png")
                save_mask_map(mask_binary[0, 0], mask_save_path)

            # Save force map
            if force_pred is not None:
                # Upsample force map to original image size (640x480)
                force_pred_upsampled = F.interpolate(
                    force_pred,  # Already [1, 1, H, W]
                    size=(opt.height, opt.width),
                    mode='bilinear',
                    align_corners=False
                )
                force_save_path = os.path.join(force_dir, f"{sample_name}_force.png")
                save_force_map(force_pred_upsampled[0, 0], force_save_path, colormap='jet')

            sample_counter += 1

        except Exception as e:
            print(f"\n⚠️  Error processing {image_path.name}: {e}")
            continue
    
    
    # Clean up Open3D visualizer
    if visualizer_3d is not None:
        visualizer_3d.close()
    
    print(f"\n{'='*60}")
    print(f"Generation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {sample_counter}")
 
    print(f"\nOutput saved to:")
    print(f"  Depth (npy):  {depth_npy_dir}")
    if opt.enable_open3d and o3d_dir is not None:
        print(f"  Open3D views: {o3d_dir}")
    print(f"  Force maps:   {force_dir}")
    if opt.save_mask and mask_dir is not None:
        print(f"  Mask maps:    {mask_dir}")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UniCal Test - Generate depth and force maps"
    )
    
    # Sensor type (primary parameter)
    parser.add_argument(
        "--sensor_type",
        type=str,
        required=True,
        choices=["9DTact", "GelSight", "Tac3D", "DM-Tac"],
        help="Sensor type (required). Determines default paths and visualization strategy."
    )
    
    # Data paths (optional, auto-set based on sensor_type)
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default=None,
        help="Path to saved weights folder (default: weights/{sensor_type})"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to input images (default: test_images/{sensor_type})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save predictions (default: output/{sensor_type})"
    )
    
    # Additional output settings
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="Save predicted contact masks"
    )
    
    # Performance settings
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Input image height (default: 480)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Input image width (default: 640)"
    )

    parser.add_argument(
        "--no_open3d",
        dest="enable_open3d",
        action="store_false",
        help="Disable Open3D point cloud rendering",
    )
    
    # Depth visualization settings
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.0,
        help="Minimum depth value for de-normalization and visualization (default: 0.0 mm)"
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=10.0,
        help="Maximum depth value for de-normalization and visualization (default: 10.0 mm)"
    )
    parser.add_argument(
        "--pointcloud_scale",
        type=float,
        default=50.0,
        help="Z-axis scaling factor for point cloud visualization (default: 50.0)"
    )

    opt = parser.parse_args()
    
    # Auto-set default paths based on sensor_type
    if opt.load_weights_folder is None:
        opt.load_weights_folder = f"weights/{opt.sensor_type}"
    if opt.image_path is None:
        opt.image_path = f"test_images/{opt.sensor_type}"
    if opt.output_dir is None:
        opt.output_dir = f"output/{opt.sensor_type}"
    
    # Run test
    test(opt)