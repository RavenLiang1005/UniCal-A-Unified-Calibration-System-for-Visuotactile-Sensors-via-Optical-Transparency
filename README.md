# UniCal-A-Unified-Calibration-System-for-Visuotactile-Sensors-via-Optical-Transparency

Test demo of UniCal for generating depth and force maps from visuotactile sensor images.

## Environment Setup

Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate monocular_tac
```

## Main Dependencies

- Python 3.8
- PyTorch 2.4.1 (CUDA 12.1)
- Open3D 0.19.0
- OpenCV 4.12.0
- NumPy 1.24.4
- Matplotlib 3.7.5
- Pillow 10.4.0

## Usage

Basic usage:

```bash
python infer.py --sensor_type <sensor_type>
```

Supported sensor types:
- `9dtact` - 9DTact sensor
- `gelsight` - GelSight sensor
- `tac3d` - Tac3D sensor
- `dm-tac` - DM-Tac sensor

Examples:

```bash
# Process 9DTact sensor images
python infer.py --sensor_type 9dtact

# Process GelSight sensor images
python infer.py --sensor_type gelsight

# Save contact masks
python infer.py --sensor_type 9dtact --save_mask

# Disable Open3D visualization
python infer.py --sensor_type tac3d --no_open3d
```

## Output

The script generates the following outputs in `output/{sensor_type}/`:

```
output/{sensor_type}/
├── depth_pred_npy/              # Depth maps (NumPy arrays)
│   └── 0000.npy                # Raw depth values (physical units: mm)
├── open3d_view/                 # 3D point cloud visualizations
│   └── 0000_open3d.png         # Rendered point cloud view
├── force_map/                   # Force map heatmaps
│   └── 0000_force.png          # Force distribution (jet colormap)
└── mask_pred/                   # Contact masks (optional, requires --save_mask)
    └── 0000_mask.png           # Binary contact mask
```

Output file descriptions:

1. **Depth maps (depth_pred_npy/)**
   - Format: `.npy` files
   - Content: De-normalized physical depth values (unit: mm)
   - Usage: For further analysis and processing

2. **3D point clouds (open3d_view/)**
   - Format: `.png` images
   - Content: Open3D rendered 3D point cloud views
   - Colormap: YlOrRd_r (red=low depth, yellow=high depth)
   - Features: Real-time display window + automatic screenshot saving

3. **Force maps (force_map/)**
   - Format: `.png` images
   - Content: Force distribution heatmaps
   - Colormap: jet (blue=low force, red=high force)

4. **Contact masks (mask_pred/)** - requires `--save_mask` flag
   - Format: `.png` images
   - Content: Binary contact area masks
   - Colors: black=non-contact, white=contact

## License

See LICENSE file.
