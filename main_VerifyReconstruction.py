import h5py
import numpy as np
import os
import numpy as np
import pyvista as pv
import torch
from   scipy.ndimage import distance_transform_edt

def write_pvti_from_arrays(
    vel_z: np.ndarray,
    vel_y: np.ndarray,
    vel_x: np.ndarray,
    edt:   np.ndarray,
    filename: str = "output.pvti",
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
):
    """
    Save 4 arrays (shape (D, H, W)) into a .pvti file as a UniformGrid.

    Fields:
        - 'Velocity_x', 'Velocity_y', 'Velocity_z', 'SignDist'
    """

    # --- 1) Sanity checks ---
    if not (vel_x.shape == vel_y.shape == vel_z.shape == edt.shape):
        raise ValueError("All arrays must have the same shape (D, H, W).")

    D, H, W = vel_x.shape  # assuming (z, y, x)

    # --- 2) Create a UniformGrid (VTK ImageData) ---
    grid = pv.ImageData()
    # VTK uses (nx, ny, nz) = (W, H, D)
    grid.dimensions = (W, H, D)
    grid.spacing    = spacing  # voxel size
    grid.origin     = origin   # origin

    # --- 3) Attach arrays as point data ---
    # VTK / PyVista expect data flattened in Fortran order (z, y, x).
    grid["Velocity_x"] = vel_x.ravel(order="F")
    grid["Velocity_y"] = vel_y.ravel(order="F")
    grid["Velocity_z"] = vel_z.ravel(order="F")
    grid["SignDist"]   = edt.ravel(order="F")

    # --- 4) Save with .pvti extension ---
    # PyVista chooses the correct VTK XML writer based on extension.
    grid.save(filename)
    print(f"Saved {filename} with fields: Velocity_x, Velocity_y, Velocity_z, SignDist")
    
    
"""
Lê a amostra de índice s do arquivo HDF5 e reconstrói
os volumes 3D completos de vel_z, vel_y, vel_x e edt.

Retorna:
    vel_z_3d, vel_y_3d, vel_x_3d, edt_3d
todos com shape (D, H, W).
"""  
def get_sample(h5_path: str, sample_idx: int):

    with h5py.File(h5_path, "r") as f:
        # Shape original do domínio
        D, H, W     = f.attrs["raw_shape"]
        
        # Get how many porous cells where represented (differ from sample to sample)
        max_points  = f.attrs["max_points"]

        # Load the sample's data from dataset specific row
        vel_z       = f["vel_z"][sample_idx]          
        vel_y       = f["vel_y"][sample_idx]
        vel_x       = f["vel_x"][sample_idx]
        coori       = f["coorX"][sample_idx]   
        coorj       = f["coorY"][sample_idx]  
        coork       = f["coorZ"][sample_idx]  
        edt         = f["edt"][sample_idx]          
        n_valid     = f["n_valid"][sample_idx]

        # Retrieve only porous cells
        vel_z_valid     = vel_z[:n_valid]
        vel_y_valid     = vel_y[:n_valid]
        vel_x_valid     = vel_x[:n_valid]
        edt_valid       = edt[:n_valid]
        i               = coori[:n_valid]
        j               = coorj[:n_valid]
        k               = coork[:n_valid]
        
        # Create solid regions
        vel_z_3d = np.zeros((D, H, W), dtype=np.float16)
        vel_y_3d = np.zeros((D, H, W), dtype=np.float16)
        vel_x_3d = np.zeros((D, H, W), dtype=np.float16)
        edt_3d   = np.zeros((D, H, W), dtype=np.float16)

        # Fill porous regions
        vel_z_3d[k, j, i] = vel_z_valid
        vel_y_3d[k, j, i] = vel_y_valid
        vel_x_3d[k, j, i] = vel_x_valid
        edt_3d  [k, j, i] = edt_valid
        
        # Turn into Pytorch tensors
        vel_z_3d         = torch.as_tensor(vel_z_3d, dtype=torch.float16)
        vel_y_3d         = torch.as_tensor(vel_y_3d, dtype=torch.float16)
        vel_x_3d         = torch.as_tensor(vel_x_3d, dtype=torch.float16)
        edt_3d           = torch.as_tensor(edt_3d, dtype=torch.float16)

        return vel_z_3d, vel_y_3d, vel_x_3d, edt_3d
    
    
# LOAD DATA FROM DATASET
dataset_path = "LBPM_Dataset.h5"
sample_idx = 0
vel_z_rec, vel_y_rec, vel_x_rec, edt_rec = get_sample(dataset_path, sample_idx)

# LOAD DATA FROM SIMULATIONS
raw_shape       = (256, 256, 256)
equivalent_simu_path = os.path.join(
    "DeePore_Samples",
    "DeePore_Sample_00000",
    "vis152000",
)
summary_path    = os.path.join(equivalent_simu_path, "summary.pvti")
mesh            = pv.read(summary_path)
vel_x_orig      = mesh["Velocity_x"].reshape(raw_shape, order="F")
vel_y_orig      = mesh["Velocity_y"].reshape(raw_shape, order="F")
vel_z_orig      = mesh["Velocity_z"].reshape(raw_shape, order="F")
sign_dist_orig  = mesh["SignDist"].reshape(raw_shape, order="F")
sign_dist_orig  = distance_transform_edt(sign_dist_orig>0)

write_pvti_from_arrays(vel_z_orig,
                       vel_y_orig,
                       vel_x_orig,
                       sign_dist_orig,
                       "example_orig.vti")

# CONVERT ALL DATA TO TENSOR FORMAT (to be comparable)
vel_x_orig      = torch.as_tensor(vel_x_orig, dtype=torch.float16)
vel_y_orig      = torch.as_tensor(vel_y_orig, dtype=torch.float16)
vel_z_orig      = torch.as_tensor(vel_z_orig, dtype=torch.float16)
sign_dist_orig  = torch.as_tensor(sign_dist_orig, dtype=torch.float16)


# VERIFY IF TENSOR IS THE SAME
is_same            = torch.equal(edt_rec, sign_dist_orig)
print("Means z: ", vel_z_orig.mean(), vel_z_rec.mean())
print("Means y: ", vel_y_orig.mean(), vel_y_rec.mean())
print("Means x: ", vel_x_orig.mean(), vel_x_rec.mean())
print("Arrays the same: ", is_same)


# SAVE TO VISUALIZE IN PARAVIEW
"""
write_pvti_from_arrays(vel_z_rec.numpy().astype(np.float32),
                       vel_y_rec.numpy().astype(np.float32),
                       vel_x_rec.numpy().astype(np.float32), 
                       edt_rec.numpy().astype(np.float32),
                       "example_rec_float16.vti")

write_pvti_from_arrays(vel_z_orig.numpy().astype(np.float32),
                       vel_y_orig.numpy().astype(np.float32),
                       vel_x_orig.numpy().astype(np.float32),
                       sign_dist_orig.numpy().astype(np.float32),
                       "example_orig_float16.vti")
"""
