import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv
from   scipy.ndimage import distance_transform_edt
import porespy as ps
import pandas as pd
# -------------------------------------------------------------------
# 1) Helpers for folder / file discovery
# -------------------------------------------------------------------

def list_sample_dirs(base_dir: str) -> List[str]:
    """
    List all directories named 'DeePore_Sample_XXXXX' inside base_dir,
    sorted by the numeric suffix (e.g. 00010 -> 10).

    Returns a list of folder names (not full paths).
    """
    pattern = re.compile(r"^DeePore_Sample_(\d+)$")

    samples: List[Tuple[int, str]] = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue

        m = pattern.match(name)
        if m:
            num_part = int(m.group(1))
            samples.append((num_part, name))

    samples.sort(key=lambda t: t[0])
    return [name for _, name in samples]


def get_raw_path(sample_dir: str, raw_filename: str) -> str:
    """
    Return the full path to the raw file (e.g. domain.raw) inside the sample_dir.
    """
    raw_path = os.path.join(sample_dir, raw_filename)
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    return raw_path


def get_latest_vis_summary_path(sample_dir: str) -> str:
    """
    Inside sample_dir, find all subdirectories named 'visY' where Y is an integer.
    Select the highest Y and return the path to 'summary.pvti' inside it.
    """
    vis_pattern = re.compile(r"^vis(\d+)$")
    vis_candidates: List[Tuple[int, str]] = []

    for name in os.listdir(sample_dir):
        full_path = os.path.join(sample_dir, name)
        if not os.path.isdir(full_path):
            continue

        m = vis_pattern.match(name)
        if m:
            y = int(m.group(1))
            vis_candidates.append((y, full_path))

    if not vis_candidates:
        raise RuntimeError(f"No 'visY' subdirectories found in: {sample_dir}")

    # Pick highest Y
    vis_candidates.sort(key=lambda t: t[0])
    _, latest_vis_dir = vis_candidates[-1]

    summary_path = os.path.join(latest_vis_dir, "summary.pvti")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"'summary.pvti' not found in: {latest_vis_dir}")

    return summary_path


# -------------------------------------------------------------------
# 2) Reading the raw volume and the pvti
# -------------------------------------------------------------------

def read_raw_volume(
    raw_path: str,
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    order: str = "C",
) -> np.ndarray:
    """
    Read a .raw file as a 3D NumPy array with the given shape and dtype.

    Parameters
    ----------
    raw_path : str
        Full path to the .raw file.
    shape : (nx, ny, nz)
        Shape of the 3D volume.
    dtype : np.dtype
        Data type stored in the raw file (e.g. np.uint8, np.float32).
    order : {'C', 'F'}
        Memory order used when reshaping.

    Returns
    -------
    np.ndarray
        3D array with the specified shape and dtype.
    """
    flat = np.fromfile(raw_path, dtype=dtype)
    expected_size = int(np.prod(shape))
    if flat.size != expected_size:
        raise ValueError(
            f"Raw file size mismatch for {raw_path}: "
            f"found {flat.size} elements, expected {expected_size} "
            f"for shape {shape}"
        )

    return flat.reshape(shape, order=order)


def read_summary_pvti(summary_path: str) -> pv.DataSet:
    """
    Read a summary.pvti file as a PyVista object.
    """
    return pv.read(summary_path)


# -------------------------------------------------------------------
# 3) Main high-level functions
# -------------------------------------------------------------------

def load_sample_raw_and_pvti(
    base_dir: str,
    sample_dir_name: str,
    raw_filename: str,
    raw_shape: Tuple[int, int, int],
    raw_dtype: np.dtype,
    raw_order: str = "C",
):
    """
    For a single sample:

    - Go to {base_dir}/{sample_dir_name}
    - Load raw_filename (e.g. domain.raw) as a 3D array
    - Find the highest visY folder inside the sample and read summary.pvti

    Returns
    -------
    (raw_volume, pvti_mesh)
      raw_volume : np.ndarray
      pvti_mesh  : pyvista.DataSet
    """

    sample_dir = os.path.join(base_dir, sample_dir_name)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Sample folder not found: {sample_dir}")

    # domain.raw dentro do sample
    raw_path = get_raw_path(sample_dir, raw_filename)

    raw_volume = read_raw_volume(
        raw_path,
        shape=raw_shape,
        dtype=raw_dtype,
        order=raw_order,
    )

    # summary.pvti na maior visY dentro do sample
    summary_path = get_latest_vis_summary_path(sample_dir)
    pvti_mesh    = read_summary_pvti(summary_path)

    return raw_volume, pvti_mesh

# True if everything is okay
def sanity_check(vol, vel, solid_value=0):
    solid_mask = (vol == solid_value)
    return not np.any(vel[solid_mask] != 0)

import numpy as np

raw_filename    = "domain.raw"
raw_shape       = (256, 256, 256)               # adjust to your volume
raw_dtype       = np.uint8                      # or np.float32, etc.
base_dir        = os.getcwd()+"/DeePore_Samples" # current folder

# Get all sample numbers present
sample_dirs = list_sample_dirs(base_dir)

# Iterate over them and read summary.pvti
data = []

raw_volume = read_raw_volume(
    raw_filename,
    shape=raw_shape,
    dtype=raw_dtype,
)
    
porosities = []
for sample_name in sample_dirs:
    
    
    try:
        raw_vol, mesh = load_sample_raw_and_pvti(
            base_dir=base_dir,
            sample_dir_name=sample_name,
            raw_filename=raw_filename,
            raw_shape=raw_shape,
            raw_dtype=raw_dtype,
        )
        
        vol         = raw_vol.reshape(raw_shape)
        vel_x       = mesh["Velocity_x"].reshape(raw_shape)
        vel_y       = mesh["Velocity_y"].reshape(raw_shape)
        vel_z       = mesh["Velocity_z"].reshape(raw_shape)
            
        # Distance transform is recalculated to increase      
        edt         = mesh["SignDist"].reshape(raw_shape)
        dt          = distance_transform_edt(edt>0)
        
        vel             = np.sqrt(vel_x**2+vel_y**2+vel_z**2)
        vel_mean_pore   = np.mean(vel[edt>0])
        
        visc        = (1.5-0.5)/3
        dens        = 1
        r_max       = np.max(edt)
        Re          = dens*r_max*vel_mean_pore/visc  
        
        data.append(
            {
            "Sample":                       sample_name,
            "Mean Velocity in Pore Space":  vel_mean_pore,
            "Re":                           Re,
            }
        )
        
        por = np.sum(edt>0)
        print(por)
        porosities.append(por)
    except:
        print(f"Sample {sample_name} do not have simulation results yet.")
    
print("Max porosity:", np.max(porosities))
    
df = pd.DataFrame(data)
df.reset_index(drop=True, inplace=True)
df.to_csv("Simulations_Summary.csv")
print(df)
    