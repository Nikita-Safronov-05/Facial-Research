from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt


def load_csv_row(csv_path: Path, row_index: int, expected_cols: int) -> np.ndarray:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)
    # If first row includes strings like x1,y1,... treat it as header
    if df.shape[0] > 0 and any(isinstance(x, str) for x in list(df.iloc[0].values)):
        df = pd.read_csv(csv_path, header=0)
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
    arr = df.values.astype(np.float32, copy=False)
    if arr.shape[1] < expected_cols:
        raise ValueError(f"CSV has {arr.shape[1]} columns but expected {expected_cols}")
    if arr.shape[1] > expected_cols:
        arr = arr[:, :expected_cols]
    if not (0 <= row_index < arr.shape[0]):
        raise IndexError(f"row_index {row_index} out of range 0..{arr.shape[0]-1}")
    return arr[row_index]


def make_colored_mesh(faces: np.ndarray, vertices: np.ndarray, interior_mask: np.ndarray, values: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Map values (assumed in [0,1]) to colors
    cmap = plt.cm.get_cmap('viridis')
    vals = np.asarray(values, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    colors = (cmap(vals)[:, :3] * 255).astype(np.uint8)
    # set all vertices to white, then paint interior
    vcols = np.ones((vertices.shape[0], 4), dtype=np.uint8) * 255
    vcols[interior_mask, :3] = colors
    mesh.visual.vertex_colors = vcols
    return mesh


def save_previews(mesh: trimesh.Trimesh, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save PLY with vertex colors (widely supported)
    mesh.export(out_dir / f"colored_{tag}.ply")
    # Try to render a small PNG (may require pyglet/pygltflib installed; trimesh will fallback)
    try:
        png = mesh.scene().save_image(resolution=(800, 600), visible=True)
        if png is not None:
            (out_dir / f"colored_{tag}.png").write_bytes(png)
    except Exception:
        # Rendering is optional; ignore errors in headless envs
        pass


def main():
    ap = argparse.ArgumentParser(description="Visualize curvature coloring on OBJ/PLY for a specific face index.")
    ap.add_argument('--outputs', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    ap.add_argument('--csv', type=Path, default=Path(__file__).resolve().parent / 'face_x_70.csv')
    ap.add_argument('--mesh', type=Path, default=None, help='Optional: explicit mesh OBJ path. Defaults to outputs/mesh_used.obj')
    ap.add_argument('--index', type=int, default=0, help='Row index in CSV and interior_curvatures.npy to visualize')
    args = ap.parse_args()

    out_dir = args.outputs
    mesh_path = args.mesh or (out_dir / 'mesh_used.obj')
    interior_mask = np.load(out_dir / 'interior_mask.npy').astype(bool)
    all_curv = np.load(out_dir / 'interior_curvatures.npy')

    mesh = trimesh.load(mesh_path, process=False)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    n_vertices = int(len(mesh.vertices))

    # Load the vertex positions for this row from CSV to make the geometry match
    row = load_csv_row(args.csv, args.index, expected_cols=3 * n_vertices)
    verts = row.reshape(-1, 3)

    # Pick the curvature vector for this row (already normalized [0,1] by the pipeline)
    if not (0 <= args.index < all_curv.shape[0]):
        raise IndexError(f"index {args.index} out of range 0..{all_curv.shape[0]-1}")
    vals = all_curv[args.index]

    colored = make_colored_mesh(faces, verts, interior_mask, vals)
    save_previews(colored, out_dir, tag=f"row{args.index:03d}")
    print(f"[done] Wrote colored preview(s) to {out_dir}")


if __name__ == '__main__':
    main()
