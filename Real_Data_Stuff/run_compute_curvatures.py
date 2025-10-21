from __future__ import annotations
import os
import json
import random
import hashlib
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from scipy.stats import zscore

# -----------------------------
# Geometry / curvature helpers
# -----------------------------

def angle_at_vertex(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    a = v1 - v0
    b = v2 - v0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos_theta = float(np.dot(a, b) / (na * nb))
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def estimate_gaussian_curvature_for_positions(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Angle-deficit Gaussian curvature per vertex for a single face instance.
    faces: (F, 3) int
    vertices: (V, 3) float
    returns: (V,) float curvatures
    """
    V = vertices.shape[0]
    angle_sum = np.zeros(V, dtype=np.float64)
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        a0 = angle_at_vertex(v0, v1, v2)
        a1 = angle_at_vertex(v1, v2, v0)
        a2 = angle_at_vertex(v2, v0, v1)
        angle_sum[i] += a0
        angle_sum[j] += a1
        angle_sum[k] += a2
    return (2.0 * np.pi) - angle_sum


def compute_boundary_mask(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Return boolean mask of interior vertices given triangle faces.
    A vertex is boundary if it touches an edge used by only one face.
    """
    # Count face-usage of each undirected edge
    from collections import defaultdict
    edge_face_count = defaultdict(int)
    for tri in faces:
        e01 = tuple(sorted((int(tri[0]), int(tri[1]))))
        e12 = tuple(sorted((int(tri[1]), int(tri[2]))))
        e20 = tuple(sorted((int(tri[2]), int(tri[0]))))
        for e in (e01, e12, e20):
            edge_face_count[e] += 1
    boundary_vertices = set()
    for (u, v), cnt in edge_face_count.items():
        if cnt == 1:
            boundary_vertices.add(u)
            boundary_vertices.add(v)
    mask = np.ones(n_vertices, dtype=bool)
    if boundary_vertices:
        mask[list(boundary_vertices)] = False
    return mask


# -----------------------------
# Data loading helpers
# -----------------------------

def list_obj_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob('*.obj') if p.is_file()])


def read_faces_from_obj(obj_path: Path) -> tuple[np.ndarray, int]:
    mesh = trimesh.load(obj_path, process=False)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    n_vertices = int(len(mesh.vertices))
    return faces, n_vertices


def read_face_matrix(csv_path: Path, expected_cols: int | None = None) -> np.ndarray:
    """Read face matrix from CSV.
    Handles both with-header and headerless cases.
    Returns an array of shape (N, M) where M should be 3*V.
    If extra columns exist beyond expected_cols, they are truncated; if fewer, raises.
    """
    # Try reading with header, then without
    try:
        df = pd.read_csv(csv_path)
        # If the first row has non-numeric strings like 'x1', 'y1', it's a header; dtypes may be numeric already
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    # If any non-numeric in the first data row, try header=None
    if df.shape[0] > 0 and any(isinstance(x, str) for x in list(df.iloc[0].values)):
        df = pd.read_csv(csv_path, header=0)

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    if df.isnull().values.any():
        # drop rows with NaNs
        df = df.dropna(axis=0, how='any')

    arr = df.values.astype(np.float32, copy=False)
    if expected_cols is not None:
        if arr.shape[1] < expected_cols:
            raise ValueError(f"CSV has {arr.shape[1]} columns but expected at least {expected_cols}")
        if arr.shape[1] > expected_cols:
            arr = arr[:, :expected_cols]
    return arr


# -----------------------------
# Main
# -----------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo_dir: Path) -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo_dir, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compute interior curvatures for real faces from CSV+OBJ topology.")
    here = Path(__file__).resolve().parent
    ap.add_argument('--csv', type=Path, default=here / 'face_x_70.csv', help='Path to face CSV (flattened x,y,z columns)')
    ap.add_argument('--objs', type=Path, default=here / 'Face_OBJs', help='Directory containing OBJ meshes with shared topology')
    ap.add_argument('--outputs', type=Path, default=here / 'outputs', help='Output directory')
    ap.add_argument('--seed', type=int, default=42, help='Seed for random mesh selection and NumPy RNG')
    args = ap.parse_args()

    # Reproducibility: seed both Python random and NumPy
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_csv = args.csv
    obj_dir = args.objs
    out_dir = args.outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_files = list_obj_files(obj_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {obj_dir}")

    mesh_path = random.choice(obj_files)
    print(f"[info] Using mesh: {mesh_path.name}")
    faces, n_vertices = read_faces_from_obj(mesh_path)

    expected_cols = 3 * n_vertices
    X = read_face_matrix(data_csv, expected_cols=expected_cols)
    if X.shape[1] != expected_cols:
        print(f"[warn] Truncated/validated CSV columns to {expected_cols}")

    # Interior mask computed once from topology
    interior_mask = compute_boundary_mask(faces, n_vertices)
    np.save(out_dir / 'interior_mask.npy', interior_mask)

    n_faces = X.shape[0]
    n_interior = int(np.sum(interior_mask))
    all_curv = np.zeros((n_faces, n_interior), dtype=np.float32)

    for i in range(n_faces):
        row = X[i]
        verts = row.reshape(-1, 3)
        if verts.shape[0] != n_vertices:
            raise ValueError(f"Row {i} has {verts.shape[0]} vertices but mesh has {n_vertices}")
        curv = estimate_gaussian_curvature_for_positions(faces, verts)
        interior_curv = np.exp(curv[interior_mask])
        z = zscore(interior_curv)
        z = np.clip(z, -2, 2)
        norm_curv = (z - z.min()) / (np.ptp(z) + 1e-8)
        all_curv[i] = norm_curv
        if (i + 1) % 50 == 0 or (i + 1) == n_faces:
            print(f"  processed {i+1}/{n_faces} faces")

    # Save outputs and the mesh we used
    np.save(out_dir / 'interior_curvatures.npy', all_curv)
    # re-export the chosen mesh for record
    mesh = trimesh.load(mesh_path, process=False)
    mesh.export(out_dir / 'mesh_used.obj')
    # Write run metadata for reproducibility
    meta = {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "seed": args.seed,
        "csv_path": str(data_csv.resolve()),
        "csv_sha256": _sha256(data_csv) if data_csv.exists() else None,
        "obj_dir": str(obj_dir.resolve()),
        "mesh_used": mesh_path.name,
        "mesh_sha256": _sha256(mesh_path) if mesh_path.exists() else None,
        "n_vertices": int(n_vertices),
        "n_faces_rows": int(X.shape[0]),
        "n_interior": int(n_interior),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": getattr(__import__('scipy'), '__version__', None),
            "trimesh": trimesh.__version__,
        },
        "git": _git_info(here.parent),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(f"[done] Saved: {out_dir / 'interior_mask.npy'} and {out_dir / 'interior_curvatures.npy'} and metadata.json")


if __name__ == '__main__':
    main()
