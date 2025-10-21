from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import trimesh

rng = np.random.default_rng(42)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_base_mesh(n_subdiv: int = 1) -> trimesh.Trimesh:
    """Create a small closed mesh (icosphere) with reasonable vertex count."""
    mesh = trimesh.creation.icosphere(subdivisions=n_subdiv, radius=1.0)
    # Make sure it's triangular faces (it is) and watertight (it is)
    return mesh


def export_obj_variants(mesh: trimesh.Trimesh, out_dir: Path, k: int = 3) -> None:
    ensure_dir(out_dir)
    V = mesh.vertices.shape[0]
    for i in range(1, k + 1):
        noise = rng.normal(scale=0.02, size=(V, 3))
        verts = mesh.vertices + noise
        m = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
        m.export(out_dir / f"face_{i:03d}.obj")


def make_face_csv(mesh: trimesh.Trimesh, out_csv: Path, n_faces: int = 8, with_header: bool = True) -> None:
    V = mesh.vertices.shape[0]
    cols = []
    for vid in range(1, V + 1):
        cols.extend([f"x{vid}", f"y{vid}", f"z{vid}"])
    data = []
    for _ in range(n_faces):
        # Start from base mesh and add a slightly larger random deformation per face
        noise = rng.normal(scale=0.05, size=(V, 3))
        verts = mesh.vertices + noise
        row = verts.reshape(-1)
        data.append(row)
    df = pd.DataFrame(data, columns=cols if with_header else None)
    df.to_csv(out_csv, index=False, header=with_header)


def main():
    ap = argparse.ArgumentParser(description="Generate a tiny synthetic dataset of OBJ variants + CSV faces.")
    here = Path(__file__).resolve().parent
    ap.add_argument('--objs', type=Path, default=here / 'Face_OBJs', help='Output directory for OBJ files')
    ap.add_argument('--csv', type=Path, default=here / 'face_x_70.csv', help='Output CSV path')
    ap.add_argument('--n-obj', type=int, default=3, help='Number of OBJ variants to write')
    ap.add_argument('--n-faces', type=int, default=8, help='Number of CSV rows (faces) to generate')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    args = ap.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    ensure_dir(args.objs)

    base_mesh = make_base_mesh(n_subdiv=1)
    export_obj_variants(base_mesh, args.objs, k=args.n_obj)
    make_face_csv(base_mesh, args.csv, n_faces=args.n_faces, with_header=True)
    print(f"[done] Wrote OBJ variants to {args.objs} and CSV to {args.csv}")


if __name__ == '__main__':
    main()
