import os
import re
from collections import defaultdict
import numpy as np
import trimesh
from scipy.stats import zscore

# Define curvature estimation function
def angle_at_vertex(v0, v1, v2):
    a = v1 - v0
    b = v2 - v0
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def estimate_gaussian_curvature(mesh):
    angle_sum = np.zeros(len(mesh.vertices))
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        a0 = angle_at_vertex(v0, v1, v2)
        a1 = angle_at_vertex(v1, v2, v0)
        a2 = angle_at_vertex(v2, v0, v1)
        angle_sum[face[0]] += a0
        angle_sum[face[1]] += a1
        angle_sum[face[2]] += a2
    return 2 * np.pi - angle_sum


def compute_boundary_mask(mesh):
    # Kept for reference; not used when a global mask is provided.
    edge_face_count = defaultdict(int)
    for face in mesh.faces:
        edges = [tuple(sorted([face[j], face[(j + 1) % 3]])) for j in range(3)]
        for edge in edges:
            edge_face_count[edge] += 1
    boundary_edges = np.array([edge for edge, count in edge_face_count.items() if count == 1])
    boundary_vertices = np.unique(boundary_edges)
    n_vertices = len(mesh.vertices)
    interior_mask = np.ones(n_vertices, dtype=bool)
    interior_mask[boundary_vertices] = False
    return interior_mask


def list_face_objs(folder):
    # Find files like simface_5pc_123.obj and sort by index
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith('.obj') and f.startswith('simface_')]
    def idx(f):
        m = re.search(r"simface_(\d+)\.obj$", f, re.IGNORECASE)
        return int(m.group(1)) if m else 0
    return [os.path.join(folder, f) for f in sorted(files, key=idx)]


def process_seed_faces(seed_dir, interior_mask):
    faces_dir = os.path.join(seed_dir, 'faces')
    obj_paths = list_face_objs(faces_dir)
    if not obj_paths:
        print(f"[skip] No OBJ files in {faces_dir}")
        return False

    n_interior = int(np.sum(interior_mask))

    # Prepare array for all faces
    all_curvatures = np.zeros((len(obj_paths), n_interior), dtype=np.float32)

    for i, path in enumerate(obj_paths, start=1):
        mesh = trimesh.load(path)
        curvatures = estimate_gaussian_curvature(mesh)
        interior_curvatures = np.exp(curvatures[interior_mask])
        z = zscore(interior_curvatures)
        z = np.clip(z, -2, 2)
        norm_curv = (z - z.min()) / (np.ptp(z) + 1e-8)
        all_curvatures[i - 1] = norm_curv

        if i % 200 == 0:
            print(f"  processed {i}/{len(obj_paths)} faces in {faces_dir}")

    # Save per-seed outputs
    np.save(os.path.join(seed_dir, 'interior_curvatures.npy'), all_curvatures)
    print(f"[done] Saved curvatures to {os.path.join(seed_dir, 'interior_curvatures.npy')}")
    return True


def main():
    root = os.path.dirname(__file__)
    processed_root = os.path.join(root, 'processed')

    # Load one global interior mask and reuse it for all seeds
    global_mask_path = os.path.join(processed_root, 'interior_mask.npy')
    if not os.path.isfile(global_mask_path):
        raise FileNotFoundError(f"Global interior mask not found: {global_mask_path}")
    interior_mask = np.load(global_mask_path).astype(bool)

    # Iterate through expected seed folders sim_2000_d4_{seed}
    for seed in range(1, 101):
        seed_dir = os.path.join(processed_root, f'sim_2000_d4_{seed}')
        if not os.path.isdir(seed_dir):
            print(f"[skip] Missing folder: {seed_dir}")
            continue
        print(f"[seed {seed}] Processing {seed_dir}")
        process_seed_faces(seed_dir, interior_mask)


if __name__ == '__main__':
    main()