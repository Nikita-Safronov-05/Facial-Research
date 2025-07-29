import numpy as np
import trimesh
from scipy.sparse import csr_matrix, diags
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

# Load the mesh
mesh = trimesh.load('simface 178.obj')

def safe_cotangent(v0, v1, v2):
    a = v1 - v0
    b = v2 - v0
    dot_ab = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    cos_angle = np.clip(dot_ab / (norm_a * norm_b), -1.0, 1.0)
    sin_angle = np.sqrt(max(1 - cos_angle**2, 1e-8))
    cot = cos_angle / sin_angle
    return cot if np.isfinite(cot) else 0  # Ensure NaN becomes 0

n_vertices = len(mesh.vertices)
row, col, data = [], [], []

# Compute cotangent weights
for face in mesh.faces:
    for i in range(3):
        i0, i1, i2 = face[i], face[(i+1) % 3], face[(i+2) % 3]
        v0, v1, v2 = mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]
        cot = safe_cotangent(v0, v1, v2)
        scaled_cot = cot * 1000
        row.extend([i0, i1])
        col.extend([i1, i0])
        data.extend([scaled_cot, scaled_cot])

# Create the sparse similarity matrix
W = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
W.setdiag(0)  # Set self-loops to zero

# Make the degree matrix and compute the Laplacian
degree = np.array(W.sum(axis=1)).flatten()
D = diags(degree)
L = D - W

# Ensure no NaN in any matrix by replacing them
L.data = np.nan_to_num(L.data)  # Replace NaNs with zero

# Counter-check for any remaining NaNs or Infs
assert not np.any(np.isnan(L.toarray())), "Laplacian contains NaNs."
assert not np.any(np.isinf(L.toarray())), "Laplacian contains Infs."

# Use Spectral Embedding to get Laplacian Eigenmaps
embedding = SpectralEmbedding(n_components=3, affinity='precomputed')
laplacian_eigenmaps = embedding.fit_transform(L)

# Visualize the embedding
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    laplacian_eigenmaps[:, 0],
    laplacian_eigenmaps[:, 1],
    laplacian_eigenmaps[:, 2],
    c='b', marker='o', s=5
)
ax.set_title('Laplace Eigenmaps of the Mesh')
plt.show()

# Map the first component to color vertices
norm_eigenmaps = np.nan_to_num(laplacian_eigenmaps[:, 0])
norm_eigenmaps = (norm_eigenmaps - norm_eigenmaps.min()) / max(norm_eigenmaps.ptp(), 1e-8)
colors = plt.cm.viridis(norm_eigenmaps)
mesh.visual.vertex_colors = (colors[:, :3] * 255).astype(np.uint8)

# Show the mesh
mesh.show()