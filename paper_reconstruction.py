import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
import os
import trimesh
from scipy.linalg import eigh
from sklearn.cluster import KMeans

# Load Data
os.chdir("facial_research")
reconstructed_faces = pd.read_csv("data/reconstructed_faces.csv").values
n_faces, n_coords = reconstructed_faces.shape
n_vertices = n_coords // 3

# Step 1: Reshape Data
face_data = reconstructed_faces.reshape(n_faces, n_vertices, 3)
vertex_signals = face_data.transpose(1, 0, 2).reshape(n_vertices, n_faces * 3)
print(face_data.shape, face_data.transpose(1, 0, 2).shape, vertex_signals.shape)

# Step 2: Form Spatially Informed k-NN Graph
# k = 80  # Number of neighbors for k-NN
# spatial_affinity = kneighbors_graph(vertex_signals, n_neighbors=k, mode='connectivity', include_self=True)

# # Step 3: Construct Normalized Laplacian
# laplacian = csgraph.laplacian(spatial_affinity, normed=True)

# # Step 4: Eigen Decomposition
# num_eigenvectors = 200
# eigvals, eigvecs = eigsh(laplacian, k=num_eigenvectors, which='SM')
# X = eigvecs

# Step 2: Form Affinity Matrix
euclidean_distances = pdist(vertex_signals, 'euclidean')
correlation_distances = pdist(vertex_signals, 'correlation')
sigma_corr = np.mean(correlation_distances)
sigma_euc = np.mean(euclidean_distances)
affinity_matrix_corr = np.exp(-correlation_distances ** 2 / (2 * sigma_corr ** 2))
affinity_matrix_corr = squareform(affinity_matrix_corr)
affinity_matrix_euc = np.exp(-euclidean_distances ** 2 / (2 * sigma_euc ** 2))
affinity_matrix_euc = squareform(affinity_matrix_euc)
affinity_matrix = affinity_matrix_euc
print(affinity_matrix[:10, :10])
np.fill_diagonal(affinity_matrix, 0)

# # Step 3: Construct Normalized Laplacian
# D = np.diag(affinity_matrix.sum(axis=1))
# D_inv_sqrt = np.linalg.inv(np.sqrt(D))
# laplacian = D_inv_sqrt @ affinity_matrix @ D_inv_sqrt

# # Step 4: Eigen Decomposition
# num_eigenvectors = 20
# eigvals, eigvecs = np.linalg.eigh(laplacian)

# X = eigvecs[:, -num_eigenvectors:]

# # Normalize eigenvectors
# row_norms = np.linalg.norm(X, axis=1, keepdims=True)
# row_norms[row_norms == 0] = 1e-10
# Y = X / row_norms
# print(Y[:5])

# Step 5: Recursive Clustering with Spatial Constraints
def visualize_clusters(n_faces, n_clusters, labels, interior_mask):
    for j in range(1, n_faces + 1):
        mesh = trimesh.load(f"5pc_simfaces/simface_5pc_{j}.obj")
        n_vertices = len(mesh.vertices)
        segment_colors = np.zeros((n_vertices, 3))  # Default color for all vertices
        interior_indices = np.where(interior_mask)[0]

        for i in range(n_clusters):
            cluster_mask = (labels == i)
            cluster_member_indices = np.where(cluster_mask)[0]
            
            if len(cluster_member_indices) == 0:
                continue
                
            # Get the corresponding global vertex indices on the full mesh
            global_vertex_indices = interior_indices[cluster_member_indices]
            
            # Assign a random color to this segment
            color = np.random.rand(3)
            segment_colors[global_vertex_indices] = color

        # Convert to uint8 for visualization
        mesh.visual.vertex_colors[:, :3] = (segment_colors * 255).astype(np.uint8)

        # Optionally set transparency for better visualization
        mesh.visual.vertex_colors[:, 3] = 255  # Full opacity

        mesh.show()

class HierarchicalSpectralClustering:
    """
    Performs hierarchical bifurcated clustering.
    Now includes a choice of partitioning method: 'median' or 'kmeans'.
    """
    def __init__(self, min_cluster_size=20, num_eigenvectors=20):
        """
        Args:
            partition_method (str): 'median' or 'kmeans'. Defines how to split
                                    the Fiedler vector.
        """
            
        self.min_cluster_size = min_cluster_size
        self.num_eigenvectors = num_eigenvectors
        self.assignments = None
        self._next_cluster_id = 0

    def fit(self, affinity_matrix):
        self.A = affinity_matrix
        n_samples = self.A.shape[0]
        self.assignments = np.full(n_samples, -1, dtype=int)
        initial_indices = np.arange(n_samples)
        self._recursive_split(initial_indices, self.num_eigenvectors)
        self.n_clusters_ = self._next_cluster_id
        return self

    def _compute_normalized_laplacian(self, sub_affinity_matrix):
        d = np.sum(sub_affinity_matrix, axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d>0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        #I = np.identity(sub_affinity_matrix.shape[0])
        L_sym = D_inv_sqrt @ sub_affinity_matrix @ D_inv_sqrt
        return L_sym


    def _recursive_split(self, indices, num_eigenvectors):
        """The core recursive function, now with a choice of partitioning."""
        
        if len(indices) < self.min_cluster_size:
            self.assignments[indices] = self._next_cluster_id
            self._next_cluster_id += 1
            return

        sub_A = self.A[np.ix_(indices, indices)]
        L_sym = self._compute_normalized_laplacian(sub_A)
        eigvals, eigvecs = eigh(L_sym)

        X = eigvecs[:, -num_eigenvectors:]

        # Normalize eigenvectors
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1e-10
        Y = X / row_norms
            
        kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
            
        # Reshape vector to be (n_samples, 1) as expected by sklearn
        labels = kmeans.fit_predict(Y)
        mask = (labels == 1) # Assign one of the labels to be the 'True' part of the mask

        indices_c1 = indices[mask]
        indices_c2 = indices[~mask]

        # --- Final recursion logic (unchanged) ---
        if len(indices_c1) == 0 or len(indices_c2) == 0:
            self.assignments[indices] = self._next_cluster_id
            self._next_cluster_id += 1
            return
        else:
            self._recursive_split(indices_c1, num_eigenvectors)
            self._recursive_split(indices_c2, num_eigenvectors)

# Perform Hierarchical Clustering
hsc = HierarchicalSpectralClustering(min_cluster_size=300, num_eigenvectors=20)
hsc.fit(affinity_matrix)
final_assignments = hsc.assignments

print(f"\nClustering complete.")
print(f"Found a total of {hsc.n_clusters_} clusters.")
print("Example assignments:", final_assignments[:50])
print("Cluster sizes:", np.bincount(final_assignments[final_assignments >= 0]))

visualize_clusters(3, hsc.n_clusters_, final_assignments, [True] * n_vertices)