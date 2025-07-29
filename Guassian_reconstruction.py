import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from collections import defaultdict
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.csgraph import shortest_path
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
import os

def load_data(mask_file="interior_mask.npy", processed_file="interior_curvatures_good.npy"):
    os.chdir("facial_research")
    global all_curvatures, interior_mask, n_faces, n_interior, interior_data
    all_curvatures = np.load(f"processed/{processed_file}")
    interior_mask = np.load(f"processed/{mask_file}")
    n_faces, n_interior = all_curvatures.shape
    print(f"Number of faces: {n_faces}, Number of interior vertices: {n_interior}\n")
    # Concatenate all interior curvature data
    interior_data = np.concatenate(all_curvatures)

def corr_matrix(option: int):
    std_per_vertex = all_curvatures.std(axis=0)
    print("Vertices with zero std:", np.where(std_per_vertex == 0)[0])
    if option == 1:
        # Option 1: Use already-normalized curvature directly
        with np.errstate(invalid='ignore', divide='ignore'):
            result = np.corrcoef(all_curvatures.T)
            if np.isnan(result).any() or np.isinf(result).any():
                print("\nWarning: invalid values encountered during computation.")
            else:
                print("\nComputation completed without invalid values.")
            result = np.nan_to_num(result, nan=0.0)
            np.fill_diagonal(result, 1.0)
            return result
    elif option == 2:
        # Option 2: Use standardized concatenated data
        scaler = StandardScaler()
        reshaped = all_curvatures.reshape(n_faces * n_interior, 1)
        standardized = scaler.fit_transform(reshaped).reshape(n_faces, n_interior)
        result = np.corrcoef(standardized.T)
        if np.isnan(result).any() or np.isinf(result).any():
            print("\nWarning: invalid values encountered during computation.")
        else:
            print("\nComputation completed without invalid values.")
        result = np.nan_to_num(result, nan=0.0)
        np.fill_diagonal(result, 1.0)
        return result
    else:
        raise ValueError("\nInvalid option. Choose 1 or 2.")

def compute_spatial_affinity_geodesic(mesh, interior_mask, sigma=50.0):
    """
    Computes a "soft" affinity matrix based on geodesic distance on the mesh.

    Args:
        mesh (trimesh.Trimesh): The base mesh object.
        interior_mask (np.array): Boolean mask for interior vertices.
        sigma (float): The width of the Gaussian kernel. THIS IS A CRITICAL
                       PARAMETER TO TUNE. It controls the "radius of influence".

    Returns:
        np.ndarray: A dense spatial affinity matrix for interior vertices.
    """
    print("Computing spatial affinity based on geodesic distance...")
    
    interior_indices = np.where(interior_mask)[0]
    n_interior = len(interior_indices)

    # Build the local adjacency graph for ONLY the interior vertices, weighted by edge length
    v_mask = np.zeros(len(mesh.vertices), dtype=bool)
    v_mask[interior_indices] = True
    edges = mesh.edges[np.all(v_mask[mesh.edges], axis=1)]
    global_to_local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(interior_indices)}
    local_edges = np.array([(global_to_local_map[u], global_to_local_map[v]) for u, v in edges])
    
    # Use actual edge lengths for a weighted graph, which is crucial for shortest_path
    edge_lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    
    local_adj_matrix = np.zeros((n_interior, n_interior))
    local_adj_matrix[local_edges[:, 0], local_edges[:, 1]] = edge_lengths
    local_adj_matrix[local_edges[:, 1], local_edges[:, 0]] = edge_lengths

    # --- KEY CALCULATION ---
    print("  Calculating all-pairs shortest paths (geodesic distances)...")
    # This computes the shortest path distance between all pairs of nodes in the weighted graph
    distances = shortest_path(csgraph=local_adj_matrix, directed=False)
    
    # Handle infinite distances for any disconnected components
    max_dist = np.max(distances[np.isfinite(distances)])
    distances[np.isinf(distances)] = max_dist * 1.5

    print(f"  Applying Gaussian kernel with sigma = {sigma}...")
    # Apply the Gaussian (RBF) kernel to convert distances to affinities
    # This creates the smooth fall-off we need
    spatial_affinity = np.exp(-distances**2 / (2 * sigma**2))
    
    print("Spatial affinity computed.")
    return spatial_affinity

def check_correlation_properties(correlation_matrix):
    print("\nCorrelation matrix properties:\n")
    print("Correlation matrix shape:", correlation_matrix.shape)
    print("Nans or infinite values:", np.isnan(correlation_matrix).any(), np.isinf(correlation_matrix).any())
    print('Diagonal:', np.allclose(np.diag(correlation_matrix), 1))
    print('Symmetric:', np.allclose(correlation_matrix, correlation_matrix.T))
    # eigvals = np.linalg.eigvalsh(correlation_matrix)
    # print("Min eigenvalue:", eigvals.min())

def visualize_correlation(correlation_matrix):
    off_diag = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    eigvals = np.linalg.eigvalsh(correlation_matrix)
    plt.hist(off_diag, bins=50)
    plt.title("Distribution of off-diagonal correlations")
    plt.show()
    plt.plot(eigvals)
    plt.show()
    diffs = np.diff(eigvals)
    plt.plot(diffs[:100])
    plt.show()

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
    def __init__(self, min_cluster_size=20, fiedler_value_threshold=0.1, partition_method='kmeans'):
        """
        Args:
            partition_method (str): 'median' or 'kmeans'. Defines how to split
                                    the Fiedler vector.
        """
        if partition_method not in ['median', 'kmeans']:
            raise ValueError("partition_method must be 'median' or 'kmeans'")
            
        self.min_cluster_size = min_cluster_size
        self.fiedler_value_threshold = fiedler_value_threshold
        self.partition_method = partition_method # Store the chosen method
        self.assignments = None
        self._next_cluster_id = 0

    def fit(self, affinity_matrix):
        self.A = affinity_matrix
        n_samples = self.A.shape[0]
        self.assignments = np.full(n_samples, -1, dtype=int)
        initial_indices = np.arange(n_samples)
        self._recursive_split(initial_indices)
        self.n_clusters_ = self._next_cluster_id
        return self

    def _compute_normalized_laplacian(self, sub_affinity_matrix):
        d = np.sum(sub_affinity_matrix, axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d>0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        I = np.identity(sub_affinity_matrix.shape[0])
        L_sym = I - (D_inv_sqrt @ sub_affinity_matrix @ D_inv_sqrt)
        return L_sym


    def _recursive_split(self, indices):
        """The core recursive function, now with a choice of partitioning."""
        
        if len(indices) < self.min_cluster_size:
            self.assignments[indices] = self._next_cluster_id
            self._next_cluster_id += 1
            return

        sub_A = self.A[np.ix_(indices, indices)]
        L_sym = self._compute_normalized_laplacian(sub_A)
        eigvals, eigvecs = eigh(L_sym, subset_by_index=[0, 1])
        fiedler_value = eigvals[1]
        
        if fiedler_value < self.fiedler_value_threshold:
            self.assignments[indices] = self._next_cluster_id
            self._next_cluster_id += 1
            return

        fiedler_vector = eigvecs[:, 1]
        
        if self.partition_method == 'median':
            # Option 1: Simple median split
            mask = fiedler_vector >= np.median(fiedler_vector)
            
        elif self.partition_method == 'kmeans':
            # Option 2: Use K-Means on the 1D Fiedler vector
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
            
            # Reshape vector to be (n_samples, 1) as expected by sklearn
            labels = kmeans.fit_predict(fiedler_vector.reshape(-1, 1))
            mask = (labels == 1) # Assign one of the labels to be the 'True' part of the mask

        indices_c1 = indices[mask]
        indices_c2 = indices[~mask]

        # --- Final recursion logic (unchanged) ---
        if len(indices_c1) == 0 or len(indices_c2) == 0:
            self.assignments[indices] = self._next_cluster_id
            self._next_cluster_id += 1
            return
        else:
            self._recursive_split(indices_c1)
            self._recursive_split(indices_c2)
        
load_data()
correlation_matrix = corr_matrix(1) # 1- normalized curvature, 2-standardized concatenated data
print("\nCorrelation matrix sample:\n", correlation_matrix[:5, :5])
check_correlation_properties(correlation_matrix)
visualize_correlation(correlation_matrix)

# functional_affinity = np.exp(-0.5 * (1 - correlation_matrix))
# np.fill_diagonal(functional_affinity, 1) # Set diagonal to 1
# # --- COMPUTE SPATIAL AFFINITY (THE NEW STEP) ---
# # Load one of the meshes to get its structure. The topology should be the same for all.
# base_mesh = trimesh.load("5pc_simfaces/simface_5pc_1.obj")
# spatial_affinity = compute_spatial_affinity_geodesic(base_mesh, interior_mask)
# print("\nSpatial affinity matrix sample:\n", spatial_affinity[:5, :5])
# plt.imshow(spatial_affinity**0.01, cmap='hot')
# plt.title("Spatial Affinity Matrix Heatmap")
# plt.colorbar()
# plt.show()

# # --- COMBINE THE TWO AFFINITIES ---
# # Element-wise multiplication ensures high affinity only if BOTH are high.
# final_affinity = functional_affinity * (spatial_affinity ** 0.05)

# print("\nCombined affinity matrix stats:")
# print(f"Min: {np.min(final_affinity):.4f}, Max: {np.max(final_affinity):.4f}, Mean: {np.mean(final_affinity):.4f}")

# print("\nAffinity matrix sample:\n", final_affinity[:5, :5])
# print("Affinity matrix stats:")
# print("Min:", np.min(final_affinity))
# print("Max:", np.max(final_affinity))
# print("Mean:", np.mean(final_affinity))
# print("Sparsity (% zeros):", np.mean(final_affinity == 0) * 100)
# plt.imshow(final_affinity, cmap='hot')
# plt.title("Affinity Matrix Heatmap")
# plt.colorbar()
# plt.show()

# print("\nStarting Hierarchical Spectral Clustering...\n")
# hsc = HierarchicalSpectralClustering(
#     min_cluster_size=3000, 
#     fiedler_value_threshold=0.01, # This is a good starting point to tune from
#     partition_method='kmeans'  # Change to 'median' if you prefer that method
# )

# hsc.fit(final_affinity)
# final_assignments = hsc.assignments

# print(f"\nClustering complete.")
# print(f"Found a total of {hsc.n_clusters_} clusters.")
# print("Example assignments:", final_assignments[:50])
# print("Cluster sizes:", np.bincount(final_assignments[final_assignments >= 0]))

# # kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
# # final_assignments = kmeans.fit_predict(correlation_matrix)

# visualize_clusters(n_faces=3, n_clusters=hsc.n_clusters_, labels=final_assignments, interior_mask=interior_mask)
