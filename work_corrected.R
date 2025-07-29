.libPaths("C:/Users/safro/AppData/Local/R/win-library/4.5.1")
library(rgl)

# Set working directory to the facial_research folder
setwd("C:/Users/safro/VSCODETESTER/facial_research")
load("given_data/facesim100000.RData")

# CORRECTED VERSION:
# The issue was confusing eigenvalues with PC scores
# pc.pcs contains PC SCORES (coefficients), not eigenvalues
# simulated.eigenvalues should actually be simulated PC SCORES

# Number of simulated faces
n_faces <- nrow(simulated.eigenvalues)

# Create a matrix for all PC scores, starting with the original ones
all_pc_scores <- matrix(rep(pc.pcs[1,], n_faces), nrow=n_faces, byrow=TRUE)
all_pc_scores[, 1:5] <- simulated.eigenvalues

# Preallocate matrix for reconstructed faces
reconstructed_faces <- matrix(NA, nrow = n_faces, ncol = nrow(pc.eigenvectors))

# Reconstruct faces using PC scores and eigenvectors
for (i in 1:n_faces) {
  # Multiply eigenvectors by PC SCORES and add mean face
  reconstructed_faces[i, ] <- pc.center + pc.eigenvectors %*% all_pc_scores[i,]
}

# Check reconstruction ranges
cat("Reconstructed faces range:", range(reconstructed_faces), "\n")

# Visualize face i (e.g., i = 1)
plot_face <- function(face_vector, title = "Simulated Face") {
  # Reshape 21480-length vector to 7160 x 3 (x, y, z)
  face_matrix <- matrix(face_vector, ncol = 3, byrow = TRUE)

  # 3D scatter plot
  plot3d(face_matrix[,1], face_matrix[,2], face_matrix[,3],
         col = "skyblue", size = 3, type = "s", xlab = "X", ylab = "Y", zlab = "Z",
         main = title)
}

write.csv(reconstructed_faces, "processed/2000_reconstructed_faces.csv", row.names = FALSE)

# Ensure the output directory exists
output_dir <- "processed/2000_simfaces"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Read template .obj file to get face (triangulation) lines
template_lines <- readLines("given_data/simface 178.obj")
face_lines <- template_lines[grepl("^f ", template_lines)]

# Total number of simulated faces
n_faces <- nrow(reconstructed_faces)

# Loop through all simulated faces and write each as an .obj file
for (i in 1:n_faces) {
  # Reshape reconstructed face vector to 7160 x 3 matrix
  face_matrix <- matrix(reconstructed_faces[i, ], ncol = 3, byrow = TRUE)
  
  # Create "v x y z" lines for each vertex
  vertex_lines <- apply(face_matrix, 1, function(row) {
    sprintf("v %.6f %.6f %.6f", row[1], row[2], row[3])
  })
  
  # Combine vertex and face lines
  obj_lines <- c(vertex_lines, face_lines)
  
  # Define output file path
  filename <- sprintf("%s/simface_5pc_%d.obj", output_dir, i)
  
  # Write to file
  writeLines(obj_lines, filename)
}

cat("Generated", n_faces, "corrected face files in", output_dir, "\n")
