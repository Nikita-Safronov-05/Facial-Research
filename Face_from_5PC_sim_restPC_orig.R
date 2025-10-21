.libPaths("C:/Users/safro/AppData/Local/R/win-library/4.5.1")
library(rgl)

# Set working directory to the facial_research folder
setwd("C:/Users/safro/VSCODETESTER/facial_research")
load("given_data/facesim100000.RData")

# Read template .obj file to get face (triangulation) lines once
template_lines <- readLines("given_data/simface 178.obj")
face_lines <- template_lines[grepl("^f ", template_lines)]

# Process all seeds: create faces per seed in processed/sim_2000_d4_{seed}/faces
for (seed in 1:100) {
  seed_dir <- file.path("processed", sprintf("sim_2000_d4_%d", seed))
  faces_dir <- file.path(seed_dir, "faces")

  # Check inputs exist
  eigvals_rdata <- file.path(seed_dir, "eigvals.RData")
  eigvals_csv <- file.path(seed_dir, "eigvals.csv")

  if (!file.exists(eigvals_rdata) && !file.exists(eigvals_csv)) {
    cat("[seed", seed, "] skipping: no eigenvalue file found in", seed_dir, "\n")
    next
  }

  # Load simulated PC scores (simulated.eigenvalues)
  if (file.exists(eigvals_rdata)) {
    load(eigvals_rdata)
  } else {
    simulated.eigenvalues <- as.matrix(read.csv(eigvals_csv))
  }

  if (!exists("simulated.eigenvalues")) {
    cat("[seed", seed, "] skipping: variable 'simulated.eigenvalues' not found after load\n")
    next
  }

  n_faces <- nrow(simulated.eigenvalues)

  # Prepare PC scores matrix: start from a baseline row, replace first 5 PCs
  all_pc_scores <- matrix(rep(pc.pcs[1,], n_faces), nrow = n_faces, byrow = TRUE)
  all_pc_scores[, 1:5] <- simulated.eigenvalues

  # Reconstruct faces
  reconstructed_faces <- matrix(NA_real_, nrow = n_faces, ncol = nrow(pc.eigenvectors))
  for (i in 1:n_faces) {
    reconstructed_faces[i, ] <- pc.center + pc.eigenvectors %*% all_pc_scores[i,]
  }

  # Ensure faces output directory exists
  if (!dir.exists(faces_dir)) dir.create(faces_dir, recursive = TRUE, showWarnings = FALSE)

  # Write OBJ files
  for (i in 1:n_faces) {
    face_matrix <- matrix(reconstructed_faces[i, ], ncol = 3, byrow = TRUE)
    vertex_lines <- apply(face_matrix, 1, function(row) sprintf("v %.6f %.6f %.6f", row[1], row[2], row[3]))
    obj_lines <- c(vertex_lines, face_lines)
    filename <- file.path(faces_dir, sprintf("simface_%d.obj", i))
    writeLines(obj_lines, filename)
  }

  cat("[seed", seed, "] generated", n_faces, "face files in", faces_dir, "\n")
}
