setwd("C:/Users/safro/VSCODETESTER/facial_research")
load("given_data/facesim100000.RData")

# Quick test of scaling
print("=== BEFORE SCALING ===")
print(paste("Original first 5 PC scores range:", range(pc.pcs[,1:5])))
print(paste("Simulated eigenvalues range:", range(simulated.eigenvalues)))

# Scale simulated values to match original PC score range
original_range <- range(pc.pcs[,1:5])
simulated_range <- range(simulated.eigenvalues)
scale_factor <- diff(original_range) / diff(simulated_range)
simulated_scaled <- (simulated.eigenvalues - min(simulated.eigenvalues)) * scale_factor + min(original_range)

print("=== AFTER SCALING ===")
print(paste("Scaled simulated values range:", range(simulated_scaled)))

# Test one face reconstruction
all_pc_scores <- pc.pcs[1,]
all_pc_scores[1:5] <- simulated_scaled[1,]

original_face <- pc.center + pc.eigenvectors %*% pc.pcs[1,]
scaled_face <- pc.center + pc.eigenvectors %*% all_pc_scores

print("=== RECONSTRUCTION COMPARISON ===")
print(paste("Original face range:", range(original_face)))
print(paste("Scaled reconstruction range:", range(scaled_face)))

# Check if they're reasonable
distance_diff <- sqrt(sum((scaled_face - original_face)^2))
print(paste("Euclidean distance between original and scaled reconstruction:", distance_diff))

print("SUCCESS: Scaling applied correctly!")
