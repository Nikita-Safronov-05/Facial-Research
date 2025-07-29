setwd("C:/Users/safro/VSCODETESTER/facial_research")
load('given_data/facesim100000.RData')

# Check basic structure
print("=== DATA STRUCTURE CHECK ===")
print(paste("pc.center length:", length(pc.center)))
print(paste("pc.eigenvectors dimensions:", paste(dim(pc.eigenvectors), collapse=" x ")))
print(paste("pc.pcs dimensions:", paste(dim(pc.pcs), collapse=" x ")))
print(paste("simulated.eigenvalues dimensions:", paste(dim(simulated.eigenvalues), collapse=" x ")))

# Check ranges - this is crucial for identifying distortion issues
print("\n=== EIGENVALUE RANGES ===")
print("Original first 5 eigenvalues range:")
print(range(pc.pcs[1,1:5]))
print("Simulated eigenvalues range:")
print(range(simulated.eigenvalues))

# Check if simulated values are realistic
print("\n=== EIGENVALUE COMPARISON ===")
print("Original first 5 eigenvalues (first row):")
print(pc.pcs[1,1:5])
print("First simulated eigenvalues:")
print(simulated.eigenvalues[1,])

# Test reconstruction with original values first
print("\n=== RECONSTRUCTION TEST ===")
original_face <- pc.center + pc.eigenvectors %*% pc.pcs[1,]
print(paste("Original reconstruction length:", length(original_face)))

# Test with simulated values
all_eigenvalues_test <- pc.pcs[1,]
all_eigenvalues_test[1:5] <- simulated.eigenvalues[1,]
simulated_face <- pc.center + pc.eigenvectors %*% all_eigenvalues_test
print(paste("Simulated reconstruction length:", length(simulated_face)))

# Check for extreme values that could cause distortion
print("\n=== DISTORTION CHECK ===")
print("Range of original reconstruction:")
print(range(original_face))
print("Range of simulated reconstruction:")
print(range(simulated_face))

# Check if simulated eigenvalues are orders of magnitude different
ratio_check <- abs(simulated.eigenvalues[1,]) / abs(pc.pcs[1,1:5])
print("Ratio of simulated to original eigenvalues:")
print(ratio_check)
