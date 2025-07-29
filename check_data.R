load('given_data/facesim100000.RData')
cat('Objects in workspace:\n')
print(ls())
cat('\n\nDimensions of key objects:\n')
cat('pc.center length:', length(pc.center), '\n')
cat('pc.eigenvectors dim:', dim(pc.eigenvectors), '\n')
cat('pc.pcs dim:', dim(pc.pcs), '\n')
cat('simulated.eigenvalues dim:', dim(simulated.eigenvalues), '\n')

cat('\nFirst few values of pc.pcs[1,]:\n')
print(pc.pcs[1,1:10])
cat('\nFirst few values of simulated.eigenvalues[1,]:\n')
print(simulated.eigenvalues[1,])

# Check the range of simulated eigenvalues vs original
cat('\nRange of simulated eigenvalues:\n')
print(range(simulated.eigenvalues))
cat('\nRange of original first 5 eigenvalues:\n')
print(range(pc.pcs[1,1:5]))

# Check if the reconstruction formula makes sense
cat('\nFirst reconstructed face dimensions after reconstruction:\n')
test_face <- pc.center + pc.eigenvectors %*% pc.pcs[1,]
cat('Length of reconstructed face:', length(test_face), '\n')
cat('Should be 21480 (7160 vertices * 3 coordinates)\n')
