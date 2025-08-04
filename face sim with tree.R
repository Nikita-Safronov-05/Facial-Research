.libPaths("C:/Users/safro/AppData/Local/R/win-library/4.5.1")
library(MASS)
library(Matrix)
library(tidyverse)
library(glmnet)
library(RSpectra)
library(factoextra)
library(BBmisc)


# need to load "pc for face data" to get all pc information before the simulation
setwd("C:/Users/safro/VSCODETESTER/facial_research")
load("given_data/pc_for_face_data.RData")

ls()
# Dimensions and sample size
# predictor size
p = 1000
#response size
q = 5
# size of each face 7160
n1 = 7160
# sample size 2342
nn = 2342


# covariance matrix of X
mu = matrix(0, p, 1)
rho = 0.8
mm = 1
#T = seq(0, 1, len = 50)
ar1_cor = function(nn, mm,rho) {
  exponent = abs(matrix(1:nn - 1, nrow = nn, ncol = nn, byrow = TRUE) - (1:nn - 1))
  L = rho^exponent
  diag(L) = mm
  L
}
x_sigma = ar1_cor(p, mm, rho) 

# generate sample x from N(0, x_sigma) 
n2 = 2000  # Increased to 2000 faces
delta = 4  # Much smaller than 20

# Set seed for reproducibility
set.seed(42)

X = (mvrnorm(n2, mu, x_sigma)) 

means <- c(rep(1.5,p))
sds <- c(rep(1,p))
X <- round(X * sds + means)
X[X < 0] <- 0
X[X > 3] <- 3


# set the truth: 1,101,201,301
x1 = matrix(X[,1], ncol = 1)
x2 = matrix(X[,101], ncol = 1)
x101 = matrix(X[,301], ncol = 1)
x102 = matrix(X[,201], ncol = 1)

# Calculate truth functions (keep exponentials)
x.truth =  x1^2 + x1 * x2 * exp(x102)
x.truth1 = 0.5 * exp(x101) * (x2)
x.truth2 = 0.2 * (x1 * exp(x101) + x1)
x.truth3 =  0.5*exp(x1) * x2 + ((x101))
x.truth4 =  x1^2 * exp(x101) 

# # Normalize each truth function to reasonable range (-5 to +5)
# normalize_truth <- function(truth_vec) {
#   truth_scaled <- scale(truth_vec)[,1]  # Z-score normalization
#   truth_scaled * 2.5  # Scale to roughly ±5 range
# }

# x.truth_norm <- normalize_truth(x.truth)
# x.truth1_norm <- normalize_truth(x.truth1)
# x.truth2_norm <- normalize_truth(x.truth2)
# x.truth3_norm <- normalize_truth(x.truth3)
# x.truth4_norm <- normalize_truth(x.truth4)

# Now use normalized versions with delta
PC1 = pc.pcs[1:n2, 1]
PC2 = pc.pcs[1:n2, 2]
PC3 = pc.pcs[1:n2, 3]
PC4 = pc.pcs[1:n2, 4]
PC5 = pc.pcs[1:n2, 5]
PC1.new = PC1 + 2*delta*x.truth + matrix(rnorm(n2,0,1),n2,1)
PC2.new = PC2 + 1*delta*x.truth1 + matrix(rnorm(n2,0,1),n2,1)
PC3.new = PC3 + 1*delta*x.truth2 + matrix(rnorm(n2,0,1),n2,1)
PC4.new = PC4 + 2*delta*x.truth3 + matrix(rnorm(n2,0,1),n2,1)
PC5.new = PC5 + 1*delta*x.truth4 + matrix(rnorm(n2,0,1),n2,1)

# Create matrix for simulated eigenvalues (2000 faces x 5 PCs)
simulated.eigenvalues <- matrix(0, nrow = n2, ncol = 5)

# Fill in the simulated values using the noisy versions
simulated.eigenvalues[,1] <- PC1.new
simulated.eigenvalues[,2] <- PC2.new
simulated.eigenvalues[,3] <- PC3.new
simulated.eigenvalues[,4] <- PC4.new
simulated.eigenvalues[,5] <- PC5.new

# Save simulated eigenvalues to a separate file to preserve original data
save(simulated.eigenvalues, file = "given_data/sim_eigvals_2000_d=4.RData")

# Also save as CSV for easy inspection
write.csv(simulated.eigenvalues, "given_data/sim_eigvals_2000_d=4.csv", row.names = FALSE)

# PC1.tree = matrix(0, nrow = n2, ncol = 1)
# PC2.tree = matrix(0, nrow = n2, ncol = 1)
# PC3.tree = matrix(0, nrow = n2, ncol = 1)
# PC4.tree = matrix(0, nrow = n2, ncol = 1)
# PC5.tree = matrix(0, nrow = n2, ncol = 1)

# # use trees to connect PC and X
# spv = 1.5
# index <- c(1,101,301,201)
# count <- rep(0,5)
# tree.feature <- list(NA)
# for(j in 1:2){
#   i <- sample(index,3, replace = FALSE)
#   tree.feature[[j]] <- i
#   count[length(i[1:2])] <- count[length(i[1:2])] + 1
#   count[length(i[1:3])] <- count[length(i[1:3])] + 1
#   if(i[2] == 1){
#     PC1.tree = PC1.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC1.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC1.tree
#     PC2.tree = PC2.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC2.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC2.tree
#     PC3.tree = PC3.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC3.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC3.tree
#     PC4.tree = PC4.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC4.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC4.tree
#     count[1] <- count[1] + 1
#     count[2] <- count[2] + 1
#     count[3] <- count[3] + 1
#     count[4] <- count[4] + 1
#   } else {
#     if (i[2] == 101) {
#       PC3.tree = PC3.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC3.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC3.tree
#       PC2.tree = PC2.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC2.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC2.tree
#       PC4.tree = PC4.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC4.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC4.tree
#       PC5.tree = PC5.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC5.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC5.tree
#       count[5] <- count[5] + 1
#       count[2] <- count[2] + 1
#       count[3] <- count[3] + 1
#       count[4] <- count[4] + 1
#     } else {
#       if (i[2] == 301) {
#         PC1.tree = PC1.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC1.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC1.tree
#         PC3.tree = PC3.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC3.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC3.tree
#         PC4.tree = PC4.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC4.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC4.tree
#         PC5.tree = PC5.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC5.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC5.tree
#         count[1] <- count[1] + 1
#         count[5] <- count[5] + 1
#         count[3] <- count[3] + 1
#         count[4] <- count[4] + 1
#       } else {
#         if (i[2] == 201) {
#           PC1.tree = PC1.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC1.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC1.tree
#           PC4.tree = PC4.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC4.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC4.tree
#           PC5.tree = PC5.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC5.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC5.tree
#           PC2.tree = PC2.new*I(X[,i[1]]< spv)*I(X[,i[2]]< spv) + PC2.new.r*I(X[,i[1]]< spv)*I(X[,i[2]]>= spv) + PC2.tree
#           count[1] <- count[1] + 1
#           count[2] <- count[2] + 1
#           count[5] <- count[5] + 1
#           count[4] <- count[4] + 1
#         }
#       }
#     }
#   }
#   if(i[3] == 1){
#     PC1.tree = PC1.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC1.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC1.tree
#     PC2.tree = PC2.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC2.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC2.tree
#     PC3.tree = PC3.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC3.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC3.tree
#     PC4.tree = PC4.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC4.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC4.tree
#     count[1] <- count[1] + 1
#     count[2] <- count[2] + 1
#     count[3] <- count[3] + 1
#     count[4] <- count[4] + 1
#   } else {
#     if (i[3] == 101) {
#       PC3.tree = PC3.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC3.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC3.tree
#       PC2.tree = PC2.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC2.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC2.tree
#       PC4.tree = PC4.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC4.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC4.tree
#       PC5.tree = PC5.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC5.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC5.tree
#       count[5] <- count[5] + 1
#       count[2] <- count[2] + 1
#       count[3] <- count[3] + 1
#       count[4] <- count[4] + 1
#     } else {
#       if (i[3] == 301) {
#         PC1.tree = PC1.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC1.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC1.tree
#         PC3.tree = PC3.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC3.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC3.tree
#         PC4.tree = PC4.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC4.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC4.tree
#         PC5.tree = PC5.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC5.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC5.tree
#         count[1] <- count[1] + 1
#         count[5] <- count[5] + 1
#         count[3] <- count[3] + 1
#         count[4] <- count[4] + 1
#       } else {
#         if (i[3] == 201) {
#           PC1.tree = PC1.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC1.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC1.tree
#           PC2.tree = PC2.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC2.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC2.tree
#           PC4.tree = PC4.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC4.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC4.tree
#           PC5.tree = PC5.new*I(X[,i[1]]>= spv)*I(X[,i[3]]< spv) + PC5.new.r*I(X[,i[1]]>= spv)*I(X[,i[3]]>= spv) + PC5.tree
#           count[1] <- count[1] + 1
#           count[2] <- count[2] + 1
#           count[5] <- count[5] + 1
#           count[4] <- count[4] + 1
#         } 
#       }
#     }
#   }
  
# }

# PC1.tree = PC1.tree / count[1]
# PC2.tree = PC2.tree / count[2]
# PC3.tree = PC3.tree / count[3]
# PC4.tree = PC4.tree / count[4]
# PC5.tree = PC5.tree / count[5]

# pc.pcsnew = matrix(rep(pc$x[230,],n2),n2, byrow = TRUE)
# pc.pcsnew[,1] = PC1.tree
# pc.pcsnew[,2] = PC2.tree
# pc.pcsnew[,3] = PC3.tree
# pc.pcsnew[,4] = PC4.tree
# pc.pcsnew[,5] = PC5.tree

# x.pca1new = t(t(pc.pcsnew %*% t(pc$rotation)) * pc$scale + pc$center)

# Y = as.data.frame(x.pca1new)

# # dimension reduction
# system.time(
#   # Compute PCA
#   pcy <- prcomp(Y, scale = TRUE))

# Y <- as.data.frame(pcy$x[,1:5])









