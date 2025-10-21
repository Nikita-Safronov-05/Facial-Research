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
for (seed in 1:100) {
  set.seed(seed)

  # Create per-seed output folder inside 'processed'
  out_dir <- file.path("processed", sprintf("sim_2000_d4_%d", seed))
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  X = (mvrnorm(n2, mu, x_sigma)) 

  means <- c(rep(1.5,p))
  sds <- c(rep(1,p))
  X <- round(X * sds + means)
  X[X < 0] <- 0
  X[X > 3] <- 3

  # Save simulated features matrix X for downstream analysis (to per-seed folder)
  write.csv(X, file.path(out_dir, "X.csv"), row.names = FALSE)


  # set the truth: 1,101,201,301
  x1 = matrix(X[,1], ncol = 1)
  x2 = matrix(X[,101], ncol = 1)
  x101 = matrix(X[,301], ncol = 1)
  x102 = matrix(X[,201], ncol = 1)

  # Calculate truth functions (keep exponentials)
  x.truth =  x1^2 + x1 * x2 * exp(x102)
  x.truth1 = 0.5 * exp(x101) * (x2)
  x.truth2 = 0.2 * (x1 * exp(x101) + x1)
  x.truth3 =  0.5 * exp(x1) * x2 + ((x101))
  x.truth4 =  x1^2 * exp(x101) 

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

  # Save simulated eigenvalues to a separate file to preserve original data (per-seed folder)
  save(simulated.eigenvalues, file = file.path(out_dir, "eigvals.RData"))

  # Also save as CSV for easy inspection (per-seed folder)
  write.csv(simulated.eigenvalues, file.path(out_dir, "eigvals.csv"), row.names = FALSE)
}