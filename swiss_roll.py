#1. Import Libraries

import numpy as np
from numpy import linalg #for eigenvalue calculations
from sklearn.datasets import make_swiss_roll #for example #1
import matplotlib.pyplot as plt #for plotting

#2. Generate Data

n_samples = 1000

X,t = make_swiss_roll(n_samples=n_samples,noise=0.05,random_state=0)
# X = array of points, t = univariate position

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    X[:, 0], X[:, 1], X[:, 2], c=t, s=50, alpha=0.8
)
ax.set_title("Swiss Roll Scatter Plot")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1000", transform=ax.transAxes)

plt.show()
plt.clf()

#3. Compute Euclidian distances for each pair

#initialize distance_matrix with size num_points x num_points
dist_mat = np.zeros((n_samples,n_samples))

#fill with Euclidian distances between each point i,j
for i in range(n_samples):
    for j in range(i, n_samples):  # calc for i ≤ j only bc symmetry
        #D[i, j] = sqrt[(Xi, 0 - Xj, 0) ^ 2 + (Xi, 1 - Xj, 1) ^ 2 + ... + (Xi, n - Xj, n) ^ 2)]
        distance = np.linalg.norm(X[i] - X[j])
        dist_mat[i, j] = distance
        dist_mat[j, i] = distance  # symmetric property
# dist_mat now stores distances between points

#4. Create kernels to build affinity matrix

# define sigma as avg of all distances in D[i,j] -> controls scale of kernel
# ^ most common defn for sigma
sigma = np.median(dist_mat)
print('sigma = ',sigma)

# initialize affinity matrix A with size num_points x num_points
#A[i,j] = exp[-(D[i,j]^2)/(2*sigma)]
aff_mat = np.exp(-dist_mat**2 / (2 * sigma**2))

np.fill_diagonal(aff_mat,0) # so diags (self-loops) won't occur

#5. Normalize to build Markov matrix

# get row sums from affinity matrix
row_sums = np.sum(aff_mat,axis=1)

# to avoid div by zero error in next step
row_sums[row_sums == 0] = 1

# normalize to 1
mark_mat = aff_mat / row_sums[:,np.newaxis]

#6. Compute Eigenvalues & Eigenvectors

# use np.linalg.eig to get e_vals & e_vecs from M[i,j]
evals,evecs = np.linalg.eig(mark_mat)

# sort pairs by putting e_vals in descending order
sorted_pairs = np.argsort(-evals)
evals = evals[sorted_pairs]
evecs = evecs[:,sorted_pairs]

print('first five eigenvalues: ',evals[:5])

# drop first value = 1
evals = evals[1:]
evecs = evecs[:,1:]

# plot first 20 eigenvalues
evals_toplot = evals[:20]
plt.plot(range(len(evals_toplot)),evals_toplot,'ko')
plt.title('First 20 Eigenvalues')
plt.show()

# drop redundant first eigenvalue/eigenvector
evals = evals[1:]
evecs = evecs[:,1:]

#7. Choose Diffusion Coordinates

# trial and error
diff_time = 1

# keep first n e_vals (where any e_vals past nth are insignificantly small)
# determining n by cutting off any values < 0.05
num_coords = np.sum(evals >= 0.05)

# almost certainly within the first 20, so cropping the array there for speed
print('number of coords: ',num_coords)

# drop all after num_coords
diff_evals = evals[:num_coords]
diff_evecs = evecs[:,:num_coords]

# scale evecs by evals^t
diff_map = diff_evecs * (diff_evals**diff_time)

#now there are only n dimensions instead of original dimensionality

# 8. Plot diffusion coordinates
if num_coords == 2:
    plt.scatter(diff_map[:, 0], diff_map[:, 1])
    plt.xlabel("Coordinate 1")
    plt.ylabel("Coordinate 2")
    plt.title("Swiss Roll Diffusion Map")
    plt.show()
elif num_coords == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(diff_map[:, 0], diff_map[:, 1], diff_map[:, 2])
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.set_zlabel("Coordinate 3")
    ax.set_title("Swiss Roll Diffusion Map")
    plt.show()
elif num_coords == 4:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc=ax.scatter(
        diff_map[:, 0], diff_map[:, 1], diff_map[:, 2],
        c = diff_map[:,3], cmap = 'viridis')
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.set_zlabel("Coordinate 3")
    cbar = fig.colorbar(sc)
    cbar.set_label("Coordinate 4")
    ax.set_title("Swiss Roll Diffusion Map")
    plt.show()
