import numpy as np
from numpy import linalg #for eigenvalue calculations
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt #for plots
import plotly.express as px # for interactive plots
from sklearn.preprocessing import StandardScaler

#2. Import Data

# load wiki data into dataframe
orig_data = pd.read_csv('wiki_data.csv',index_col=0)

# sample 1000 rows for faster testing
orig_data = orig_data.sample(n=1000, random_state=12)

# clean up page names
orig_data.index = orig_data.index.str.replace('_all-access_all-agents', '', regex=False)
#orig_data.fillna(0,inplace=True)
print(orig_data.head(10))

dates = orig_data.columns.tolist()

# print sample to briefly explain dataset
print('samples: ',orig_data.shape[0])
print('features: ',orig_data.shape[1])

# grab original values for later
max_dates_orig = orig_data.idxmax(axis=1).tolist()
max_hits_orig = orig_data.max(axis=1).tolist()
hit_range_orig = (orig_data.max(axis=1) - orig_data.min(axis=1)).tolist()

# normalize so more popular pages don't have higher impact
hits = orig_data.reset_index(drop=True)

scaler = StandardScaler()
hits_scaled = scaler.fit_transform(hits)

wiki_data = pd.DataFrame(hits_scaled,columns=dates,index=orig_data.index)

#3. Compute distances for each pair

# build distance_matrix with size num_points x num_points
dist_mat = cdist(wiki_data,wiki_data,metric='Chebyshev')

#4. Create kernels to build affinity matrix

# define sigma as avg of all distances in D[i,j] -> controls scale of kernel
sigma = np.median(dist_mat)
print('sigma: ',sigma)

# initialize affinity matrix A with size num_points x num_points
aff_mat = np.exp(-dist_mat**2 / (2 * sigma**2))
np.fill_diagonal(aff_mat,0) # so diags (self-loops) won't occur

#5. Normalize to build Markov matrix

# get row sums from affinity matrix
row_sums = np.sum(aff_mat,axis=1)

# to avoid div by zero error in next step
row_sums[row_sums == 0] = 1e-10

# normalize to 1
mark_mat = aff_mat / row_sums[:,np.newaxis]

#6. Compute Eigenvalues & Eigenvectors

# use np.linalg.eig to get e_vals & e_vecs from M[i,j]
evals,evecs = np.linalg.eig(mark_mat)

# sort pairs by putting e_vals in descending order
sorted_pairs = np.argsort(-evals)
evals = evals[sorted_pairs]
evecs = evecs[:,sorted_pairs]

print('First five eigenvalues: ',evals[:5])

# drop first value = 1
evals = evals[1:]
evecs = evecs[:,1:]

# plot first 20 eigenvalues
evals_toplot = evals[:20]
plt.plot(range(len(evals_toplot)),evals_toplot,'ko')
plt.show()

#7. Choose Diffusion Coordinates

diff_time = 1

# can see significant dropoff after 2nd point
num_coords = 2

print('number of coords: ',num_coords)

diff_evals = evals[:num_coords]
diff_evecs = evecs[:,:num_coords]

# scale evecs by evals^t
diff_coords = diff_evecs * (diff_evals**diff_time)

#now there are only n dimensions instead of original dimensionality

# add more info for analysis
max_dates = wiki_data.idxmax(axis=1).tolist()
max_hits = wiki_data.max(axis=1).tolist()
hit_range = (wiki_data.max(axis=1) - wiki_data.min(axis=1)).tolist()

# convert to dataframes for plotly
if num_coords == 2:
    final_df = pd.DataFrame(diff_coords, columns=['C1','C2'])
elif num_coords == 3:
    final_df = pd.DataFrame(diff_coords, columns=['C1','C2','C3'])

final_df['Page'] = wiki_data.index
final_df['Max Date'] = max_dates
final_df['Max Hits'] = max_hits
final_df['Hit Range'] = hit_range

# for analysis
final_df['Max Hits Actual'] = max_hits_orig
final_df['Hit Range Actual'] = hit_range_orig

# build scatter plot
if num_coords == 2:
    fig = px.scatter(final_df, x='C1', y='C2',
                     color='Hit Range',
                     hover_name='Page',hover_data='Max Date',
                     title='Wikipedia Page Diffusion Map')
    fig.show()
elif num_coords == 3:
    fig = px.scatter_3d(final_df, x='C1', y='C2', z='C3',
                        color='Hit Range',
                        hover_name='Page', hover_data='Max Date',
                        title="Wikipedia Page Diffusion Map"
                        )
    fig.show()
else:
    print('Invalid number of coordinates')

filtered_df = final_df[final_df['Max Date'] != '7/6/2016']
print('Top Ten Coordinate 1')
top1 = filtered_df.nlargest(10,'C1')
print(top1[['Page','Max Hits Actual','Hit Range Actual','Max Date','C1']])

print('Bottom Ten Coordinate 1')
bottom1 = filtered_df.nsmallest(10,'C1')
print(bottom1[['Page','Max Hits Actual','Hit Range Actual','Max Date','C1']])

print('Top Ten Coordinate 2')
top2 = final_df.nlargest(10,'C2')
print(top2[['Page','Max Hits Actual','Hit Range Actual','Max Date','C2']])

print('Bottom Ten Coordinate 2')
bottom2 = final_df.nsmallest(10,'C2')
print(bottom2[['Page','Max Hits Actual','Hit Range Actual','Max Date','C2']])


# coordinate 1 = impact of date on hits
# coordinate 2 = pattern of dates on hits
