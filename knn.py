import numpy as np
from scipy import spatial
import csv
import math
import random
import time
import datetime

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def knn4bubbles(data):
	"""
	Required

	*xys:* [np.array] 
		An np.array of coordinates in x, y and size.

	*dxys:* [np.array]
		An np.array of accompanying errors. 
	"""

	s = np.sqrt(data[:,2] ** 2 + data[:,3] ** 2)
	xys = np.column_stack([data[:,0], data[:,1], s, data[:,2], data[:,3], data[:,4], data[:,5]])

	i = 0
	container = {}

	while True:
	# while i < 125:

		# Formatting arrays
		x, y, s = xys[:,0], xys[:,1], xys[:,2]
		xy = np.column_stack([x, y])

		# Generating tree
		tree = spatial.KDTree(np.array(xy))

		# Querying tree for nearest neighbours within distance of bubble radius
		distance, index = tree.query(xy[0], distance_upper_bound=s[0]/3., k=20)

		# Comparing bubble sizes and indexing
		# print(s)
		sizes_ratio = s * 1.0 / s[0]
		upper = sizes_ratio < 1.33
		lower = sizes_ratio > 0.66
		size_index = np.where(upper & lower)

		# If there are sources closer than bubble radius
		# and they have similar bubble size them clump them.
		if distance.shape[0] > 1:
			a = set(size_index[0])
			b = set(index)
			ab = a.intersection(b)
			if len(ab) > 0:
				container[i] = xys[list(ab)]
			else:
				container[i] = xys[i]

		bool_index = np.zeros(xys.shape[0]).astype(int) + 1
		bool_index[list(ab)] = 0
		xys = xys[bool_index.astype(bool)]

		i += 1

		# When the xys array contains no rows the
		# return function returns mbubble
		# and stops the while loop. 
		mbubbles = []
		if xys.shape[0] < 2:
			for bubbles in container.itervalues():
				mbubbles.append(np.mean(bubbles, axis=0))
			return np.array(mbubbles)

def w_mean(data, x_index, w_index):
    return np.mean(np.sum(data[:, x_index]*data[:, w_index])/np.sum(data[:, w_index]))

def get_mean_bubble(data):
    lon = w_mean(data, 0, 6)
    lat = w_mean(data, 1, 6)
    width = w_mean(data, 2, 6)
    height = w_mean(data, 3, 6)
    thick = w_mean(data, 4, 6)
    angle = w_mean(data, 5, 6)
    score = np.sum(data[:,6])
    return np.array([lon, lat, width, height, thick, angle, score])

# Import data from CSV
path_to_csv = "sample_bubbles.csv"
bubbles = np.genfromtxt(path_to_csv, dtype=float, delimiter=',')

X = np.delete(bubbles, 0, 0)
Xraw = X
x1, x2 = min(X[:, 0]), max(X[:, 0])
y1, y2 = min(X[:, 1]), max(X[:, 1])

# DBSCAN component
### THESE ARE THE META PARAMETERS
###
B = np.column_stack([ X[:, 0], X[:, 1], X[:, 0]/(X[:, 2]/1.0), X[:, 1]/(X[:, 3]/1.0), (X[:, 4]/X[:, 2])/0.50, X[:, 5]/30 ])
###
###
db = DBSCAN(eps=1, min_samples=3).fit(B)
core_samples = db.core_sample_indices_
components = db.components_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Cleaning up labels for iteration
unique_labels = np.unique(labels)
unique_labels = labels[labels > -1]

# store is a dictionary that holds all the bubble members of each DBSCAN
# defined cluster. 
store = {}
for label in unique_labels:
    # Creating index where N == n, where n -> 0, 1, 2, ..., j
    index = labels == label
    store[label] = X[index]

# DBSCAN clusters added together to single components
mean_store = {}
for key in store.keys():
    mean_store[key] = get_mean_bubble(store[key])

data = np.row_stack(mean_store.values())

# data = np.loadtxt('./sample.csv', skiprows=1, delimiter=',')
d = knn4bubbles(data)

print('Drawing output...')
import matplotlib.pylab as pl
from pylab import figure, show, rand
from matplotlib.patches import Ellipse
import time
import datetime

## Create 'canvas' for map
fig = pl.figure(figsize=(7, 7))
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(data[:,0].max()+0.2, data[:,0].min()-0.2)
ax.set_ylim(-1, 1)

#Plot raw data
for xys in data:
    e = Ellipse(xy=[xys[0], xys[1]], width=xys[2], height=xys[3], angle=xys[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.1)
    e.set_facecolor('black')
    e.set_edgecolor('none')
    ax.add_artist(e)

# Plot cluster list
for xys in d:
    e = Ellipse(xy=[xys[0], xys[1]], width=xys[3], height=xys[4], angle=xys[6])
    e.set_clip_box(ax.bbox)
    e.set_alpha(1)
    e.set_facecolor([0.1, 0.1, 0.1])
    e.set_facecolor('none')
    e.set_edgecolor('red')
    # e.set_lw(3)
    ax.add_artist(e)

# pl.show()
ts = time.time()
stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
pl.savefig("output/knn_test_"+stamp+".png", dpi=300, bbox_inches='tight')
pl.close()

