import numpy as np
import csv
import math
import random
import time
import datetime
# import pandas

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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
B = np.column_stack([ X[:, 0], X[:, 1], X[:, 0]/(X[:, 2]/2.0), X[:, 1]/(X[:, 3]/2.0), (X[:, 4]/X[:, 2])/0.5, X[:, 5]/45 ])
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

X = np.row_stack(mean_store.values())

#### This is where we need to include the next clustering algorithm. ####
X2 = np.column_stack([ X[:,0], X[:,1], X[:,2], X[:,3] ])
bandwidth = estimate_bandwidth(X2, quantile=0.05, n_samples=1000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X2)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print "End number of estimated clusters: %d" % n_clusters_
new_store = {}
for label in labels_unique:
    # Creating index where N == n, where n -> 0, 1, 2, ..., j
    index = labels == label
    new_store[label] = X[index]

# MeanShift clusters
new_mean_store = {}
for key in new_store.keys():
    new_mean_store[key] = get_mean_bubble(new_store[key])

##############################################################################
# Plot result
print('Drawing output...')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from pylab import figure, show, rand
from matplotlib.patches import Ellipse

## Create 'canvas' for map
fig = pl.figure(figsize=(7, 7))
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(x2, x1)
ax.set_ylim(y1, y2)
colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

#Plot raw data
for x in Xraw:
    e = Ellipse(xy=[x[0], x[1]], width=x[2], height=x[3], angle=x[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.01)
    e.set_facecolor([0.1, 0.1, 0.1])
    # ax.add_artist(e)

# Plot DBSCAN cluster list
iter_mean_store = mean_store.itervalues()

for x in iter_mean_store:
    e = Ellipse(xy=[x[0], x[1]], width=x[2], height=x[3], angle=x[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.50)
    e.set_facecolor('green')
    e.set_edgecolor('black')
    # e.set_lw(3)
    ax.add_artist(e)

# Plot final list
new_iter_mean_store = new_mean_store.itervalues()

for x in new_iter_mean_store:
    e = Ellipse(xy=[x[0], x[1]], width=x[2], height=x[3], angle=x[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.75)
    e.set_facecolor('red')
    e.set_edgecolor('black')
    # e.set_lw(3)
    ax.add_artist(e)

ts = time.time()
stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
pl.savefig("output/test"+stamp+".png", dpi=300, bbox_inches='tight')
pl.close()
