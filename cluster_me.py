import numpy as np
import csv
import math
import random
import time
import datetime
# import pandas

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def w_mean(data,x_index,w_index):
    return np.mean(np.sum(data[x_index,:]*data[w_index,:])/np.sum(data[w_index,:]))

def get_mean_bubble(data,score):
    lon = w_mean(data,0,6)
    lat = w_mean(data,1,6)
    width = w_mean(data,2,6)
    height = w_mean(data,3,6)
    thick = w_mean(data,4,6)
    angle = w_mean(data,5,6)
    return np.array([lon, lat, width, height, thick, angle, score])

# Import data from CSV
path_to_csv = "sample_bubbles.csv"
bubbles = np.genfromtxt(path_to_csv, dtype=float, delimiter=',')

X = np.delete(bubbles,0,0)
Xraw = X
x1, x2 = min(X[:, 0]), max(X[:, 0])
y1, y2 = min(X[:, 1]), max(X[:, 1])

# Xp = np.column_stack([ X[:,0]/0.1, X[:,1]/0.1, X[:,2]/0.01, X[:,3]/0.01, X[:,5]/30 ])

bubble_list = []
i = 0
exit=0
old_count = 0
new_count = 1
while (exit<3):
    # Compute DBSCAN
    if i==0:
        lim=3
    else:
        lim=1
    B = np.column_stack([ X[:,0], X[:,1], X[:,0]/(X[:,2]/2.0), X[:,1]/(X[:,3]/2.0), (X[:,4]/X[:,2])/0.5, X[:,5]/45 ])
    db = DBSCAN(eps=1.0, min_samples=lim).fit(B)
    core_samples = db.core_sample_indices_
    components = db.components_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    old_count = new_count
    new_count = n_clusters_
    if old_count == new_count:
        exit+=1
    else:
        exit=0
    print('Iteration %d predicting %d nodes (delta=%d)' % (i, new_count,(new_count-old_count)))

    unique_labels = set(labels)

    bubble_list = []
    for k in unique_labels:
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples if labels[index] == k]
        first = 1
        for index in class_members:
            if k != -1:
                if first==0:
                    ce = np.column_stack([ce, X[index]])
                else:
                    ce = np.column_stack([X[index]])
                    first = 0
        
        score = np.sum(ce[6,:])
        if score >= 10:
            # print('Node of %d drawings and score %d with std_dev %f' % (ce[1,:].size, score, np.std(ce[0,:])))
            newb = get_mean_bubble(ce,score)
            bubble_list.append(newb)

    X = np.array(bubble_list)
    print('X length is', X.shape)
    i+=1

##############################################################################
# Plot result
print('Drawing output...')
import pylab as pl
from pylab import figure, show, rand
from matplotlib.patches import Ellipse

## Create 'canvas' for map
fig = figure(figsize=(7,7), dpi=300)
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(x2, x1)
ax.set_ylim(y1, y2)
colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

#Plot raw data
for x in Xraw:
    e = Ellipse(xy=[x[0],x[1]], width=x[2], height=x[3], angle=x[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.01)
    e.set_facecolor([0.1,0.1,0.1])
    ax.add_artist(e)

#Plot final list
for x in bubble_list:
    e = Ellipse(xy=[x[0],x[1]], width=x[2], height=x[3], angle=x[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.33)
    e.set_facecolor(rand(3))
    e.set_edgecolor('black')
    # e.set_lw(3)
    ax.add_artist(e)

ts = time.time()
stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
pl.savefig("output/test"+stamp+".png", dpi=300)

