import csv
import math
import random
import time
import datetime
import pymysql
import os

import numpy as np
from scipy import spatial
from sklearn.cluster import DBSCAN

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def dbscan4bubbles(X,freedom_pos,freedom_size, freedom_rot):
	
	### DBSCAN META PARAMETERS
	# B = np.column_stack([ X[:, 0]/0.5, X[:, 1]/0.5, X[:, 0]/X[:, 2], X[:, 1]/X[:, 3], (X[:, 4]/X[:, 2])/0.33, X[:, 5]/45 ])
	B = np.column_stack([ X[:, 0]/freedom_pos, X[:, 1]/freedom_pos, (X[:, 0]/X[:, 2])/freedom_size, (X[:, 1]/X[:, 3])/freedom_size, (X[:, 4]/X[:,2])/freedom_size, X[:, 5]/freedom_rot ])
	###

	db = DBSCAN(eps=1, min_samples=5).fit(B)
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

	return np.row_stack(mean_store.values())

def knn4bubbles(data):
	"""
	Required

	*xys:* [np.array] 
		An np.array of coordinates in x, y and size.

	*dxys:* [np.array]
		An np.array of accompanying errors. 
	"""

	s = np.sqrt(data[:,2] ** 2 + data[:,3] ** 2)
	xys = np.column_stack([data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], s])

	i = 0
	container = {}

	while True:
	# while i < 125:

		# Formatting arrays
		x, y, s = xys[:,0], xys[:,1], xys[:,7]
		xy = np.column_stack([x, y])

		# Generating tree
		tree = spatial.KDTree(np.array(xy))

		# Querying tree for nearest neighbours within distance of bubble radius
		distance, index = tree.query(xy[0], distance_upper_bound=s[0]/2., k=20)

		# Comparing bubble sizes and indexing
		sizes_ratio = s * 1.0 / s[0]
		upper = sizes_ratio < 1.5
		lower = sizes_ratio > 0.5
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
				# mbubbles.append(np.mean(bubbles, axis=0))
				mbubbles.append(knn_get_mean_bubble(bubbles))
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

def knn_get_mean_bubble(data):
    lon = w_mean(data, 0, 6)
    lat = w_mean(data, 1, 6)
    width = w_mean(data, 2, 6)
    height = w_mean(data, 3, 6)
    thick = w_mean(data, 4, 6)
    angle = w_mean(data, 5, 6)
    score = np.sum(data[:,6])

    if width>=height:
    	Rin = width/2
    	rin = height/2
    	Rout = (width+thick)/2
    	rout = (height+thick)/2
    else:
    	rin = width/2
    	Rin = height/2
    	rout = (width+thick)/2
    	Rout = (height+thick)/2

    r_eff = 60*(math.sqrt(Rin*rin) + math.sqrt(Rout*rout))/2
    thickness = (math.sqrt(Rout*rout) - math.sqrt(Rin*rin))*60
    e = math.sqrt(Rin*Rin - rin*rin)/Rin

    if lon>180:
    	abslon=lon-360
    else:
    	abslon=lon

    return np.array([lon, lat, width, height, r_eff, thickness, e, angle, score, abslon])

### Import data from CSV
###
# path_to_csv = "sample_data/bubbles_26.csv"
# bubbles = np.genfromtxt(path_to_csv, dtype=float, delimiter=',')
###
###

### Import data from MySQL
###
step = 0.5
lons = drange(15, 25, step)
lon_range = step*2.0
score = 3
l=1.0
s=1.0
r=45
data = np.array([[0,0,0,0,0,0,0]])
db = pymysql.connect(host="localhost", user="root", passwd="", db="milkyway-development")
cur = db.cursor() 

for lon_c in lons:
	sql = "SELECT lon, lat, (inner_x_diameter*pixel_scale) as width, (inner_y_diameter*pixel_scale) as height, ((outer_x_diameter-inner_x_diameter)*pixel_scale) as thickness, (angle % 90) as angle, score as score, pixel_scale as scale FROM bubbles, zooniverse_users WHERE bubbles.user_id = zooniverse_user_id AND lon BETWEEN "+str(lon_c-lon_range/2)+" AND "+str(lon_c+lon_range/2)+" AND (inner_x_diameter/pixel_scale) >= 20 AND (inner_y_diameter/pixel_scale) >= 20 AND score >= "+str(score)+";"
	cur.execute(sql)
	sqlres = []
	for row in cur.fetchall():
		sqlres.append(row)

	bubbles = np.array(sqlres)
	bubbles = bubbles.astype(np.float)
	###
	###

	X = np.delete(bubbles, 0, 0)
	user_drawings = X
	x1, x2 = min(X[:, 0]), max(X[:, 0])
	y1, y2 = min(X[:, 1]), max(X[:, 1])

	print('Scanning region around '+str(lon_c))
	res = dbscan4bubbles(X,l,s,r)
	data = np.concatenate((data,res))

db.close()
data = np.delete(data, 0, 0)
d = knn4bubbles(data)

print ('Saving output...')
np.savetxt("DR2.csv", d, delimiter=",")

print('Drawing output...')
import matplotlib.pylab as pl
from pylab import figure, show, rand
from matplotlib.patches import Ellipse
import time
import datetime

## Create 'canvas' for map
fig = pl.figure(figsize=(15, 15))
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(data[:,0].max()+0.2, data[:,0].min()-0.2)
ax.set_ylim(-1, 1)

#Plot original data
# for xys in user_drawings:
#     e = Ellipse(xy=[xys[0], xys[1]], width=xys[2], height=xys[3], angle=xys[5])
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.01)
#     e.set_facecolor('black')
#     e.set_edgecolor('none')
#     ax.add_artist(e)

#Plot DBSCAN data
for xys in data:
    e = Ellipse(xy=[xys[0], xys[1]], width=xys[2], height=xys[3], angle=xys[5])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.33)
    e.set_facecolor('blue')
    e.set_edgecolor('none')
    ax.add_artist(e)

# Plot KNN-clustered bubbles
for xys in d:
    e = Ellipse(xy=[xys[0], xys[1]], width=xys[2], height=xys[3], angle=xys[7])
    e.set_clip_box(ax.bbox)
    e.set_alpha(1)
    e.set_facecolor([0.1, 0.1, 0.1])
    e.set_facecolor('none')
    e.set_edgecolor('red')
    # e.set_lw(3)
    ax.add_artist(e)
for xys in d:
    e = Ellipse(xy=[xys[0], xys[1]], width=xys[2]+(2*xys[5]/60), height=xys[3]+(2*xys[5]/60), angle=xys[7])
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.8)
    e.set_facecolor([0.1, 0.1, 0.1])
    e.set_facecolor('none')
    e.set_edgecolor('red')
    # e.set_lw(3)
    ax.add_artist(e)

ts = time.time()
stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
directory = "output/DR2/"
if not os.path.exists(directory):
    os.makedirs(directory)
pl.savefig(directory+"reduction_test_"+str(score)+"_"+str(l)+"_"+str(s)+"_"+str(r)+"_"+stamp+".png", dpi=300, bbox_inches='tight')
pl.close()

