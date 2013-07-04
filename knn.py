import numpy as np
from scipy import spatial


# Testing
# Import data from CSV
path_to_csv = "sample_bubbles.csv"
bubbles = np.genfromtxt(path_to_csv, dtype=float, delimiter=',')

def knn4bubbles(data):
	"""
	Required

	*xys:* [np.array] 
		An np.array of coordinates in x, y and size.

	*dxys:* [np.array]
		An np.array of accompanying errors. 
	"""

	s = np.sqrt(data[:,2] ** 2 + data[:,3] ** 2)
	xys = np.column_stack([data[:,0], data[:,1], s])

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


test = True

if test:
	print('Drawing output...')
	import matplotlib.pylab as pl
	from pylab import figure, show, rand
	from matplotlib.patches import Ellipse
	import time
	import datetime

	data = np.loadtxt('./sample.csv', skiprows=1, delimiter=',')

	d = knn4bubbles(data)

	## Create 'canvas' for map
	fig = pl.figure(figsize=(7, 7))
	ax = fig.add_subplot(111, aspect='equal')
	ax.set_xlim(data[:,0].max()+0.2, data[:,0].min()-0.2)
	ax.set_ylim(-1, 1)

	#Plot raw data
	for xys in data:
	    e = Ellipse(xy=[xys[0], xys[1]], width=xys[2], height=xys[2], angle=0)
	    e.set_clip_box(ax.bbox)
	    e.set_alpha(0.1)
	    e.set_facecolor('black')
	    e.set_edgecolor('none')
	    ax.add_artist(e)


	# Plot DBSCAN cluster list

	for xys in d:
	    e = Ellipse(xy=[xys[0], xys[1]], width=xys[2], height=xys[2], angle=0)
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

