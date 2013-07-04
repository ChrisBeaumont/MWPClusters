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

	# columns 2,3 are w and h of bubble

	# Vertical splitting
	i = 0
	stop = False
	container = {}
	while stop == False:
	# while i < 125:
		# Formatting arrays
		x, y, s = np.hsplit(xys, 3)
		xy = np.column_stack([x, y])

		# Generating tree
		tree = spatial.KDTree(np.array(xy))

		# Querying tree for nearest neighbours within distance of bubble radius
		distance, index = tree.query(xy[i], distance_upper_bound=s[i], k=20)

		# Comparing bubble sizes and indexing
		print(s)
		sizes_ratio = s * 1.0 / s[i]
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
		xys = xys[bool_index]

		if xys.shape[0] < 2:
			stop == True

		# debug 
		if i == 50:
			heck
		
		i += 1

	return container
