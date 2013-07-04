import numpy as np
from scipy import spatial


# Testing
# Import data from CSV
path_to_csv = "sample_bubbles.csv"
bubbles = np.genfromtxt(path_to_csv, dtype=float, delimiter=',')

def knn4bubbles(xys, dxys):
	"""
	Required

	*xys:* [np.array] 
		An np.array of coordinates in x, y and size.

	*dxys:* [np.array]
		An np.array of accompanying errors. 
	"""

	# Vertical splitting
	x, y, s = np.vsplit(xys, 3)
	xy = np.column_stack([x, y])

	i = 0
	stop = False
	while stop == False:
		# Generating tree
		tree = spatial.KDTree(np.array(xy))

		# Querying tree for nearest neighbours within distance of bubble radius
		distance, index = tree.query(xy[i], distance_upper_bound=s[i], k=20)

		if distance.shape[0] > 1:
			pass


		i += 1
	
	if len(xy) <= 1:
		stop == True
