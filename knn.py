import numpy as np
from scipy import spatial



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
	xytemp = np.copy(xy)


	# Need to store which bubbles are merged and not
	merged = []

	i = 0
	stop = False
	while stop == False:
		# Generating tree
		tree = spatial.KDTree(np.array(xy))

		distance, index = tree.query(xytemp[i])

		i += 1
	if len(xy) <= 1:
		stop == True
