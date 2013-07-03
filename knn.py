import numpy as np
from scipy import spatial



def knn4bubbles(xys, dxys):
	"""
	Required

	*xys:* [list | np.array] 
		A list or np.array of coordinates in x, y and size.

	*dxys:* [list | np.array]
		A list or np.array of accompanying widths (a.k.a. errors). 
	"""
	tree = spatial.KDTree(np.array(xys))
	for i in xrange(len(xys)):
