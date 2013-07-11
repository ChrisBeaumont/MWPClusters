#!/usr/bin/env python
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import os
import datetime
import time

with open('DR1.csv') as f:
  DR1 = np.loadtxt(f, delimiter=",", dtype={'names':('id', 'cat_id', 'lat', 'lon', 'inner_x_diameter', 'inner_y_diameter', 'outer_x_diameter', 'angle', 'dispersion'), 'formats':('<i8', 'S18', 'float', 'float', 'float', 'float', 'float', 'float', 'float')}, skiprows=1)

with open('DR2.csv') as f:
  DR2 = np.loadtxt(f, delimiter=",", dtype={'names':('lon', 'lat', 'width', 'height', 'thick', 'angle', 'score'), 'formats':('float', 'float', 'float', 'float', 'float', 'float', 'float')})

ts = time.time()
stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
directory = "output/charts/"
if not os.path.exists(directory):
	os.makedirs(directory)

properties = [['lon', 100], ['lat',20], ['angle',10]]
for prop in properties:
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	n, bins, patches = ax1.hist([DR1[prop[0]],DR2[prop[0]]], bins=prop[1], normed=1, color=['crimson', 'burlywood'], histtype='bar', rwidth=0.8)
	plt.savefig(directory+prop[0]+"_"+stamp+".png", dpi=300, bbox_inches='tight')
	plt.close()
