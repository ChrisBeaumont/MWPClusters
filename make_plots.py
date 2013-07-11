#!/usr/bin/env python
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import os
import datetime
import time

with open('DR1.csv') as f:
  DR1 = np.loadtxt(f, delimiter=",", dtype={'names':('id', 'cat_id', 'lat', 'lon', 'inner_x_diameter', 'inner_y_diameter', 'outer_x_diameter', 'r_eff', 'thickness', 'e', 'angle', 'dispersion', 'abslon'), 'formats':('<i8', 'S18', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float')}, skiprows=1)

with open('DR2.csv') as f:
  DR2 = np.loadtxt(f, delimiter=",", dtype={'names':('lon', 'lat', 'width', 'height', 'r_eff', 'thickness', 'e', 'angle', 'score', 'abslon'), 'formats':('float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float')})

ts = time.time()
stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
directory = "output/charts/"
if not os.path.exists(directory):
	os.makedirs(directory)

charts = [['abslon', 100, 'linear',[65,-65]], ['lat', 20, 'linear',[-1,1]], ['e', 20, 'log', [0,1]], ['r_eff', 30, 'log', [0,20]], ['thickness', 20, 'log', [0,10]]]
for prop in charts:
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	# ax1.xlim(prop[3])
	ax1.set_yscale(prop[2])
	plt.xlim(prop[3])
	n, bins, patches = ax1.hist([DR1[prop[0]],DR2[prop[0]]], bins=prop[1], color=['crimson', 'burlywood'], histtype='step')
	plt.savefig(directory+prop[0]+"_"+stamp+".png", dpi=300, bbox_inches='tight')
	plt.close()
