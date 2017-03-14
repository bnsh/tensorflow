#! /usr/bin/python

import sys
import os
import numpy as np
"""
	This is the _simple_ version of npsequence.py.
	That one is much more general, whereas this
	one has the weights hardcoded to better illustrate.
"""

def single_step(Xt, htm1):
	# This should return ht, Y
	# These weights correspond to h1=178, h2=251, o=4 (Even Parity
	# Meaning, (paradoxically to _me_) that it outputs 1 if there
	# are an odd number of 1's.
	# which I chose rather arbitrarily. See minimal3.py
	Wh = np.array([
		[-10, 5, -5],
		[-10, 5, -5]
	])
	bh = np.array([[5,15]]).transpose()
	Wy = np.array([[-5,5]])
	by = np.array([[-5]])

	ht = np.tanh(Wh.dot(np.concatenate([Xt, htm1], axis=0)) + bh)
	yraw = (Wy.dot(ht) + by)
	y = 1/(1+np.exp(-yraw))
	return ht, y

def main():
	hstate = np.array([[0],[0]])
	data = np.random.randint(0,2,size=(20,1))
	for i in xrange(0, data.shape[0]):
		hstate, y = single_step((data[i:i+1]), hstate)
		print data[(i,0)], int(np.round(y[(0,0)]))

if __name__ == "__main__":
	main()
