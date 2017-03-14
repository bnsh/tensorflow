#! /usr/bin/python

import sys
import os
import numpy as np
"""
	Wow, we're in some serious intellectual masturbation
	territory here. I'll make a simpler version too.
	This one, is over engineered to test all the
	parity combinations that minimal3.py
	produces.
"""

def bits(b, off, on):
	def func(n):
		cc = 1 << (b-1)
		arr = []
		while cc > 0:
			arr.append(on if (n & cc) else off)
			cc = cc >> 1
		return np.array([arr])
	return func

# I can't think of a good way to merge hiddengen and outputgen, even
# tho, conceptually, they are nearly identical. *sigh*
def hiddengen(n, margin):
	a = bits(8, -margin, margin)(n)
	w = (np.array([
		[-2,-2,-2,-2, 2, 2, 2, 2],
		[-1,-1, 1, 1,-1,-1, 1, 1],
		[-1, 1,-1, 1,-1, 1,-1, 1],
		[ 2, 2, 2, 2, 0, 0, 0, 0]
	])/8.0).dot(a.transpose())

	# Great. But, now let's actually test it. We want
	# the minpositive - maxnegative to be margin
	# and, we want to scale minpositive and maxnegative
	# to be margin/2 and -margin/2
	minpositive = None
	maxnegative = None
	inp = np.concatenate([bits(3,0,1)(i) for i in xrange(0, a.shape[1])],0)
	inp = np.concatenate([inp, np.ones((inp.shape[0],1))],1)
	inp = inp.dot(np.diagflat([1,2,2,1])) - np.ones(inp.shape).dot(np.diagflat([0,1,1,0]))
	indices = np.arange(a.shape[1]).reshape(a.shape)
	out = inp.dot(w)
	if (a>0).any():
		minpositive = out[indices[a>0]].min()
	else:
		minpositive = 0
	if (a<0).any():
		maxnegative = out[indices[a<0]].max()
	else:
		maxnegative = 0

	w[3] = w[3] - maxnegative
	currentscale = (minpositive - maxnegative)
	if currentscale > 0:
		w = w * margin / currentscale
		w[3] = w[3] - margin / 2.0
	out = inp.dot(w)
	return w[0:3,:].transpose(), w[3:4,:].transpose()

def outputgen(n, margin):
	a = bits(4, -margin, margin)(n)
	w = (np.array([
		[-1,-1, 1, 1],
		[-1, 1,-1, 1],
		[ 1, 1, 1, 1]
	]) / 4.0).dot(a.transpose())

	minpositive = None
	maxnegative = None
	inp = np.concatenate([bits(2,-1,1)(i) for i in xrange(0, a.shape[1])],0)
	inp = np.concatenate([inp, np.ones((inp.shape[0],1))],1)
	indices = np.arange(a.shape[1]).reshape(a.shape)
	out = inp.dot(w)
	if (a>0).any():
		minpositive = out[indices[a>0]].min()
	else:
		minpositive = 0
	if (a<0).any():
		maxnegative = out[indices[a<0]].max()
	else:
		maxnegative = 0

	w[2] = w[2] - maxnegative
	currentscale = (minpositive - maxnegative)
	if currentscale > 0:
		w = w * margin / currentscale
		w[2] = w[2] - margin / 2.0
	out = inp.dot(w)
	return w[0:2,:].transpose(), w[2:3,:].transpose()

def weightgen(h1, h2, o, margin):
	Wh1, bh1 = hiddengen(h1, margin)
	Wh2, bh2 = hiddengen(h2, margin)
	Wh = np.concatenate([Wh1, Wh2], axis=0)
	bh = np.concatenate([bh1, bh2], axis=0)
	Wo,  bo = outputgen(o, margin)
	return Wh, bh, Wo, bo

def single_step(Xt, htm1, Wh, bh, Wy, by):
	# This should return ht, Y

	ht = np.tanh(Wh.dot(np.concatenate([Xt, htm1], axis=0)) + bh)
	yraw = (Wy.dot(ht) + by)
	y = 1/(1+np.exp(-yraw))
	return ht, y


def main():
	Wh, bh, Wy, by = weightgen(4,77,4,10)
	hstate = np.array([[0],[0]])
	data = np.random.randint(0,2,size=(20,1))
	for i in xrange(0, data.shape[0]):
		hstate, y = single_step((data[i:i+1]), hstate, Wh, bh, Wy, by)
		print data[(i,0)], int(np.round(y[(0,0)]))

if __name__ == "__main__":
	main()
