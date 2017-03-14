#! /usr/bin/python

import sys
import os

# THIS HYPOTHESIS DOES _NOT_ WORK!
# So, function 27, for instance, has all reachable vertices
# yet, it is _not_ linearly separable.
# Look instead at linearsep3.py

def hamming(a, b):
	x = a ^ b
	ones = 0
	while x > 0:
		if (x & 0x01) == 0x01:
			ones += 1
		x = x >> 1
	return ones

def reachable(src, dest, canvas, adjacency):
	assert(canvas[src] == canvas[dest])
	if src == dest:
		rv = True
	else:
		rv = False
		for q in xrange(0, 8):
			if adjacency[src][q] == 1 and canvas[q] == canvas[src]:
				adjacency[src][q] = 0
				rv = rv or reachable(q, dest, canvas, adjacency)
				adjacency[src][q] = 1
	return rv

def main():
	adjacency = [[0 for _ in xrange(0,8)] for _ in xrange(0, 8)]
	for i in xrange(0, 8):
		for j in xrange(0, 8):
			ones = hamming(i,j)
			if (ones == 1):
				adjacency[i][j] = 1

	# OK. Now, rip through each function.
	for i in xrange(27, 28):
		canvas = [int(x) for x in ("00000000" + bin(i)[2:])[-8:]]
		invalid = None
		for j in xrange(0, 8):
			for k in xrange(j+1, 8):
				if (canvas[j] == canvas[k]):
					if (not reachable(j, k, canvas, adjacency)):
						if not invalid:
							invalid = set()
						invalid.add((j,k))
		if invalid is not None: print "%d	bad" % (i)
		else: print "%d	good" % (i)
		print canvas

if __name__ == "__main__":
	main()

"""
27	good
000	0
001	0
010	0
011	1
100	1
101	0
110	1
111	1
"""
