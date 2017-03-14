#! /usr/bin/python

import sys
import os
import random

# TODO: Change this to accept 3 input "boolean" functions... Like
#     f(a,b,c) => (2^(2^3) = 256 functions)
# Edit: 
#     No, but then the search space becomes _enormous_: 256^5 for 5 functions.
#     Ugh.

def boolgen(n):
	def func(a, b):
		# f(0, 0) = n & 0x08
		# f(0, 1) = n & 0x04
		# f(1, 0) = n & 0x02
		# f(1, 1) = n & 0x01
		v = a*2+b
		ii = 8 >> v
		return 0 if (n & ii) == 0 else 1
	return func

def traverse(functions, func):
	if len(functions) >= 5:
		apply(func, functions)
	else:
		for f in xrange(0,16):
			if ((f != 6) and (f != 9)): # Single layer MLP can't do XOR or NXOR
				traverse(functions + [f], func)

def oddbefore(accumulator, x):
	lst, ob = accumulator

	v = 0 if ob == x else 1
	lst.append(v)

	return lst, v

def create_rnn(h1, h2, h3, h4, output):
	h1f = boolgen(h1)
	h2f = boolgen(h2)
	h3f = boolgen(h3)
	h4f = boolgen(h4)
	outputf = boolgen(output)
	def rnn(a):
		h1new = h1f(a, rnn.h1prev)
		h2new = h2f(a, rnn.h2prev)
		h3new = h3f(h1new, h2new)
		h4new = h4f(h1new, h2new)
		output = outputf(h3new, h4new)
		rnn.h1prev = h1new
		rnn.h2prev = h2new
		return output
	rnn.h1prev = 0
	rnn.h2prev = 0
	return rnn

def func(h1, h2, h3, h4, output):
	data = [random.randint(0,1) for _ in xrange(0, 1024)]
	oddones = reduce(oddbefore, data, [[], 0])
	rnn = create_rnn(h1, h2, h3, h4, output)
	out = reduce(lambda acc, x: acc + [rnn(x)], data, [])
	q = reduce(lambda acc, x: acc and x, map(lambda x: x[0] == x[1], zip(oddones[0], out)), True)
	if (q):
		print h1, h2, h3, h4, output

def main():
	traverse([], func)

if __name__ == "__main__":
	main()
