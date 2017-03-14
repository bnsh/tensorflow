#! /usr/bin/python

import sys
import os
import random

# blessed numbers are the function numbers that are linearly separable.
# so, blessed2 omits 6 (XOR) and 9 (NXOR)
blessed2 = set([ 0, 1, 2, 3, 4, 5,    7, 8,  10,11,12,13,14,15])
# blessed3 is .. uh. Complicated. see linearsep3.py
blessed3 = set([
	  0,  1,  2,  3,  4,  5,      7,
	  8,     10, 11, 12, 13, 14, 15,
	 16, 17,     19,     21,     23,
				     31,
	 32,     34, 35,                
		 42, 43,             47,
	 48, 49, 50, 51,             55,
		     59,             63,
	 64,             68, 69,        
			 76, 77,     79,
	 80, 81,         84, 85,     87,
			     93,     95,
					
					
	112,113,    115,    117,    119,
				    127,
	128,                            
	136,    138,    140,    142,143,
					
					
	160,    162,                    
	168,    170,171,        174,175,
	176,    178,179,                
		186,187,            191,
	192,            196,            
	200,            204,205,206,207,
	208,            212,213,        
			220,221,    223,
	224,                            
	232,    234,    236,    238,239,
	240,241,242,243,244,245,    247,
	248,    250,251,252,253,254,255
])

def boolgen3(n):
	def func(a, b, c):
		# f(0, 0, 0) = n & 0x080
		# f(0, 0, 1) = n & 0x040
		# f(0, 1, 0) = n & 0x020
		# f(0, 1, 1) = n & 0x010
		# f(1, 0, 0) = n & 0x008
		# f(1, 0, 1) = n & 0x004
		# f(1, 1, 0) = n & 0x002
		# f(1, 1, 1) = n & 0x001
		v = a*4+b*2+c
		ii = 0x080 >> v
		return 0 if (n & ii) == 0 else 1
	return func

def boolgen2(n):
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
	if len(functions) >= 3:
		apply(func, functions)
	else:
		assert((0 <= len(functions)) and (len(functions) < 3))
		if len(functions) < 2:
			blessed = blessed3
		elif len(functions) == 2:
			blessed = blessed2
		else:
			blessed = [] # WE SHOULD NEVER BE HERE!
		for f in blessed:
			traverse(functions + [f], func)
			if len(functions) == 0:
				print "%d/%d" % (f, len(blessed))

def evenbefore(accumulator, x):
	lst, eb = accumulator

	v = 0 if eb == x else 1
	lst.append(v)

	return lst, v

def create_rnn(h1, h2, output):
	h1f = boolgen3(h1)
	h2f = boolgen3(h2)
	outputf = boolgen2(output)
	def rnn(a):
		h1new = h1f(a, rnn.h1prev, rnn.h2prev)
		h2new = h2f(a, rnn.h1prev, rnn.h2prev)
		output = outputf(h1new, h2new)
		rnn.h1prev = h1new
		rnn.h2prev = h2new
		return output
	rnn.h1prev = 0
	rnn.h2prev = 0
	return rnn

def writer(fp):
	def func(h1, h2, output):
		data = [random.randint(0,1) for _ in xrange(0, 1024)]
		evenones = reduce(evenbefore, data, [[], 1]) # Change the initial value to 0 to make it do "even parity" https://en.wikipedia.org/wiki/Parity_bit#Parity
		rnn = create_rnn(h1, h2, output)
		out = reduce(lambda acc, x: acc + [rnn(x)], data, [])
		q = reduce(lambda acc, x: acc and x, map(lambda x: x[0] == x[1], zip(evenones[0], out)), True)
		if (q):
			fp.write("%d	%d	%d\n" % (h1, h2, output))
			fp.flush()
	return func

def main():
	with open("/tmp/minimal3.txt", "w") as fp:
		traverse([], writer(fp))

if __name__ == "__main__":
	main()

"""
Even Parity: See https://en.wikipedia.org/wiki/Parity_bit#Parity
	4	77	4
	4	79	4
	8	112	7
	8	113	7
	23	254	1
	23	254	14
	31	254	1
	43	2	2
	47	2	2
	112	8	7
	113	8	7
	176	251	4
	178	251	4
	253	208	2
	253	212	2
	254	23	14
	254	31	1

Odd Parity:
	4	77	11
	4	79	11
	8	112	8
	8	113	8
	23	254	1
	23	254	14
	31	254	14
	43	2	13
	47	2	13
	112	8	8
	113	8	8
	176	251	11
	178	251	11
	253	208	13
	253	212	13
	254	23	1
	254	31	14

(D-uh, it seems to just negate the output for flipping from even to odd.)
"""
