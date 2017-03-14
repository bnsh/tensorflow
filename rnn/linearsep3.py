#! /usr/bin/python

import sys
import os

def weights(a0,a1,a2,a3,a4,a5,a6,a7):
	w1 = (-a0-a1-a2-a3+a4+a5+a6+a7)/4.0
	w2 = (-a0-a1+a2+a3-a4-a5+a6+a7)/4.0
	w3 = (-a0+a1-a2+a3-a4+a5-a6+a7)/4.0
	w4 = (2*a0+a1-a2+a4-a7)/4.0
	return w1, w2, w3, w4

def bits(b, off, on):
	def func(n):
		cc = 1 << (b-1)
		arr = []
		while cc > 0:
			arr.append(on if (n & cc) else off)
			cc = cc >> 1
		return arr
	return func

def printf(fmt, *args):
	sys.stdout.write(fmt % args)

def analyze(n):
	# These are the _outputs_ we want for function_n. (Effectively, we're coding a "tanh" here,
	# but the distinction is irrelevant to the point of finding linearly separable 3 boolean
	# input functions.
	arr = bits(8,-1,1)(n)
	w1, w2, w3, w4 = weights(*arr)
	def func(a,b,c):
		return a * w1 + b * w2 + c * w3 + w4

	minpositive = None
	maxnegative = None
	for i, a in enumerate(arr):
		inp = bits(3,0,1)(i)
		o = (func(*inp))
		if (a < 0):
			if maxnegative is None or maxnegative < o:
				maxnegative = o
		elif (a > 0):
			if minpositive is None or minpositive > o:
				minpositive = o
	if minpositive is None:
		minpositive = 0
	if maxnegative is None:
		maxnegative = 0

	good = (maxnegative < minpositive)

	"""
	if good: printf("%d	good\n", n)
	else: printf("%d	bad\n", n)
	for i, a in enumerate(arr):
		inp = bits(3,0,1)(i)
		stat = "pass" if (((func(*inp) < 0) and (a < 0)) or ((func(*inp) > 0) and (a > 0))) else "fail"
		printf("	%s: f%d(%d,%d,%d) = %.7f", stat, n, inp[0],inp[1],inp[2], w4)
		if (inp[0] > 0.5):
			printf(" + %.7f", w1)
		if (inp[1] > 0.5):
			printf(" + %.7f", w2)
		if (inp[2] > 0.5):
			printf(" + %.7f", w3)
		printf(" = %12.7f (should be %12.7f)\n", func(*inp), a)
	printf("	{ %.7f, %.7f, %.7f, %.7f }\n", w1, w2, w3, w4)
	"""
	return good

def main():
	printf("[")
	needs_comma = False
	for n in xrange(0, 256):
		analyze(n)
		if n > 0:
			if needs_comma: printf(",")
			else: printf(" ")
		if (n % 8) == 0:
			printf("\n	")
		if analyze(n):
			printf("%3d", n)
			needs_comma = True
		else:
			printf("   ")
			needs_comma = False
	printf("\n]")

if __name__ == "__main__":
	main()

"""
0	good
1	good
2	good
3	good
4	good
5	good
7	good
8	good
10	good
11	good
12	good
13	good
14	good
15	good
16	good
17	good
19	good
21	good
23	good
31	good
32	good
34	good
35	good
42	good
43	good
47	good
48	good
49	good
50	good
51	good
55	good
59	good
63	good
64	good
68	good
69	good
76	good
77	good
79	good
80	good
81	good
84	good
85	good
87	good
93	good
95	good
112	good
113	good
115	good
117	good
119	good
127	good
128	good
136	good
138	good
140	good
142	good
143	good
160	good
162	good
168	good
170	good
171	good
174	good
175	good
176	good
178	good
179	good
186	good
187	good
191	good
192	good
196	good
200	good
204	good
205	good
206	good
207	good
208	good
212	good
213	good
220	good
221	good
223	good
224	good
232	good
234	good
236	good
238	good
239	good
240	good
241	good
242	good
243	good
244	good
245	good
247	good
248	good
250	good
251	good
252	good
253	good
254	good
255	good
"""
