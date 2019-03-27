#inputs text file and scrambles all the lines
#python scramble.py textfile

import sys
import random

f = open(sys.argv[1],"r")

lines = []

for rl in f.readlines():
	lines.append(rl)
#strip if necessary

f.close()

f = open(sys.argv[1],"w")

while(lines):
	i = random.randint(0,len(lines)-1)
	f.write(lines[i])
	lines = lines[:i] + lines[i+1:]

f.close()
