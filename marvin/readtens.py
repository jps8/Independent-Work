import numpy
import sys
import os
import multiprocessing
import itertools
import random
import time
import marvin_io



tfile = sys.argv[1]
tl = marvin_io.read_tensor(tfile)
for t in tl:
    print t
    print 'name ', t.name
    print 'shape is: ', t.value.shape
    print 'value is: ', t.value
    i = 0
    for a in t.value:
        i+=1
        if i >10:
            break
        for b in a:
            for c in b:
                print ""
                for d in c:
                    print str(d)
