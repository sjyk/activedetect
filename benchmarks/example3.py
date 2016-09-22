#!/usr/bin/env python

from USCensusBenchmark import USCensusBenchmark
from NFLBenchmark import NFLBenchmark
from LABenchmark import LABenchmark
from BPBenchmark import BPBenchmark

"""
u = USCensusBenchmark()
u.saveResults()

n = NFLBenchmark()
n.saveResults()

l = LABenchmark()
l.saveResults()
"""

b = BPBenchmark()
print b.getResults()