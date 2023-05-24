import sys
import os 
sys.path.insert(0, os.path.abspath("/Users/es2814/live/dcarte/"))
import dcarte     


phys_dailies = dcarte.load('physiology_dailies','profile', reapply=True)
