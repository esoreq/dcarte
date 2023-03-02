import os
import sys
sys.path.insert(0, os.path.abspath("/Users/es2814/live/dcarte/"))
import dcarte  
from dcarte.minder import MinderDataset

data = dcarte.load('Activity','raw',update=True)
