import sys
import os 
sys.path.insert(0, os.path.abspath("/Users/es2814/live/dcarte/"))
import dcarte     
import pandas as pd 
print(pd.__version__)



print(dcarte.__version__)
dcarte.download_domain('raw')