import sys 
import os
sys.path.insert(0, os.path.abspath("/Users/eyalsoreq/github/DCARTE/"))
import dcarte


df = dcarte.load('activity','raw',update=True)