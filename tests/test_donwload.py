import os
import sys
sys.path.insert(0, os.path.abspath("/Users/es2814/live/dcarte/"))
import dcarte  

from dcarte.derived import create_weekly_profile,create_base_datasets
# create_base_datasets()
# create_weekly_profile()
# dcarte.download_domain('raw')
Sleep_Model = dcarte.load('Sleep','base',update=True)