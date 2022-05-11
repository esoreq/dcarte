# from dcarte import __version__
# import dcarte

# def test_version():
#     assert __version__ == '0.2.1'


import sys
import os 
sys.path.insert(0, os.path.abspath("/Users/eyalsoreq/github/DCARTE/"))
import dcarte     
    
Light = dcarte.load('Light', 'profile', update=True)

