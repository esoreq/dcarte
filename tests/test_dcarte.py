# from dcarte import __version__
# import dcarte

# def test_version():
#     assert __version__ == '0.2.1'


import sys
import os 
sys.path.insert(0, os.path.abspath("/Users/es2814/live/dcarte/"))
import dcarte     
    
# Light = dcarte.load('Temperature', 'profile', update=True)

Sleep_Event = dcarte.load('Motion','base',update=True)