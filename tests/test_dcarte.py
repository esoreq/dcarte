# from dcarte import __version__
# import dcarte

# def test_version():
#     assert __version__ == '0.2.1'


import sys
import os 
sys.path.insert(0, os.path.abspath("/Users/es2814/live/dcarte/"))
import dcarte     
    
# Light = dcarte.load('Temperature', 'profile', update=True)

# Sleep_Event = dcarte.load('Motion','base',update=True)
sleep_dailies = dcarte.load('sleep_dailies','profile',update=False)
pid = '2GN1PHeHwRzNYQ7q4Nvg7g'
sleep_dailies.query('patient_id == @pid and ("2021-09-22" < start_date < "2021-09-26")')