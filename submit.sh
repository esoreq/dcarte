. env/bin/activate
. ~/.bash_functions
python setup.py bdist_wheel
twine upload --skip-existing dist/*
commit $PWD
