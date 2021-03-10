python3 -m venv env
source env/bin/activate
gh repo create dcarte --public
git init
git add --all
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/esoreq/dcarte.git
git push -u origin main
python -m pip install --upgrade pip
python -m pip install numpy pandas wheel twine
pip freeze > requirements.txt
python setup.py sdist bdist_wheel
python setup.py bdist_wheel
python -m twine upload dist/*
