# Ingesting UKDRI CRT Data with the DCARTE

## Configuration

We start by creating a folder that will hold the different examples we will be creating today.

```bash
mkdir -p sandbox/{notebooks,code,results}
cd sandbox
```

We go into that folder using cd and create a basic git framework. Git functionalities are essential as they will track any changes we make in our work. And potentially, in the future, share our work with collaborators.

```bash
touch README.md
touch .gitignore
git init
```

## Initial setup

Dcarte primary objective is to simplify connecting to the [minder research portal](https://research.minder.care/portal/exports) and creating datasets derived from the continuously acquired datasets hosted in the portal.

First, you need to be granted access to the minder portal by the minder team.
Once this is done, you will need to create an access token using the following [link](https://research.minder.care/portal/access-tokens)

In fact, when you import dcarte for the first time on a new computer, you will be asked for this information.

Let's test this using a Jupyter notebook 

### First Jupyter notebook

Open a Jupyter notebook and at the top cell write and run the following line:

```python
import dcarte 
```

If you are setting dcarte for the first time,  you will need to copy and paste a minder token into the input bar. Once completed you should have access to selected raw datasets natively defined in dcarte.
