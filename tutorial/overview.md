# Ingesting UKDRI CRT Data with the DCARTE

## Overview

DCARTE provides a managed solution for storing and accessing UKDRI CRT smart home dataset either locally or in the RCS Cloud, providing a bridge between the minder cloud infrastructure and conducting data-driven research on this dynamic cohort. Using DCARTE, you can manage multiple independent studies share reproducible experiments, and improve the quality and reliability of your data analysis.

In this lab, you will learn about and use DCARTE's different features to speed up your data analytics capabilities.

### What you will learn

1. Gain a general understanding of how to set up an anaconda and Jupyter environment.
2. Understand how to setup DCARTE for the first time
3. Discover how to use DCARTE to download UKDRI CRT raw datasets.
5. How to re-create derived data domain snapshots in your local workspace using existing recipes.
6. Finally, how to develop your own recipe to share with colleagues.

### Setup and requirements

#### Before you can start the lab please read these instructions

To make sure there are no dependency issues we will start by setting up a new anaconda environment.

1. Make sure you have anaconda installed ([or follow these instructions if it isn't installed](https://docs.anaconda.com/anaconda/install/)).
2. Once we verified anaconda is installed we will create a new environment. 
* Open a terminal and run the following command:
   
   ```bash
   conda create --name sandbox python=3.8
   ```
    This commands creates a virtual environment called sandbox with python version 3.8 and no other depednencies.  
* Once this has been installed we will activate the environment

   ```bash
   conda activate sandbox 
   ```
   you should see a visual indicator in the command line that shows that the base environment was succsefuly changed to sandbox.

* Once this has been installed we will activate the environment

   ```bash
   pip install dcarte
   ```
   This will install the various dependcies needed by dcarte.

* We now need to register the environment as a valid ipykernel in the Jupyter notebook framework we will be using for the rest of the lab.

   ```bash
   pip install ipykernel
   python -m ipykernel install --name sandbox
   ```

* Last thing for those using a local computer - I recommend installing [Visual studio code](https://code.visualstudio.com/) as it allows for a superior coding experiance and it's free.

