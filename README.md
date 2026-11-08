# URMC Neural Networks with Categoricals Assignment

This repository contains the starter code, shared utility library (`urmc_nn_asgn`), and Jupyter notebooks for the assignment.


## Prerequisites

### Input datasets

Your instructor will give you a link to download the input data. Put all input files into one folder on your system.

### .env file

You must create a file called .env.  This will contain paths to your input data and an second folder for storing intermediate files and outputs.    

To create your .env, open env.example and modify the paths.  Then do 

```bash
cp env.example .env
```
### uv installation

Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.  You can use the website or brew.


## Setup

#### Virtual environment

The file pyproject.toml specifies the Python version and packages to install.  Create the virtual environment by calling `uv sync` in this folder:

```bash
uv sync
```

#### Jupyter kernel

To make your project environment available as a kernel inside Jupyter Notebook / JupyterLab, run:

```bash
uv run python -m ipykernel install --user --name=urmc_cat_asgn --display-name="urmc_cat_asgn"
```

## Notebooks

The Jupyter notebooks (.ipynb extensions) should be run in order.  You will select the kernel you created above - you should see this kernel in the Jupyterlab launcher, or select it in your notebook by clicking near the upper right corner. 