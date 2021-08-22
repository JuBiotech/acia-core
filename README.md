# AutomatedCellularImageAnalysis

[![coverage report](https://jugit.fz-juelich.de/j.seiffarth/acia/badges/master/coverage.svg)](https://jugit.fz-juelich.de/j.seiffarth/acia/-/commits/master)



Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT license
* Documentation: https://acia.readthedocs.io.


Features
--------

* TODO

Quickstart Guide (3-steps)
--------

1. Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) installed. You can check a valid installation by entering `anaconda` into your windows search. This should show an app (`Anaconda Prompt`). If you start that, it will open a command line and the command

```
conda --version
```
shows you the version information of anaconda. If this works without any error, you're ready to go on.
Otherwise see the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) to install Anaconda first.

2. Create a new environment using the following command inside the anaconda terminal

```
conda create -n mlflow -c conda-forge -y mlflow
```

You only have to do this once. Next time you can immediately start with step (3).

3. Now we are ready to perform our first cell segmentations. First activate our software using

```
conda activate mlflow
```

To perform the segmentation you need your omero server url (e.g. ibt056), your username (e.g. root), and the project id (e.g. 4) of the project you want to segment. If you have everything at hand you can run the segmentation (the example uses the exemplary server url, username and project id. Please use your's!).

```
mlflow run https://jugit.fz-juelich.de/j.seiffarth/acia.git -P serverUrl=ibt056 -P user=root -P projectId=4
```

If you are asked for credentials you have to enter your credentials for `jugit.fz-juelich.de`. If the execution takes some time this is okay: The software needs to be installed and the cell segmentations also take some time ;)

Developer installation
-------

Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
