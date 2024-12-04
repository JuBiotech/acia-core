# **acia**: Automated single-cell image analysis

![pipeline](https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/acia/badges/master/pipeline.svg)
![coverage](https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/acia/badges/master/coverage.svg)

The `acia` library provides utility functionality for analysing 2D+t time-lapse image sequences in microfluidic live-cell imaging experiments. It provides:
- Abstraction for various image sources (local, OMERO)
- automated image analysis for instance segmentation and tracking
- automated and unit-aware single-object property extraction.

Although the funtionality is developed with microfluidic applications in mind, the library can be used for any objects detected in images.

## Installation

Install `acia` from pypi:

```bash
pip install acia
```


## Developers

1. Clone this repository
    ```bash
    git clone https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/acia
    ```

2. Create the conda environment (including dependencies) and install `acia`

    ```bash
    conda env create -f conda.yaml
    conda activate acia
    pip install -e .
    ```
