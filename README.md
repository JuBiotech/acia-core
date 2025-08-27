# **Acia**: Automated single-cell image analysis

![pipeline](https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/acia/badges/master/pipeline.svg)
![coverage](https://jugit.fz-juelich.de/IBG-1/ModSim/imageanalysis/acia/badges/master/coverage.svg)

**Accio** 🪄 - and your single-cell insights appear - Not quite but - `acia` - and your single-cell insights appear to become much easier 😉

The `acia` library provides a modular image analysis pipeline utility functionality for analysing 2D+t time-lapse image sequences in microfluidic live-cell imaging experiments. It provides:
- Abstraction for various image sources (local, OMERO)
- automated image analysis for instance segmentation and tracking (eight SOTA AI approaches supported out of the box)
- automated and unit-aware single-object property extraction.
- extended visualization in videos, charts and interactive charts including segmentation masks and lineage trees

Although the funtionality is developed with microfluidic applications in mind, the library can be used for any objects detected in images.

**Note:** For examples of its usage please visit our application workflow collection including more than **10** real-world examples: https://github.com/JuBiotech/acia-workflows

## Installation

Install `acia` from pypi:

```bash
pip install acia
```


## Developers

1. Clone this repository
    ```bash
    git clone https://github.com/JuBiotech/acia-core.git
    cd acia-core
    ```

2.Install `acia` in development mode

    ```bash
    pip install -e .
    ```
