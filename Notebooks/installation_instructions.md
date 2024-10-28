# How to run the demo notebooks

## Google Colab

The simplest way to run these notebooks is by using Google Colab, following the links below. The training time is about 50 min.

| Notebook content | Link |
| - | --- |
|01 - Random network with symmetric iSTDP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comp-neural-circuits/structured-stabilization-in-recurrent-neural-circuits/blob/main/Notebooks/01_random_network_symmetric.ipynb) |
|02 - Random network with antisymmetric iSTDP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comp-neural-circuits/structured-stabilization-in-recurrent-neural-circuits/blob/main/Notebooks/02_random_network_antisymmetric.ipynb) |

## Local install

The main dependency for these scripts are numpy, Brian2, matplotlib. They might therefore work out-of-the-box in most environments that already have these packages.

Note that the notebooks do not generate any saved file or plot, figures are only generated internally. The expected running time is between 20 and 30 min.

If you want to reproduce the environment I tested, follow these steps on Linx platforms, or equivalent steps on Windows/OS X.  Start with a recent version of python (>=3.10) and of venv. Then do the following:
  - Start from the `Notebooks` directory
  - Create a local environment with `python -m venv .venv`
  - Activate the local environment with `source .venv/bin/activate` 
  - Install all dependencies with `pip install -r ./requirements.txt`
  - Run jupyter lab as `jupyter lab` and open the notebooks, or...
  - ... open the `.py` scripts in your favorite python IDE.

## Details on scripts

The `*.py` and the `*.ipynb` files match each other, as they are synchronized through jupytext.

If you experience issues, or you notice bugs, please open an issue in this repository.

---

<img src="../ImagesForReadme/work_in_progress.webp" alt="Work in progress!" width="600"/>


## License

This code is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). You are free to use, modify, and distribute this software, provided that you give appropriate credit to the original authors.