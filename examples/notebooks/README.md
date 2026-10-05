# Notebooks

`tensward-colab.ipynb` runs Tensward on Google Colab's free T4 GPU. It installs Tensward and vLLM, downloads Qwen2.5-1.5B-Instruct, registers a deliberately throttled setup (at most 2 requests at once), measures it, follows the first suggested change and prints what changed. Open it in Colab, choose a T4 runtime and run all cells.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tensward/tensward/blob/main/examples/notebooks/tensward-colab.ipynb)
