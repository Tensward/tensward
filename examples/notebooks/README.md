# Notebooks

`tensward-colab.ipynb` runs Tensward on Google Colab's free T4 GPU, in about 20 minutes. It installs Tensward and vLLM, downloads Qwen2.5-1.5B-Instruct, and registers a setup with two deliberate problems for an agent that sends the same instructions and tools with every request. It measures that setup, follows the first suggested change, then the next one, and prints what each change did. Open it in Colab, choose a T4 runtime and run all cells.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tensward/tensward/blob/main/examples/notebooks/tensward-colab.ipynb)
