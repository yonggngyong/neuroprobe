# Neuroprobe

<p align="center">
  <a href="https://neuroprobe.dev">
    <img src="neuroprobe_animation.gif" alt="Neuroprobe Logo" style="height: 10em" />
  </a>
</p>

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-1f425f.svg?color=purple">
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
    </a>
    <a href="https://mit-license.org/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://neuroprobe.dev">
        <img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fneuroprobe.dev">
    </a>
</p>

<p align="center"><strong>Evaluating Intracranial Brain Responses to Naturalistic Stimuli</strong></p>

<p align="center">
    <a href="https://neuroprobe.dev">Website</a> |
    <a href="https://azaho.org/papers/NeurIPS_2025__BTBench_paper.pdf">Paper</a>
</p>

---

By **Andrii Zahorodnii¹²***, **Bennett Stankovits¹***, **Christopher Wang¹***, **Charikleia Moraitaki¹**, **Geeling Chau³**, **Ila R Fiete¹²**, **Boris Katz¹**, **Andrei Barbu¹**

¹MIT CSAIL, CBMM  |  ²MIT McGovern Institute  |  ³Caltech  |  *Equal contribution

## 🎯 Overview
Neuroprobe is a benchmark for understanding how the brain processes information across multiple tasks. It analyzes intracranial recordings during naturalistic stimuli using techniques from modern natural language processing. By probing neural responses across many tasks simultaneously, Neuroprobe aims to reveal the functional organization of the brain and relationships between different cognitive processes. The benchmark includes tools for decoding neural signals using both simple linear models and advanced neural networks, enabling researchers to better understand how the brain processes information across vision, language, and audio domains.

## 🌟 Key Features

- 19 standardized decoding tasks spanning vision, audio and language domains
- High temporal resolution intracranial recordings from 10 human subjects
- 43 hours of neural activity aligned with movie stimuli
- Standardized train/test splits and evaluation metrics
- Public leaderboard for tracking model progress
- Focus on naturalistic language processing and brain responses

## 🚀 Getting Started

### Prerequisites

1. Copy over just the folder `neuroprobe`, which contains all of the necessary components for evaluation, into your codebase where you'd like to use Neuroprobe.

2. Create a virtual environment (optional):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Installation

1. Install required packages:
```bash
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn psutil librosa
```

2. Configure dataset path in `neuroprobe/config.py`:
```python
# In neuroprobe/config.py
ROOT_DIR = "braintreebank"  # Root directory for the braintreebank data
```

3. Download and extract the dataset:
```bash
python braintreebank_download_extract.py
```

4. Start experimenting with `quickstart.ipynb` to create datasets and evaluate models.

## 📊 Evaluation

Run the linear regression model evaluation:
```bash
python single_electrode.py --subject SUBJECT_ID --trial TRIAL_ID --verbose --lite --eval_name onset --splits_type SS_DM
```

Results will be saved in the `eval_results` directory according to `leaderboard_schema.json`.

## Citation

If you use Neuroprobe in your work, please cite our paper:
```bibtex
[Citation TBD]
```

## License

This project is licensed under the MIT License.
TBD