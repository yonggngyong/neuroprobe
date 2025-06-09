# Neuroprobe

<p align="center">
  <img src="neuroprobe_animation.gif" alt="Neuroprobe Logo" style="height: 10em" />
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

By Andrii Zahorodnii¹²*, Bennett Stankovits¹*, Christopher Wang¹*, Charikleia Moraitaki¹, Geeling Chau³, Ila R Fiete¹², Boris Katz¹, Andrei Barbu¹

¹MIT CSAIL, CBMM  |  ²MIT McGovern Institute  |  ³Caltech  |  *Equal contribution

## 🎯 Overview
Understanding the relationship between the various tasks the brain performs can shed light on its functional organization. We introduce a benchmark, Neuroprobe, which targets a wide range of multimodal tasks. Neuroprobe borrows several ideas from modern natural language processing: using large scale naturalistic datasets, probing at scale across tasks as a means to understand black box systems, and evaluating on large benchmarks that test many different skills. For artificial networks, probe analysis attempts to decode attributes from different layers. It is one of the main vehicles used to shed light on the relationship and dependencies between tasks and the algorithms that networks learn. While prior neuroscience benchmarks tend to focus on a single or a very small number of tasks, Neuroprobe uses a fixed set of subjects with a large amount of data across many annotated tasks, which will allow us to create an integrated picture. Furthermore, the results obtained from Neuroprobe evaluations can yield time-orderings between different tasks and recover the functional relationships between tasks that reveal properties of the algorithms the brain uses. The main remaining bottleneck to achieving these type of results is that decoding performance for many tasks is very poor. We demonstrate a few tasks both with simple linear decoders and neural foundation models, then introduce a large number of additional attributes that should, in principle, be decodable but are not. Neuroprobe gives us an opportunity to build higher accuracy decoders, better neural foundation models that are tested across many tasks, and to bring neuroscience closer to the methodology that has worked so well in natural language understanding, and to ultimately discover the functional organization of the brain across many tasks.


## 🌟 Key Features

- 📊 19 standardized decoding tasks spanning vision, audio and language domains
- 🧠 High temporal resolution intracranial recordings from 10 human subjects
- ⏱️ 43 hours of neural activity aligned with movie stimuli
- 📈 Standardized train/test splits and evaluation metrics
- 🏆 Public leaderboard for tracking model progress
- 🔍 Focus on naturalistic language processing and brain responses

## 🚀 Getting Started

### Prerequisites

Create a virtual environment (optional):
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

2. Configure dataset path:
```python
# In neuroprobe_config.py
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

## 📚 Citation

If you use Neuroprobe in your work, please cite our paper:
```bibtex
[Citation TBD]
```

## 📝 License

This project is licensed under the MIT License.
TBD