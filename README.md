#  Spike Sorting Pipeline

This repository provides a reproducible pipeline for preprocessing, sorting, and analyzing spikes based on different feature sets (amplitude and widht, features from the SS-SPDF method [Caro-Martín et al., 2018](https://pubmed.ncbi.nlm.nih.gov/30542106/), raw waveform)  via **microneurography**—using **[Snakemake](https://snakemake.readthedocs.io/en/stable/)** and modular Python scripts. It also includes a Jupyter notebook for evaluating and visualizing classification results (e.g., heatmaps of accuracy).

The pipeline supports proprietary data formats like **Dapsys** and the process data format **HDF5/NIX**, converts them to a unified **NIX** format, and executes a full analysis including spike extraction (timestamps must be provided), feature set extraction, template computation, and classification evaluation.

## Spike Tracking via the marking method

During the experiment the **[marking method](https://pubmed.ncbi.nlm.nih.gov/7672025/)** is applied, a special electrical stimulation protocl to create spike tracks (vertical alignment of fiber responses). They can be extracted and analyzed post hoc. 

In our workflow, we use two different tracking algorithms for microneurography data to identify and track spikes evoked by background stimuli:
- **Dapsys** (proprietary) – [www.dapsys.net](http://www.dapsys.net) based on [Turnquist et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26778609/)
- **SpikeSpy** (open-source) – [https://github.com/Microneurography/SpikeSpy](https://github.com/Microneurography/SpikeSpy)
  
### The extracted spike times and track labels are essential inputs for running our supervised spike sorting pipeline. The setup instructions are provided below.
---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Digital-C-Fiber/SpikeSortingPipeline.git
cd SpikeSortingPipeline
```

### 2. Create the Conda environment
This pipeline uses `conda` and `snakemake` with `Python 3.11`. We provide an environment file.

```bash
conda env create -f environment.yml
conda activate Snakemake311
```

If you haven't already installed `snakemake`:
```bash
conda install -c conda-forge snakemake
```

---

## Snakemake Directory Overview

```
├── Snakefile                      # Main Snakemake workflow
├── config.yaml                    # Configuration for dataset paths and parameters
├── environment.yml                # Conda environment file
├── scripts/                       # Core processing scripts
│   ├── read_in_data.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── create_nix.py
│   ├── templates_and_filtering.py
│   ├── compute_template_errors.py
│   ├── classification.py
│   ├── clustering.py
│   └── collect_scores.py
│   └── test_clusters.py
├── datasets_test/
│   ├── testset_1.dps              # Example Dapsys file
│   └── nix/                       # Output folder for NIX
```

---

## Configuration Format

Datasets and processing parameters are defined in `config.yaml`. Example:

```yaml
datasets:
  Testset_1:
    path: "datasets_test/testset_1.dps"
    path_nix: "datasets_test/nix/testsets_1.nix"
    name: "testset_1"
    path_dapsys: "NI Puls Stimulator/Continuous Recording"
    flags:
      use_bristol_processing: false
    time1: 200
    time2: 922
```

**Explanation:**
- `path`: Raw data file (e.g. Dapsys)
- `path_nix`: Target output path for the `.nix` file
- `path_dapsys`: Root path inside the Dapsys hierarchy, usually it is `NI Puls Stimulator/Continuous Recording` or `NI Pulse Stimulator/Continuous Recording`, for h5/nix files just write `""`
- `use_bristol_processing`: Adjusts preprocessing method for Bristol data
- `time1`, `time2`: Time window for analysis

---

## Run the Pipeline

```bash
snakemake --cores 8
```

This will:
- Read raw data
- Create data frames
- Generate `.nix` files
- Apply preprocessing (align spikes, compute derivatives, and compute templates) 
- Extract features
- Perform clustering/classification
- Output performance metrics for 6 implemented feature sets

---

## Results Visualization

Use the provided notebook to plot and explore your results:

```bash
jupyter notebook visualize_results.ipynb
```

This notebook includes:
- Heatmaps of classification accuracy across feature sets
- Metric summaries (e.g. precision, recall, etc.)

---

## Contact

If you have any questions, issues, or suggestions, feel free to reach out:

Alina Troglio
Email: alina.troglio@rwth-aachen.de

---

## How to Cite

If you use this pipeline in your work, please cite our preprint:

**Supervised Spike Sorting Feasibility of Noisy Single-Electrode Extracellular Recordings: Systematic Study of Human C-Nociceptors recorded via Microneurography**

[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.12.31.630860-red)](https://doi.org/10.1101/2024.12.31.630860)

