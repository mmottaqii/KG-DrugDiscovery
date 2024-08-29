# Personalized Biomedical Knowledge Graph for Drug Repurposing

This repository is dedicated to the development of a personalized biomedical knowledge graph that utilizes large language models (LLMs) for drug repurposing. The goal is to harness comprehensive biomedical data to uncover new therapeutic potentials of existing drugs.

## Project Overview

The project combines advanced data preprocessing, integration of diverse biomedical datasets, and application of LLMs to predict drug repurposing opportunities. This approach aims to accelerate the discovery of drug repositioning and support personalized medicine.

![Overview Figure](figures/overview_figure.png)

## Repository Structure

- `scripts/`: Contains the preprocessing notebooks and Python scripts essential for building the knowledge graph.
  - `kg_preprocessing.ipynb`: Handles the integration and cleaning of drug data.
  - `preprocessing_KG_python.ipynb`: Processes gene expression data and prepares it for integration into the knowledge graph.
- `figures/`: Directory for images and figures used in documentation.
- `assets/`: Includes essential resources and metadata for the knowledge graph construction.

## Getting Started

To replicate and build upon this knowledge graph for your research:

1. Clone this repository.
2. Ensure all dependencies are installed.
3. Execute the scripts in the `scripts/` directory to preprocess the data.
4. Follow the detailed steps in the notebooks to integrate data into the knowledge graph.

## Data Description

The project utilizes various datasets including:
- Drug information from multiple repositories.
- Gene expression and genomic data from public biomedical databases.
- Additional biomedical attributes relevant to drug properties and biological interactions.

## Dependencies

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- PyTorch (for LLMs)

## License

This project is open-sourced under the MIT License.

## Contact

For queries or collaborations, please reach out via [sadeq.mottaqi@gmail.com].

## Acknowledgments

- Contributors to the public data repositories.
- Research teams and developers who maintain the LLMs.
