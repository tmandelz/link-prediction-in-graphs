<table style="border: none; border-collapse: collapse; width: 100%;">
  <tr>
    <td style="border: none; width: 70%;">
      <div>
        <h1>Investigating Transfer Learning for Link Prediction in Graph Neural Networks</h1>
      </div>
    </td>
    <td style="border: none; width: 30%; padding: 0; text-align: right;">
      <div style="display: flex; justify-content: flex-end; align-items: center; height: 200px;">
        <div style="width: 200px; height: auto; overflow: hidden; border-radius: 50px;">
          <img src="assets/titlepage.png" alt="Circular Image" style="width: 100%; height: auto;">
        </div>
      </div>
    </td>
  </tr>
</table>

This repository contains code for the Paper [Investigating Transfer Learning for Link Prediction
in Graph Neural Networks]() by [Thomas Mandelz](https://www.linkedin.com/in/tmandelz/), [Jan Zwicky](https://www.linkedin.com/in/jan-zwicky-894939311/),[Daniel Perruchoud](https://www.linkedin.com/in/daniel-olivier-perruchoud-799aaa38/) and [Stephan Heule](https://www.linkedin.com/in/stephanheule/).

## Abstract

Graph Neural Networks (GNNs) emerged as powerful tools for learning representations of graph-structured data, increasingly applied to various domains. Despite their growing popularity, the transferability of GNNs remains underexplored.
Transfer learning showed remarkable success in traditional deep learning tasks, enabling faster training and enhanced performance. Although GNNs are gaining popularity and are being applied in many areas, their transferability in link prediction is not well-studied. 

This research investigates the applications of transfer learning in link prediction using GNNs, focusing on enhancing model performance as well as training efficiency through pre-training GNN models, followed by fine-tuning. Specifically, we train Graph Convolutional Network (GCN), GraphSAGE and Graph Isomorphism Network (GIN) architectures and investigate the benefits of transfer learning by pre-training and fine-tuning models on public data (i.e. on ogbn-papers100M and ogbn-arxiv datasets). 
Reference models, constructed with identical capacity and trained on the same datasets, ensure a fair comparison to the fine-tuned models. Jumpstart and asymptotic performance are used to determine the transferability between models, while training time ratios measure training efficiency.

Our findings show that transfer learning improves fine-tuned model performance, boosting jumpstart scores in relation to the reference models range from 0.63 (jumpstart) for GCN, 0.47 for GraphSAGE, 0.48 for GIN, while also reducing training time up to 15 times for GraphSAGE.

## Methods

* Graph Neural Networks (GNN)
* Graph Theory,
* Link Prediction
* Transfer Learning
* Deep Learning

## Technologies

* Python
* PyTorch
* PyTorch-geometric
* NetworkX
* Docker
* Comet-ml

## Datasets

* [ogbl-citation2](https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2)
Used for reproduction and validation of the GNN pipeline.
* [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
Used to train reference models and finetune models.
* [ogbl-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M)
Used to pretrain models for later finetuning.

## Overview Folder Structure

* Graph data will be downloaded to [here](dataset)
* Scripts for Dataset Choices and Explorative Data Analysis are being kept [here](EDA)
* Scripts for Qualitative Evaluation and Quantitative Visualisation are being kept [here](evaluation)
* Source code for GNN Pipeline, Heuristics, Dataset Split implementation, Qualitative evaluation and models are being kept [here](modelling)
* Various additianl supporting file are being kept [here](additional)
* Temporary Folder which caches the training split and other temporary files at execution time [here](temp)

## Featured Files

* [Main Explorative Data Analysis Notebook](/EDA/2_eda_dataset.ipynb) - Includes all explorative data analyses for the datasets.
* [Main Qualitative Evaluation Notebook](/evaluation/1_evaluation_build_up.ipynb) - Explains the qualitative evaluation plots with example data.
* [Main Quantitave Evaluation Notebook](evaluation/4_visualise_results_training_curves.ipynb) - Shows all quantitative visualisations and result aggragations used in this research.
* [Main GNN Pipeline File](/modelling/gcn/gnn.py) - Is the main training pipeline file for our GNN models.
* [Main Cosine Similarity Heuristic Pipeline File](/modelling/heuristics/baseline_cosine_similarity.ipynb) - Includes the source code for the cosine similarity heuristic.
* [Main Common Neighbor Heuristic File](/modelling/heuristics/cn_baseline.py) - Includes the source code for the common neighbor heuristic.

## Installation Pipenv Environment

### Voraussetzungen

* Pipenv installed in local Python Environment [Pipenv](https://pipenv.pypa.io/en/latest/) or just run `pip install pipenv` in your CLI

### First Installation of Pipenv Environment

* open your CLI
* run `cd /your/local/github/repofolder/`
* run `pipenv install`
* Restart VS Code or IDE
* Choose the newly created "link-prediction-in-graphs" Virtual Environment python Interpreter

### Environment already installed (Update dependecies)

* open your CLI
* run `cd /your/local/github/repofolder/`
* run `pipenv sync`

## Usage

You need to change the API keys of comet-ml to yours, change your project and run name.

To reproduce our GraphSAGE [reference model](https://www.comet.com/transfer-learning-link-prediction/long-reference/e7eddb5ce3104801aa355a461b0540c2) execute the following code.

``` sh
gnn.py --project_name "your-comet-ml-project" --run_name "your-run-name" --epochs 2100 --dataset ogbn-arxiv --batch_size 38349 --lr 0.0004 --num_layers 2 --hidden_channels 512 --model_architecture GCN --one_batch_training False --freeze_model False --save_model True --eval_n_hop_computational_graph 2 --epoch_checkpoints 10
```

## Further Resources

* [comet-ml experiments](https://www.comet.com/transfer-learning-link-prediction)
* [Open Graph Benchmark](https://ogb.stanford.edu/)


## Contributing Members

**[Thomas Mandelz](https://github.com/tmandelz)**
**[Jan Zwicky](https://github.com/swiggy123)**
**[Daniel Perruchoud](https://www.fhnw.ch/de/personen/daniel-perruchoud)**
**[Stephan Heule](https://www.fhnw.ch/de/personen/stephan-heule)**
