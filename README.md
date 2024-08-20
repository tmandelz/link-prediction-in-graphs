<table style="border: none; border-collapse: collapse; width: 100%;">
  <tr>
    <td style="border: none; width: 70%;">
      <div>
        <h1>Link Prediction in Graphs with Graph Neural Networks</h1>
        <h2>Investigating Transfer Learning</h2>
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

This repository contains code for the [Bachelor Thesis](https://web0.fhnw.ch/ht/informatik/ip6/24fs/24fs_i4ds26/index.html) by [Thomas Mandelz](https://www.linkedin.com/in/tmandelz/) and [Jan Zwicky](https://www.linkedin.com/in/jan-zwicky-894939311/).

## Abstract

This thesis explores the applications of transfer learning in link prediction using Graph Neural Networks, focusing on enhancing model performance by pre-training Graph Neural Network models and fine-tuning them afterwards. The thesis employs a robust methodology including data preprocessing, exploratory data analysis and model evaluation. Training model architectures such as GCN and NGCN, we investigate the benefits of transfer learning by pre-training models on the ogbn-papers100M dataset and fine-tuning on the ogbn-arxiv dataset. 

Our findings reveal significant improvements in model performance, with enhanced mean reciprocal rank compared to reference models, highlighting the potential of transfer learning in predicting links within graph structures. This approach not only accelerates training times but also enhances model performance for the GCN architecture. While these findings could not be fully validated for the NGCN architecture, model training remains faster.

This thesis contributes to a deeper understanding of GNN capabilities and transfer learning's impact, offering insights into their practical applications and optimisation.

* [Thesis](BA_Thesis_24FS_I4DS26_GNN_signed.pdf)

## Methods

* Graph Neural Networks (GNN)
* Graph Convolutional Networks (GCN)
* Nested Graph Convolutional Networks (NGCN)
* Open Graph Benchmarks (OGB)
* Common Neighbor
* Cosine Similarity
* Multi Dimensional Scaling (MDS)
* Integrated Gradients

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
* Model training scripts and various additianl supporting file are being kept [here](additional)
* Temporary Folder which caches the training split and other temporary files at execution time [here](temp)

## Featured Files

* [Main Explorative Data Analysis Notebook](/EDA/2_eda_dataset.ipynb) - Includes all explorative data analyses for the datasets.
* [Main Qualitative Evaluation Notebook](/evaluation/1_evaluation_build_up.ipynb) - Explains the qualitative evaluation plots with example data.
* [Main Quantitave Evaluation Notebook](evaluation/4_visualise_results_training_curves.ipynb) - Shows all quantitative visualisations and result aggragations used in the thesis.
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

To reproduce our cross-validated [reference model](https://www.comet.com/swiggy123/gcn-reference-cross-validation) execute the following code.

``` sh
gnn.py --project_name "your-comet-ml-project" --run_name "your-run-name" --epochs 2100 --dataset ogbn-arxiv --batch_size 38349 --lr 0.0004 --num_layers 2 --hidden_channels 512 --model_architecture GCN --one_batch_training False --freeze_model False --save_model True --eval_n_hop_computational_graph 2 --epoch_checkpoints 10
```

## Further Resources

* [comet-ml experiments](https://www.comet.com/swiggy123/link-prediction)
* [Open Graph Benchmark](https://ogb.stanford.edu/)

## Results

The goal of this thesis was to investigate the transferability of GNNs in the context of link prediction and to assess whether GNNs outperform traditional heuristic methods.
One of the findings of this thesis is that the GCN architecture effectively captured both structural and feature information, outperforming heuristic models. Furthermore, fine-tuned models exhibited faster training times and better initial performance metrics. This observation confirms the benefits of transfer learning thereby saving both time and computational resources.

Another important finding was the difficulties GNNs encountered in predicting edges within strongly connected components. A potential improvement could be realised by adding additional features, such as the target node’s year, target node indegree, and common neighbor values from the converted undirected graph. This suggests that adding contextual information may enhance prediction accuracy. However, it was also found that increased model capacity did not consistently lead to improved transfer learning performance. This suggests that a marginal model capacity increase does not necessarily yield better results and that the model architecture needs to be carefully tailored to the specific task. The lack of significant increases in model capacity in our work suggests the need for further research. Other model architectures could potentially enhance transferability. Kooverjee et al. (2022) demonstrated successful transfer using other architectures, such as the Graph Isomorphism Network (Xu et al., 2019) and GraphSAGE (Hamilton et al., 2018).

Additionally, architectures incorporating attention mechanisms, such as Graph Attention Networks (Veličković et al., 2018), may also enhance transferability. In future work, we encourage researchers to follow the approach of Kooverjee et al. (2022), who employed synthetic graphs in their transfer learning research. This methodology provided valuable insights into the structural components of graphs and their transferability. Additionally, the pipeline could be improved by using strategic sampling for negative links.

Choosing negative examples more carefully could optimise model performance. Another area that requires further investigation is the lower Train MRR. Additional research is needed to identify the causes of the lower values and to establish confidence in this anomaly. To achieve this, others should replicate our work and validate our findings.

In summary, this thesis has highlighted the strengths and weaknesses of GNNs in the context of link prediction and their transferability. It has also identified key areas for future research and optimisation. The findings of this investigation provide valuable insights and lay the groundwork for further studies aimed at enhancing the efficiency and accuracy of transfer learning of GNNs in link prediction.

## Contributing Members

**[Thomas Mandelz](https://github.com/tmandelz)**
**[Jan Zwicky](https://github.com/swiggy123)**
