# Information Retrieval: Query Ranking, Expansion, and Recommender Models

The assignment involved implementing and experimenting with various IR ranking models, feedback algorithms, and query expansion techniques using real-world datasets.

Check the [report](https://github.com/vooora/IR_assignment2/blob/master/IR_ASSIGNMENT_2.pdf) attached for more details. 


---

## Project Overview

We implemented and evaluated the following models:

- Vector Space Models (nnn, ntn, ntc)
- Rocchio Feedback Algorithm (with improvements)
- Probabilistic Models: Language Model and BM-25
- Entity-based Retrieval using GENA
- Query Expansion using Knowledge Graphs
- Learning to Rank: Pointwise, Pairwise, and Listwise approaches
- Custom out-of-the-box improvements

---

## Rocchio Feedback Algorithm — Our Improvement

We implemented an **improved version of the Rocchio pseudo-relevance feedback algorithm**.

### What We Did:

> Instead of choosing the top-k documents as equally relevant, we **split the top and bottom results into levels**, assigning different weights (`β` and `γ`) to each level based on estimated relevance.

- **Top-k results** were divided into levels:
  - The **first few** got a **higher β value** (more relevant)
  - The **next few** got a **lower β value**
- **Bottom-k non-relevant results** were similarly split with decreasing **γ values**

This allowed finer control over how relevance was attributed in the feedback loop.

### Result:
| Rocchio Version              | NDCG Score |
|-----------------------------|------------|
| Traditional Top-k Feedback  | 0.2893     |
| Level-based Rocchio (ours)  | **0.3051** |

---

## Recommender Systems

We also incorporated basic recommender system logic to rank results based on document similarity and relevance propagation. The aim was to personalize or guide retrieval better based on prior feedback and structure in the data.

---

## Technologies & Libraries
<p align="left">
  <!-- Python -->
  <a href="https://www.python.org" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
  </a>

  <!-- NumPy -->
  <a href="https://numpy.org/" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="NumPy" width="40" height="40"/>
  </a>

  <!-- TensorFlow -->
  <a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" width="40" height="40"/>
  </a>

  <!-- NLTK (no official logo, custom fallback) -->
  <a href="https://www.nltk.org/" target="_blank" rel="noreferrer" style="margin: 10px;">
    <img src="https://raw.githubusercontent.com/nltk/nltk.github.com/master/images/nltk-logo.png" alt="NLTK" width="40" height="40"/>
  </a>
</p>



---

## Evaluation Metric

We used **NDCG (Normalized Discounted Cumulative Gain)** at various cutoff levels to evaluate our retrieval and learning-to-rank performance. For traditional models, we used the **TREC eval tool**.


