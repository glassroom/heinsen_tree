# heinsen_tree

Reference implementation of "[Tree Methods for Hierarchical Classification in Parallel](https://arxiv.org/abs/2209.10288)" (Heinsen, 2022), for mapping a batch of scores and labels, corresponding to given nodes in a semantic tree, to scores and labels corresponding to all nodes in the ancestral paths going down the tree to every given node -- relying only on tensor transformations that execute efficiently in highly parallel hardware accelerators (e.g., GPUs, TPUs).

A toy example is helpful for conveying quickly what these methods do:

```python
# A tiny semantic tree with 6 classes in 3 levels of depth:
#
#   pet --+-- 0 "dog" --+-- 2 "small dog"
#         |             |
#         |             +-- 3 "big dog" --+-- 4 "happy big dog"
#         |                               |
#         |                               +-- 5 "angry big dog"
#         +-- 1 "other"
#
import torch
from heinsen_tree import ClassTree
tree = ClassTree([[0], [1], [0, 2], [0, 3], [0, 3, 4], [0, 3, 5]])

# Obtain a batch of four scores and labels:
scores = torch.randn(4, 6)           # 4 predictions for 6 classes
labels = torch.tensor([4, 1, 5, 2])  # 4 targets

# Map scores and labels to their respective ancestral paths:
scores_in_tree = tree.map_scores(scores)  # shape is [4, 3, 6]
labels_in_tree = tree.map_labels(labels)  # shape is [4, 3]

# Compute a classification loss at every applicable level of depth, in parallel:
idx = (labels_in_tree != tree.pad_value)  # shape is [4, 3]
loss = torch.nn.functional.cross_entropy(scores_in_tree[idx], labels_in_tree[idx])
```

For an example of hierarchical classification over a large semantic tree, see [here](#sample-usage-with-wordnet). For efficient, parallel methods to find the top k ancestral paths that have highest joint predicted probability, see [here](#tips-for-inference).


## Contents

* [Installing](#installing)

* [How Does it Work?](#how-does-it-work)

* [Sample Usage with WordNet](#sample-usage-with-wordnet)

* [Tips for Training](#tips-for-training)

* [Tips for Inference](#tips-for-inference)
  * [Example: Using Parallel Beam Search to Make Predictions](#example-using-parallel-beam-search-to-make-predictions)
  * [Example: Using Levenshtein Distance to Make Predictions](#example-using-levenshtein-distance-to-make-predictions)

* [Usage as a Model Component](#usage-as-a-model-component)

* [Citing](#citing)

* [Notes](#notes)


## Installing

```
pip install git+https://github.com/glassroom/heinsen_tree
```

Alternatively, you can download a single file to your project directory: [heinsen_tree.py](heinsen_tree/heinsen_tree.py).

The only dependency is PyTorch.


## How Does it Work?

`ClassTree` is a PyTorch module for mapping batches of classification scores and labels, corresponding to given nodes in a semantic tree, to scores and labels corresponding to all nodes in the ancestral paths going down the tree to every given node. The module has two methods, `map_scores` and `map_labels`, that map batches of scores and labels, respectively. These methods rely only on tensor transformations that common software frameworks for machine learning, like PyTorch, execute efficiently -- particularly in hardware accelerators like GPUs and TPUs. `ClassTree` enables efficient hierarchical classification in parallel.

For details, see [the paper](https://arxiv.org/abs/2209.10288).


## Sample Usage with WordNet

Instantiate a tree with all English-language synsets in WordNet (117,659 classes, 20 levels of depth):

```python
import torch
from nltk.corpus import wordnet  # see nltk docs for installation
from heinsen_tree import ClassTree

class_names = [synset.name() for synset in wordnet.all_eng_synsets()]
class_name_to_id = { name: i for i, name in enumerate(class_names) }
paths_down_tree = [
    [class_name_to_id[s.name()] for s in wordnet.synset(class_name).hypernym_paths()[-1]]
    for class_name in class_names
]  # ancestral paths ending at every class in the WordNet tree
tree = ClassTree(paths_down_tree)
```

We'll map a batch with scores and labels to their respective ancestral paths:

```python
batch_sz = 100
scores = torch.randn(batch_sz, tree.n_classes)           # normally predicted by a model
labels = torch.randint(tree.n_classes, size=[batch_sz])  # targets, each a class in the tree
```

Mapping the batch incurs negligible computation and consumes only the memory occupied by mapped data:

```python
scores_in_tree = tree.map_scores(scores)  # shape is [batch_sz, tree.n_levels, tree.n_classes]
labels_in_tree = tree.map_labels(labels)  # shape is [batch_sz, tree.n_levels]
```

## Tips for Training

When training, filter out padding values to flatten mapped scores into a matrix and mapped labels into a vector, for computing classification loss (e.g., cross-entropy) at every applicable level of depth in parallel:

```python
idx = (labels_in_tree != tree.pad_value)  # shape is [batch_sz, tree.n_levels]
loss = torch.nn.functional.cross_entropy(scores_in_tree[idx], labels_in_tree[idx])
```

## Tips for Inference

We recommend that you restrict the space of allowed predictions to *ancestral paths that exist in the tree*, stored in `tree.paths`, a PyTorch buffer (corresponding to matrix P in the paper), in two steps:

1. Predict naive probability distributions at every level of depth with a single Softmax or log-Softmax (e.g., `tree.map_scores(scores).softmax(dim=-1)`), which runs efficiently because by default the scores are masked with `-inf` values, ignored by PyTorch. The distributions are naive because at each level of depth they are independent of each other, instead of conditional on predictions at shallower levels.

2. Use any method of your choice to select the ancestral path (or top k ancestral paths) in `tree.paths` that best match the naively predicted probabilities. Below, we show examples of two methods: parallel beam search and smallest Levenshtein distance.


### Example: Using Parallel Beam Search to Make Predictions

Here we use [beam search](https://en.wikipedia.org/wiki/Beam_search) to find the top k ancestral paths that have the highest joint predicted probability. The number of possible paths is finite, so we can efficiently execute beam search over *all* paths in parallel:

```python
# Map predicted scores to tree levels:
scores_in_tree = tree.map_scores(scores)           # [batch_sz, tree.n_levels, tree.n_classes]

k = 5

# Replace pad values with a new temporary class:
tmp_class = torch.tensor(tree.n_classes)           # follows (0, 1, ..., tree.n_classes - 1)
is_pad = (tree.paths == tree.pad_value)            # [tree.n_classes, tree.n_levels]
paths = tree.paths.masked_fill(is_pad, tmp_class)  # [tree.n_classes, tree.n_levels]

# Predict naive log-probs and flatten (unmask) them:
log_probs = scores_in_tree.log_softmax(dim=-1)     # [batch_sz, tree.n_levels, tree.n_classes]
log_probs = log_probs[:, ~tree.masks]              # [batch_sz, tree.n_classes]

# Assign a pred log-prob of 0 to temporary class:
log_probs = torch.nn.functional.pad(
    log_probs, (0, 1), value=0)                    # [batch_sz, tree.n_classes + 1]

# Distribute pred log-probs over *all* tree paths:
path_log_probs = log_probs[:, paths]               # [batch_sz, tree.n_classes, tree.n_levels]

# Predict the top k paths:
topk = path_log_probs.sum(dim=-1).topk(k, dim=-1)  # k tree paths with highest joint log-probs
topk_preds_in_tree = tree.paths[topk.indices]      # [batch_sz, k, tree.n_levels]
```


### Example: Using Levenshtein Distance to Make Predictions

Here, we predict the top k ancestral paths that have the smallest [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) to each naively predicted path in the batch. The possible classes at each level of depth are fixed, so we can compute Levenshtein distance efficiently by summing elementwise differences between paths in parallel:

```python
# Map predicted scores to tree levels:
scores_in_tree = tree.map_scores(scores)           # [batch_sz, tree.n_levels, tree.n_classes]

k = 5

# Compute all Levenshtein distances:
naive_preds = scores_in_tree.argmax(dim=-1)        # [batch_sz, tree.n_levels]
is_diff = (naive_preds[:, None, :] != tree.paths)  # [batch_sz, tree.n_classes, tree.n_levels]
lev_dists = is_diff.sum(dim=-1)                    # [batch_sz, tree.n_classes]

# Predict the top k paths:
topk = lev_dists.topk(k, largest=False, dim=-1)    # k tree paths with smallest Lev dists
topk_preds_in_tree = tree.paths[topk.indices]      # [batch_sz, k, tree.n_levels]
```

In practice, weighting Levenshtein distances by path density at each level of tree depth often works better:

```python
# Compute weighted Levenshtein distances:
is_node = (tree.paths != tree.pad_value)                  # [tree.n_classes, tree.n_levels]
density = is_node.float().mean(dim=-2, keepdim=True)      # [1, tree.n_levels]
weighted_lev_dists = (is_diff * density).sum(dim=-1)      # [batch_sz, tree.n_classes]

# Predict the top k paths:
topk = weighted_lev_dists.topk(k, largest=False, dim=-1)  # k tree paths with smallest dists
topk_preds_in_tree = tree.paths[topk.indices]             # [batch_sz, k, tree.n_levels]
```


## Usage as a Model Component

You can use `ClassTree` as a component of models like any other PyTorch module. For convenience, the module's default `forward` method is a shim that calls `map_scores`. As an example, here we build and apply a linear model that classifies feature vectors into classes at all levels of ancestral depth in a given tree:

```python
class HierarchicalClassifier(nn.Module):

    def __init__(self, n_features, paths_down_tree):
        super().__init__()
        self.linear = nn.Linear(n_features, len(paths_down_tree))
        self.tree = ClassTree(paths_down_tree)

    def forward(self, x):
        x = self.linear(x)
        x = self.tree(x)
        return x

batch_sz, n_features = (100, 1024)
inputs = torch.randn(batch_sz, n_features)

model = HierarchicalClassifier(n_features, paths_down_tree)  #  e.g., WordNet tree
scores_in_tree = model(inputs)  # [batch_sz, model.tree.n_levels, model.tree.n_classes]
```


## Citing

```
@misc{heinsen2022tree,
      title={Tree Methods for Hierarchical Classification in Parallel},
      author={Franz A. Heinsen},
      year={2022},
      eprint={2209.10288},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Notes

We originally conceived and implemented these methods as part of our AI software, nicknamed Graham. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code, clean it up, and release it as stand-alone open-source software without having to disclose any key intellectual property.

Our code has been tested on Ubuntu Linux 20.04+ with Python 3.8+.

We hope others find our work and our code useful.
