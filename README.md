# heinsen_tree

Reference implementation of "Tree Methods for Hierarchical Classification in Parallel" (Heinsen, 2022), for transforming classification scores and labels, corresponding to given nodes in a semantic tree, to scores and labels corresponding to all nodes in the ancestral paths going down the tree to every given node -- efficiently, in parallel.

A toy example is worth more than a thousand words:

```python
>>> # A tiny semantic tree with 6 classes in 3 levels of depth:
>>> #
>>> #   pet --+-- 0 "dog" --+-- 2 "small dog"
>>> #         |             |
>>> #         |             +-- 3 "big dog" --+-- 4 "happy big dog"
>>> #         |                               |
>>> #         |                               +-- 5 "angry big dog"
>>> #         +-- 1 "other"
>>> #
>>> tree = ClassTree([[0], [1], [0, 2], [0, 3], [0, 3, 4], [0, 3, 5]])
>>>
>>> # Map a batch of four scores and labels to their ancestral paths:
>>> scores = torch.randn(4, 6)           # 4 predictions for 6 classes
>>> labels = torch.tensor([4, 1, 5, 2])  # 4 targets
>>>
>>> scores_in_tree = tree.map_scores(scores)  # shape is [4, 3, 6]
>>> labels_in_tree = tree.map_labels(labels)  # shape is [4, 3]
>>>
>>> print(labels_in_tree)  # pad value is -1 by default
tensor([[0,  3,  4],
        [1, -1, -1],
        [0,  3,  5]]
        [0,  2, -1]])
```

## Installing

`pip install -e git+https://github.com/glassroom/heinsen_tree#heinsen_tree`

Alternatively, you can download a single file to your project directory: [heinsen_tree.py](heinsen_tree/heinsen_tree.py).

The only dependency is PyTorch.


## How Does it Work?

`ClassTree` is a PyTorch module implementing the methods we propose in our paper. These methods are algebraically expressible as tensor transformations that common software frameworks for machine learning, like PyTorch, execute efficiently, particularly in hardware accelerators like GPUs and TPUs. Our methods enable efficient hierarchical classification in parallel. For details, see our paper.

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

As before, we'll map a batch with scores and labels to their respective ancestral paths:

```python
batch_sz = 100
scores = torch.randn(batch_sz, tree.n_classes)           # normally predicted by a model
labels = torch.randint(tree.n_classes, size=[batch_sz])  # targets, each a class in the tree
```

Mapping the batch incurs negligible computation and consumes only the memory occupied by data:

```python
scores_in_tree = tree.map_scores(scores)  # shape is [batch_sz, tree.n_levels, tree.n_classes]
labels_in_tree = tree.map_labels(labels)  # shape is [batch_sz, tree.n_levels]
```

## Tips for Training and Inference

### Training

When training a model, filter out padding values to flatten mapped scores into a matrix and mapped labels into a vector, for computing a classification loss (e.g., cross-entropy) at every applicable level of depth in parallel:

```python
import torch.nn.functional as F
idx = (labels_in_tree != tree.pad_value)
loss = F.cross_entropy(scores_in_tree[idx], labels_in_tree[idx])  # by level in parallel
```

### Inference

At inference, you can compute naive probability distributions at every level of depth with a single Softmax:

```python
pred_probs = scores_in_tree.softmax(dim=-1)  # [batch_sz, tree.n_levels, tree.n_classes]
```

We recommend that you map naive predicted probabilities at each level of depth to *valid predicted paths*, by restricting the space of possible paths to only those that exist in the tree, stored by `ClassTree` in a PyTorch buffer named `P` (corresponding to matrix P in the paper).

For example, here we predict the top k valid paths that have the smallest Levenshtein distance to each naively predicted path in the batch:

```python
k = 5
naive_preds = scores_in_tree.argmax(dim=-1)      # [batch_sz, tree.n_levels]
is_diff = (naive_preds[:, None, :] != tree.P)    # [batch_sz, tree.n_classes, tree.n_levels]
lev_dists = is_diff.sum(dim=-1)                  # [batch_sz, tree.n_classes]
topk = lev_dists.topk(k, largest=False, dim=-1)  # k valid paths with smallest Lev dists
topk_valid_preds = tree.P[topk.indices]          # [batch_sz, k, tree.n_levels]
```

In practice, weighting Levenshtein distances by path density at each level of tree depth works pretty well:

```python
is_node = (tree.P != tree.pad_value)                      # [tree.n_classes, tree.n_levels]
density = is_node.float().mean(dim=-2, keepdim=True)      # [1, tree.n_levels]
weighted_lev_dists = (is_diff * density).sum(dim=-1)      # [batch_sz, tree.n_classes]
topk = weighted_lev_dists.topk(k, largest=False, dim=-1)  # [batch_sz, k]
topk_valid_preds = tree.P[topk.indices]                   # [batch_sz, k, tree.n_levels]
```

Beam search over the paths of `P` with the highest joint predicted probability at each level of depth works pretty well too. You can use any of the many implementations of beam search for PyTorch available online.

## Citing

```
@misc{heinsen_tree_2022,
    title={Tree Methods for Hierarchical Classification in Parallel},
    author={Franz A. Heinsen},
    year={2022},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Notes

We originally conceived and implemented these methods as part of our AI software, nicknamed Graham. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code, clean it up, and release it as stand-alone open-source software without having to disclose any key intellectual property. Our code has been tested on Ubuntu Linux 20.04+ with Python 3.8+. We hope others find our work and our code useful.
