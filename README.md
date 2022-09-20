# heinsen_tree

Reference implementation of "Tree Methods for Hierarchical Classification in Parallel" (Heinsen, 2022), for transforming classification scores and labels, corresponding to given nodes in a semantic tree, to scores and labels corresponding to all nodes in the ancestral paths going down the tree to every given node -- efficiently, in parallel.

For example, here we apply our methods on a tree of all English-language synsets in WordNet 3.0, with 117,659 classes in 20 levels of depth, incurring negligible computation and consuming only a fixed ~0.04GB over the memory space occupied by data:

```python
import torch
from nltk.corpus import wordnet     # see nltk docs for installation
from heinsen_tree import ClassTree  # PyTorch module, can be incorporated in models

class_names = [synset.name() for synset in wordnet.all_eng_synsets()]
class_name_to_id = { name: i for i, name in enumerate(class_names) }
paths_down_tree = [
    [class_name_to_id[synset.name()] for synset in wordnet.synset(class_name).hypernym_paths()[-1]]
    for class_name in class_names
]
tree = ClassTree(paths_down_tree)

batch_sz = 100  # we'll map a batch of scores and labels to their respective ancestral paths

scores = torch.randn(batch_sz, tree.n_classes)           # normally would be predicted by a model
labels = torch.randint(tree.n_classes, size=[batch_sz])  # targets, each a class in the tree

scores_in_tree = tree.map_scores(scores)  # shape is [batch_sz, tree.n_levels, tree.n_classes]
labels_in_tree = tree.map_labels(labels)  # shape is [batch_sz, tree.n_levels]
```

## Installing

`pip install -e git+https://github.com/glassroom/heinsen_tree`

Alternatively, you can download a single file to your project directory: [heinsen_tree.py](heinsen_tree/heinsen_tree.py).

The only dependency is PyTorch.


## How Does it Work?

`ClassTree` is a PyTorch module implementing the methods we propose in [the paper](https://arxiv.org) (Heinsen, 2022). These methods are algebraically expressible as tensor transformations that common software frameworks for machine learning can execute efficiently, particularly in hardware accelerators like GPUs and TPUs, incurring negligible computation and only modest fixed memory consumption over the footprint of data. Our methods enable efficient hierarchical classification in parallel.


## Tips for Training and Inference

### Training

When training a model, filter out padding values to flatten mapped scores into a matrix and mapped labels into a vector, enabling you to computing classification loss at all applicable levels of depth in parallel. In our example with the WordNet tree:

```python
import torch.nn.functional as F
idx = (labels_in_tree != tree.pad_value)
loss = F.cross_entropy(scores_in_tree[idx], labels_in_tree[idx])  # at all applicable levels of depth
```

### Inference

At inference, you can compute naive probability distributions at each level of depth with a Softmax, but we recommend instead that you map the top-scored classes at each level of depth to *only valid predicted paths*, by restricting the space of possible paths to only those that exist in the tree, stored by `ClassTree` in a PyTorch buffer named `P` (corresponding to matrix P in the paper). For example, here we predict the top k valid paths that have the smallest Levenshtein distance to each guessed path in the batch:

```python
k = 5
naive_guessed_paths = scores_in_tree.argmax(dim=-1)           # [batch_sz, tree.n_levels]
is_different = (naive_guessed_paths[:, None, :] != tree.P)    # [batch_sz, tree.n_classes, tree.n_levels]
is_not_padding = (tree.P != tree.pad_value)                   # [tree.n_classes, tree.n_levels] 
lev_dists = (is_different & is_not_padding).sum(dim=-1)       # [batch_sz, tree.n_classes]
topk_idxs = lev_dists.topk(k, largest=False, dim=-1).indices  # [batch_sz, k]
topk_valid_preds = tree.P[topk_idxs]                          # [batch_sz, k, tree.n_levels]
```

In practice, we have found that weighting Levenshtein distances by path density at each level of depth works well:

```python
dens = is_not_padding.float().mean(dim=-2, keepdim=True)                        # [1, tree.n_levels]
topk_weighted_idxs = (lev_dists * dens).topk(k, largest=False, dim=-1).indices  # [batch_sz, k]
topk_weighted_valid_preds = tree.P[topk_weighted_idxs]                          # [batch_sz, k, tree.n_levels]
```

Standard beam search over the paths of `P` with the highest joint predicted probability at each level of depth works well too.

## Citing

```
@misc{heinsen_tree_2022,
    title={Tree Methods for Hierarchical Classification in Parallel},
    author={Franz A. Heinsen},
    year={2022},
    eprint={####.#######},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Notes

We originally conceived and implemented these methods as part of our AI software, nicknamed Graham. Most of the original work we do at GlassRoom tends to be either proprietary in nature or tightly coupled to internal code, so we cannot share it with outsiders. In this case, however, we were able to isolate our code, clean it up, and release it as stand-alone open-source software without having to disclose any key intellectual property. Our code has been tested on Ubuntu Linux 20.04 with Python 3.8+. We hope others find our work and our code useful.
