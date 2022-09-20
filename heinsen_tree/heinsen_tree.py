# coding: utf-8
import torch
import torch.nn as nn

class ClassTree(nn.Module):
    """
    Maps predicted scores and integer labels, corresponding to given nodes in
    a semantic tree, to all nodes in the ancestral paths ending at every given
    node, incurring negligible computation and consuming a modest fixed amount
    of memory over the footprint of data, as described in "Tree Methods for
    Hierarchical Classification in Parallel" (Heinsen, 2022).

    Args:
        paths_down_tree: list, each element of which is a list of int labels
            corresponding to the ancestral labels going down the tree to each
            class. The paths must be in ascending order, starting with class 0.
        pad_value: (optional) int, for padding paths. Default: -1.
        min_score: (optional) float, for masking scores. Default: float('-inf').

    Methods:
        Use methods `map_scores` and `map_labels`, respectively, to map scores
        and labels. See each method's docstring for details. For convenience,
        the default `forward` method is a shim that calls `map_scores`.

    Sample usage:
        >>> # Instantiate this semantic tree:
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
        >>> print(labels_in_tree)
        >>> > tensor([[0,  3,  4],
        >>> >         [1, -1, -1],
        >>> >         [0,  3,  5]]
        >>> >         [0,  2, -1]])
    """
    def __init__(self, paths_down_tree, pad_value=-1, min_score=float('-inf')):
        super().__init__()
        assert all(label == path[-1] for label, path in enumerate(paths_down_tree)), \
            "Paths going down the tree must be sorted by class in ascending order, starting with the path to class 0."
        n_classes, n_levels = (len(paths_down_tree), max(len(path) for path in paths_down_tree))
        padded_paths = torch.tensor([path + [pad_value]*(n_levels - len(path)) for path in paths_down_tree])
        self.n_classes, self.n_levels = (n_classes, n_levels)
        self.register_buffer('M', padded_paths.T != torch.arange(n_classes))  # [n_levels, n_classes] (matrix M in paper)
        self.register_buffer('P', padded_paths)                               # [n_classes, n_levels] (matrix P in paper)
        self.register_buffer('pad_value', torch.tensor(pad_value))
        self.register_buffer('min_score', torch.tensor(min_score))

    def __repr__(self):
        cfg_str = ', '.join(f'{s}={getattr(self, s)}' for s in 'n_classes n_levels pad_value min_score'.split())
        return '{}({})'.format(self._get_name(), cfg_str)

    def map_scores(self, scores):
        """
        Map a tensor of scores [..., n_classes] to a tensor of scores for each
        level in the tree [..., n_levels, n_classes], masked as necessary with
        min_score, where '...' denotes zero or more preserved dimensions.
        """
        return scores.unsqueeze(-2).masked_fill(self.M, self.min_score)  # equation (6) in paper

    def map_labels(self, labels):
        """
        Map a tensor of int labels [...] to their ancestral paths at each level
        of the tree [..., n_levels], padded as necessary with pad_value, where
        '...' denotes one or more preserved dimensions.
        """
        return self.P[labels, :]  # equation (9) in paper

    def forward(self, scores):
        return self.map_scores(scores)
