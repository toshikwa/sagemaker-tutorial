import numpy as np


class Node:

    def __init__(self, left=None, right=None, mean=None, item=None, key=None):
        self.left = left
        self.right = right
        n_left, n_right = [
            getattr(child, 'n_item', 0) for child in (self.left, self.right)]

        if (n_left + n_right) > 0:
            self.mean = ((getattr(left, 'mean', 0) * n_left + getattr(right, 'mean', 0) * n_right) / (n_left + n_right))
            self.n_item = n_left + n_right
        elif item is not None:
            self.mean = mean
            self.n_item = 1
        else:
            self.mean = None
            self.n_item = 0

        self.list = []
        self.item = item if item is not None else {}

    def to_list(self):
        if not self.list:
            if self.item:
                self.list.append(self.item)
            for child in (self.left, self.right):
                if child is not None:
                    self.list.extend(child.to_list())
        return self.list

    def get_closer_children(self, value):
        dist_left, dist_right = (np.linalg.norm(child.mean.ravel()-value.ravel()) for child in (self.left, self.right))
        return [self.left, self.right][dist_left >= dist_right]

    def binary_search(self, value, n_item=4, verbose=False):
        if verbose:
            print(self.binary_search(self.mean, n_item=1, verbose=False)[0]['sentence'] + '.', end='\n\n')
        if self.n_item <= n_item:
            return self.to_list()
        child = self.get_closer_children(value)
        return child.binary_search(value, n_item, verbose)
