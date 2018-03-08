import numpy as np
import nltk
import re

class Tree:

    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.letter = None

# Constructs a greedy tree
def construct_tree_rec(U, ids, alphamap, words):
    if U.shape[0] < 20 or U.shape[1] <= 1:
        T = Tree()
        T.data = ids
        return T

    bestd = np.inf
    bests = 0
    besti = 0

    for i, u in enumerate(U.T):
        s = int(np.sum(u))
        d = np.abs(len(U) / 2 - s)
        if (d < bestd):
            bests = s
            bestd = d
            besti = i

    T = Tree()
    T.letter = alphamap[besti]

    reorder = np.argsort(U.T[besti])
    sep = len(U) - bests

    alphamap = np.delete(alphamap, besti)
    U = np.delete(U[reorder], besti, axis=1)
    ids = ids[reorder]

    Ul = U[:sep]
    Ur = U[sep:]
    idsl = ids[:sep]
    idsr = ids[sep:]

    if (len(idsl) > 0): T.left = construct_tree_rec(Ul, idsl, alphamap, words)
    if (len(idsr) > 0): T.right = construct_tree_rec(Ur, idsr, alphamap, words)
    return T

def candidate_edit_dist(T, word, edit=2):

    if edit < 0:
        return np.array([], dtype=int)

    if T.left is None and T.right is None:
        return T.data if T.data is not None else np.array([], dtype=int)

    if T.left is None:
        left = np.array([], dtype=int)
    elif T.letter in word:
        left = candidate_edit_dist(T.left, word, edit-1)
    else:
        left = candidate_edit_dist(T.left, word, edit)

    if T.right is None:
        right = np.array([], dtype=int)
    elif T.letter in word:
        right = candidate_edit_dist(T.right, word, edit)
    else:
        right = candidate_edit_dist(T.right, word, edit-1)

    return np.hstack((left, right))

def construct_tree(cleaned_words):

    cleaned_words = np.array(cleaned_words)

    alphabet = {key: 1 for key in ''.join(cleaned_words)}
    alphamap = np.empty(len(alphabet), dtype='|S1')
    for i, key in enumerate(alphabet.keys()):
        alphabet[key] = i
        alphamap[i] = key

    U = np.empty((len(cleaned_words), len(alphabet)))
    for id, word in enumerate(cleaned_words):
        vec = np.zeros(len(alphabet))
        for l in word:
            vec[alphabet[l]] = 1

        U[id] = vec

    ids = np.arange(len(cleaned_words), dtype=int)
    T = construct_tree_rec(U, ids, alphamap, cleaned_words)
    return T


# This algorithm has been adapted from the pseudo-code given
# by https://en.wikipedia.org/wiki/Levenshtein_distance
class Levenshtein:

    def __init__(self):
        self.m = 1
        self.n = 1
        self.init_dist()

    def init_dist(self):
        self.d = np.empty((self.m + 1, self.n + 1), dtype=int)
        self.d[0, 0] = 0
        self.d[1:, 0] = np.arange(self.m) + 1
        self.d[0, 1:] = np.arange(self.n) + 1

    def dist(self, s, t):

        m, n = len(s), len(t)

        # Reallocate and initialize only if necessary
        if m > self.m or n > self.n:
            self.m = max(self.m, m)
            self.n = max(self.n, n)
            self.init_dist()

        for j in range(n):
            for i in range(m):
                cost = 0 if t[j] == s[i] else 1
                self.d[i+1, j+1] = min(self.d[i, j+1] + 1,  # deletion
                                       self.d[i+1, j] + 1,  # insertion
                                       self.d[i, j] + cost) # substitution
        return self.d[m, n]

class DamerauLevenshtein:


    def __init__(self):
        self.m = 1
        self.n = 1
        self.init_dist()

    def init_dist(self):
        self.d = np.empty((self.m + 1, self.n + 1), dtype=int)
        self.d[0, 0] = 0
        self.d[1:, 0] = np.arange(self.m) + 1
        self.d[0, 1:] = np.arange(self.n) + 1

    def dist(self, s, t):

        m, n = len(s), len(t)

        # Reallocate and initialize only if necessary
        if m > self.m or n > self.n:
            self.m = max(self.m, m)
            self.n = max(self.n, n)
            self.init_dist()

        for j in range(n):
            for i in range(m):
                cost = 0 if t[j] == s[i] else 1
                self.d[i+1, j+1] = min(self.d[i, j+1] + 1,  # deletion
                                       self.d[i+1, j] + 1,  # insertion
                                       self.d[i, j] + cost) # substitution
                if i > 0 and j > 0 and s[i] == t[j-1] and s[i-1] == t[j]:
                    self.d[i+1, j+1] = min(self.d[i+1, j+1],
                                           self.d[i-1, j-1] + cost)
        return self.d[m, n]
