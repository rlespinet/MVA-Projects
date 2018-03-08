import math
import re

from ParseTree import *

def remove_functional_rec(tree):

    if tree.terminal:
        return

    for child in tree.children:
        remove_functional_rec(child)

    tree.root = re.sub('\W.*$', '', tree.root)

def remove_functional(forest):

    for tree in forest:
        remove_functional_rec(tree)

def annotate_tree_rec(tree, depth):

    if tree.terminal:
        tree.min_leaf_dist = 0
        tree.max_leaf_dist = 1
        tree.depth = depth
        return

    min_leaf_dist = math.inf
    max_leaf_dist = 0

    for child in tree.children:
        annotate_tree_rec(child, depth+1)

        min_leaf_dist = min(min_leaf_dist, child.min_leaf_dist + 1)
        max_leaf_dist = max(max_leaf_dist, child.max_leaf_dist + 1)

    tree.min_leaf_dist = min_leaf_dist
    tree.max_leaf_dist = max_leaf_dist
    tree.depth = depth

def annotate_forest(forest):

    for tree in forest:
        annotate_tree_rec(tree, 0)

def retrieve_sentence_tree_rec(tree):

    if tree.terminal:
        return [tree.root]

    sentence = []
    for child in tree.children:
        sentence.extend(retrieve_sentence_tree_rec(child))

    return sentence

def retrieve_sentences(forest):
    sentences = []
    for tree in forest:
        sentences.append(retrieve_sentence_tree_rec(tree))

    return sentences

def CNF_tree_rec(tree):

    if tree.terminal:
        return

    for child in tree.children:
        CNF_tree_rec(child)

    while len(tree.children) > 2:

        right_tree = tree.children.pop()
        left_tree = tree.children.pop()

        new_tree = ParseTree(left_tree.root + '|' + right_tree.root)
        new_tree.children.append(left_tree)
        new_tree.children.append(right_tree)

        tree.children.append(new_tree)

def un_CNF_tree_rec(tree):

    if tree.terminal:
        return

    if len(tree.children) == 2:

        while '|' in tree.children[-1].root:

            right_tree = tree.children.pop()

            right_sub_tree = right_tree.children.pop()
            left_sub_tree = right_tree.children.pop()

            tree.children.append(left_sub_tree)
            tree.children.append(right_sub_tree)


    for child in tree.children:
        un_CNF_tree_rec(child)


def CNF(forest):

    for tree in forest:
        CNF_tree_rec(tree)

def un_CNF(forest):

    for tree in forest:
        un_CNF_tree_rec(tree)


def replace_unknown_rec(tree, unknown_words):

    if tree.min_leaf_dist == 1:
        assert(len(tree.children) == 1)

        if tree.children[0].root in unknown_words:
            tree.root = 'UNK'
    else:
        for child in tree.children:
            replace_unknown_rec(child, unknown_words)

def add_unknown(forest, lexicon, num):

    words_count = {}

    for left in lexicon:
        for right in lexicon[left]:

            if right not in words_count:
                words_count[right] = 0

            words_count[right] += lexicon[left][right]

    unknown_words = set()
    for word in words_count:
        if words_count[word] <= num:
            unknown_words.add(word)

    for tree in forest:
        replace_unknown_rec(tree, unknown_words)
