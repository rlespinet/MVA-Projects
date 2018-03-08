import math

from ParseTree import *

def add_rule_to(grammar, left, right):
    if left not in grammar:
        grammar[left] = {}

    if right not in grammar[left]:
        grammar[left][right] = 0

    grammar[left][right] += 1

def generate_rules_rec(tree, grammar, lexicon):

    if tree.min_leaf_dist == 1:

        assert(len(tree.children) == 1)

        word = tree.children[0].root
        add_rule_to(lexicon, tree.root, word)

    else:
        rule = []
        for child in tree.children:
            generate_rules_rec(child, grammar, lexicon)
            rule.append(child.root)

        rule_str = ' '.join(rule)
        add_rule_to(grammar, tree.root, rule_str)

def generate_rules(forest):

    grammar = {}
    lexicon = {}
    for tree in forest:
        generate_rules_rec(tree, grammar, lexicon)

    return grammar, lexicon


def compute_grammar_probabilities(grammar, threshold=1e-16):

    grammar_prob = {}

    for left in grammar:

        grammar_prob[left] = []

        total = 0
        for right in grammar[left]:
            total += grammar[left][right]

        for right in grammar[left]:
            log_prob = math.log(grammar[left][right]) - math.log(total)
            if log_prob > math.log(threshold):
                grammar_prob[left].append((right.split(' '), log_prob))

    return grammar_prob

def compute_lexicon_probabilities(lexicon, threshold=1e-16):

    lexicon_prob = {}
    left_counts = {}

    total = 0
    for left in lexicon:
        count = 0
        for right in lexicon[left]:
            count += lexicon[left][right]
        left_counts[left] = count
        total += count

    left_symb_prob = {}
    for left in left_counts:
        left_symb_prob[left] = math.log(left_counts[left]) - math.log(total)

    for left in lexicon:

        lexicon_prob[left] = []

        for right in lexicon[left]:
            log_prob = math.log(lexicon[left][right]) - math.log(left_counts[left])
            if log_prob > math.log(threshold):
                lexicon_prob[left].append((right, log_prob))

    return lexicon_prob, left_symb_prob
