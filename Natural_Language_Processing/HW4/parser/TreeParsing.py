import re

from ParseTree import *

def read_rule(tokens):

    if tokens[0] != '(':
        print("Error expected '('")
        return -1, None

    root = ParseTree(tokens[1])

    i = 2
    j, root.children = read_rules(tokens[i:])
    if j < 0:
        print("Error while parsing ", tokens[1])
        return -1, None

    i += j

    if tokens[i] != ')':
        print("Error expected ')'")
        return -1, None

    i += 1

    return i, root

def read_rules(tokens):

    if len(tokens) == 0:
        return 1, []

    i = 0

    if tokens[0] != '(':
        return 1, [ParseTree(tokens[0])]

    forest = []
    while i < len(tokens) and tokens[i] == '(':
        j, child = read_rule(tokens[i:])

        if j < 0:
            # Error case
            return -1, None

        i += j

        forest.append(child)

    return i, forest


def parse_forest(corpus):

    split_re = re.compile("( |[\(\)])")
    filter_re = re.compile("\S+")

    forest_all = []

    for line in corpus:

        tokens = list(filter(filter_re.match, split_re.split(line)))

        if (tokens[0] != '(' or tokens[-1] != ')'):
            print("Error with tokenized phrase :", tokens)
            continue

        forest = []

        i, forest = read_rules(tokens[1:-1])
        if i < 0:
            print("Error with tokenized sequence :", tokens)
            continue

        forest_all.extend(forest)

    return forest_all
