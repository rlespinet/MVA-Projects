import numpy as np

from ParseTree import *

def reconstruct_rec(words, P, name):

    _, k, prev = P[0, -1][name]

    tree = ParseTree(name)
    if prev == name:
        assert(len(words) == 1)
        tree.children.append(ParseTree(words[0]))
        return tree

    if len(prev) == 1:
        tree.children.append(reconstruct_rec(words, P, prev[0]))
    elif len(prev) == 2:
        tree.children.append(reconstruct_rec(words[:(k+1)], P[:(k+1), :(k+1)], prev[0]))
        tree.children.append(reconstruct_rec(words[(k+1):], P[(k+1):, (k+1):], prev[1]))

    return tree


def reconstruct(words, P):
    N = len(words)

    if 'SENT' not in P[0, -1]:
        return None

    return reconstruct_rec(words, P, 'SENT')


def PCKY_words(words, grammar, lexicon, gram_terminals, unit_prod, rev_grammar, rev_lexicon):

    N = len(words)

    P=np.empty((N, N), dtype=object)

    for j, word in enumerate(words):
        P[j, j] = {}

        if word not in rev_lexicon:
            # Unknown word ! Handle this later using lexicon model
            for left in gram_terminals:
                P[j, j][left] = (0, -1, left)
                # P[j, j][left] = (unknown_prob[left], -1, left)
            # P[j, j]['UNK'] = (0.0, -1, 'UNK')
        else:
            for left, p in rev_lexicon[word].items():
                P[j, j][left] = (p, -1, left)

        # Handle unit production
        for left in list(P[j, j]):

            if left not in unit_prod:
                continue

            for sym in unit_prod[left]:
                p_sym, orig = unit_prod[left][sym]

                pF = p_sym + P[j, j][left][0]
                P[j, j][sym] = (pF, -1, (orig,))

        for i in range(j - 1, -1,-1):

            P[i, j] = {}
            for k in range(i, j):
                for sym1 in P[i, k]:
                    for sym2 in P[k+1, j]:
                        lookup_str = sym1 + ' ' + sym2
                        if lookup_str not in rev_grammar:
                            continue

                        p1, _, orig1 = P[i,   k][sym1]
                        p2, _, orig2 = P[k+1, j][sym2]
                        for left, pG in rev_grammar[lookup_str]:

                            pF = p1 + p2 + pG
                            if left not in P[i, j] or P[i, j][left][0] < pF:
                                P[i, j][left] = (pF, k - i, (sym1, sym2))


            # TODO REMOVE
#             med = np.median([v[0] for v in P[i, j].values()])

            # Handle unit production
            for left in list(P[i, j]):

                if left not in unit_prod:
                    continue

                for sym in unit_prod[left]:
                    p_sym, orig = unit_prod[left][sym]

                    pF = p_sym + P[i, j][left][0]

                    # TODO REMOVE
#                     if pF < med - np.log(10e6):
#                         continue

                    if sym not in P[i, j] or P[i, j][sym][0] < pF:
                        P[i, j][sym] = (pF, -1, (orig,))

    return reconstruct(words, P)


def PCKY(sentences, grammar, lexicon, gram_terminals):

    # Identify the right of unit production rules
    # Precompute reverse tables
    unit_prod = {}
    rev_grammar = {}
    for left in grammar:
        for i, (right, p) in enumerate(grammar[left]):
            if len(right) == 1:
#                 print(left, '->', right)
                right = right[0]

                if right not in unit_prod:
                    unit_prod[right] = {}

                if left not in unit_prod[right]:
                    unit_prod[right][left] = (p, right)
            else:
                right_str = ' '.join(right)
                if right_str not in rev_grammar:
                    rev_grammar[right_str] = []

                rev_grammar[right_str].append((left, p))

    rev_lexicon = {}
    for left in lexicon:
        for right, prob in lexicon[left]:
            if right not in rev_lexicon:
                rev_lexicon[right] = {}

            if left not in rev_lexicon[right]:
                rev_lexicon[right][left] = prob
            else:
                raise("Rule present twice in the lexicon")


    # Expand unit rules
    while True:
        change = False
        for right in list(unit_prod):

            for center in list(unit_prod[right]):

                p_center, orig_center = unit_prod[right][center]

                if center not in unit_prod:
                    continue

                for left in list(unit_prod[center]):

                    p_left, orig_left = unit_prod[center][left]

                    add = False
                    if left not in unit_prod[right]:
                        add = True
                    else:
                        p_present, orig_present = unit_prod[right][left]
                        add = (p_left + p_center > p_present)

                    if add:
                        change = True
                        unit_prod[right][left] = (p_left + p_center, center)

        if change == False:
            break

    forest = []
    for sentence in sentences:

        tree = PCKY_words(sentence, grammar, lexicon, gram_terminals, unit_prod, rev_grammar, rev_lexicon)
        forest.append(tree)

    return forest
