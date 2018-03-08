from TreeParsing import *
from TreeProcessing import *
from RulesProcessing import *
from PCYK import *

from sys import stdout, stdin, argv

class Message:

    def __init__(self):
        self.width = 79
        self.space = 1
        self.last = None

    def _print(self):
        if self.last:
            print('.' * (self.width - 4 - self.space - self.last % self.width), end='')
            print('[OK]')

    def message(self, string):
        self._print()
        print(' ' * self.space, end='')
        print(string, end='')
        self.last = len(string)
        stdout.flush()

    def term(self):
        self._print()


def handle_args(argv):

    args = {
        'accuracy': False,
        'graphviz': False,
        'corpus': "data/sequoia-corpus+fct.mrg_strict"
    }

    not_opt = []
    for arg in argv:
        if arg == '--accuracy':
            args['accuracy'] = True
        elif arg == '--graphviz':
            args['graphviz'] = True
        else:
            if arg[0] == '-':
                raise ValueError("Unrecognized option : %s" % arg)
            else:
                not_opt.append(arg)

    if len(not_opt) >= 1:
        print("usage: python Run.py [--graphviz] [--accuracy] [corpus]")
        print()
        print("   --graphviz : Ouptuts tree in the graphviz format instead of ascii")
        print("   --accuracy : Run the program in accuracy mode (instead of the default")
        print("                interactive mode)")
        print("     corpus   : Specify a diffrent corpus file (default'")
        print()

    if len(not_opt) == 1:
        args['corpus'] = not_opt[0]

    return args

args = handle_args(argv[1:])

msg = Message()

msg.message("Opening file %s" % args['corpus'])
with open(args['corpus'], 'r') as f:
    raw_corpus = f.readlines()

msg.message("Construction trees")
forest = parse_forest(raw_corpus)

msg.message("Removing functional labels")
remove_functional(forest)

msg.message("CNF Normalization")
CNF(forest)

msg.message("Forest annotation")
annotate_forest(forest)

if not args['accuracy']:

    msg.message("Generating grammar and lexicon")
    grammar, lexicon = generate_rules(forest)

    # Uncomment to reactivate training with unknown words
    # msg.message("Adding unknown tokens")
    # add_unknown(forest, lexicon, 1)
    # grammar, lexicon = generate_rules(forest)

    msg.message("Compute grammar and lexicon probabilities")
    train_grammar = compute_grammar_probabilities(grammar)
    lexicon, grammar_terminals = compute_lexicon_probabilities(lexicon)

    msg.term()

    print('Enter your text (in the language of the corpus) :')

    while 1:
        try:
            stdout.write("I> ")
            stdout.flush()
            sentence = stdin.readline()
        except KeyboardInterrupt:
            break

        if not sentence:
            break

        tokens = sentence.strip().split(' ')

        predicted_forest = PCKY([tokens], train_grammar, lexicon, grammar_terminals)
        un_CNF(predicted_forest)

        if args['graphviz']:
            print(Graphviz(predicted_forest[0]))
        else:
            print(predicted_forest[0])

else:

    msg.message("Splitting test and train")
    train_r = 0.8
    valid_r = 0.2
    # test_r  = 0.1

    # np.random.shuffle(forest)

    forest_len = len(forest)
    N_train = int(forest_len * train_r)
    N_valid = int(forest_len * valid_r)

    train_forest = forest[:N_train]
    valid_forest = forest[N_train:(N_train+N_valid)]
    # test_forest  = forest[(N_train+N_valid):]

    msg.message("Generating grammar and lexicon")
    rules, lexicon = generate_rules(forest)
    train_grammar, _ = generate_rules(train_forest)

    msg.message("Retrieve sentences")
    train_sentences = retrieve_sentences(train_forest)
    valid_sentences = retrieve_sentences(valid_forest)
    # test_sentences  = retrieve_sentences(test_forest)

    msg.message("Compute grammar and lexicon probabilities")
    train_grammar = compute_grammar_probabilities(train_grammar)
    lexicon, grammar_terminals = compute_lexicon_probabilities(lexicon)

    max_len = 14
    msg.message("Extracting validation sentences with length less than %d" % max_len)

    valid_forest_reduced = []
    valid_sentences_reducted = []
    for sentence, tree in zip(valid_sentences, valid_forest):
        if (len(sentence) < max_len):
            valid_forest_reduced.append(tree)
            valid_sentences_reducted.append(sentence)

    msg.message("Computing PCKY")
    predicted_forest = PCKY(valid_sentences_reducted, train_grammar, lexicon, grammar_terminals)

    msg.message("Reconstructing tree (un-CNF)")
    un_CNF(predicted_forest)
    un_CNF(valid_forest)

    msg.message("Computing accuracy")
    acc = 0
    total = 0
    for predicted_tree, truth_tree in zip(predicted_forest, valid_forest_reduced):
        if str(predicted_tree) == str(truth_tree):
            acc += 1
        total += 1

    msg.term()

    print("Accuracy obtained : %f" % (acc / total))
