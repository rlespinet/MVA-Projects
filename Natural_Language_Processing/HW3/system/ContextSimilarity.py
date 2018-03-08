import numpy as np

from context2vec.common.model_reader import ModelReader
from EditDistance import *

def clean_dictionary(words):

    # Clean dictionary
    cleaned_words = []

    dic = {}
    for word in nltk.corpus.words.words():
        dic[word] = 1

    for word in words:

        word = word.lower()
        if re.match('^\w+$', word) is None:
            continue

        if word not in dic:
            continue

        cleaned_words.append(word)

    return np.array(cleaned_words)


class ContextSimilarity:

    def __init__(self, model_param_file):
        model_reader = ModelReader(model_param_file)
        self.w = model_reader.w
        self.word2index = model_reader.word2index
        self.index2word = model_reader.index2word
        self.model = model_reader.model

        self.cleaned_words = clean_dictionary(self.index2word)
        self.T = construct_tree(self.cleaned_words)

    def process_context_similarity(self, token_tweets, tweets_tags, prt=False):

        edit_dist = DamerauLevenshtein()

        aff_edit = [1, 0.4, 0.2]

        tokens_list = []

        for token_list, tag_list in zip(token_tweets, tweets_tags):
            processed_tokens = []
            for j, (word, tag) in enumerate(zip(token_list, tag_list)):

                word = word.lower()

                if tag == 'NNP' or len(tag) < 2:
                    processed_tokens.append(word)
                    continue


                contextv = self.model.context2vec(token_list, j)
                contextv = np.dot(self.w, contextv)

                ids = candidate_edit_dist(self.T, word)
                candidates = self.cleaned_words[ids]

                best_aff = 0
                for candidate in candidates:

                    ed = edit_dist.dist(word, candidate)
                    if ed > 2: continue

                    target = aff_edit[ed]
                    context = contextv[self.word2index[candidate]]

                    aff = context * target
                    if aff > best_aff:
                        best_aff = aff
                        best_candidate = candidate

                if best_aff == 0 or best_aff < 0.1:
                    best_candidate = word
                    #         else:
                    #             print word + " -> " + best_candidate + " " + str(best_aff)

                processed_tokens.append(best_candidate)

            if prt:
                print(' '.join(processed_tokens))
            tokens_list.append(processed_tokens)

        return tokens_list
