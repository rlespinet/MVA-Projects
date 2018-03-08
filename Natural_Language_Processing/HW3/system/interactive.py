import numpy as np
import re
import codecs
from NLTKUtils import *
from EditDistance import *
from TweetProcess import *
from CorpusProcess import *
import sys


if len(sys.argv) < 2:
    print """
usage: python interactive.py <c2v_param>"

   c2v_param : path to the .param file  of a pretrained contex2vec model. Tested on ukwac :
               http://irsrv2.cs.biu.ac.il/downloads/context2vec/context2vec.ukwac.model.package.tar.gz

example: python interactive.py path/to/context2vec.ukwac.model.params
"""
    sys.exit(0)

# Load context2vec
print('***************************  LOAD CONTEXT2VEC  *********************************')

from ContextSimilarity import *
cs = ContextSimilarity(sys.argv[1])

print('********************************************************************************')

print('Enter your text :')

while 1:
    try:
        sys.stdout.write("I> ")
        tweet = sys.stdin.readline()
    except KeyboardInterrupt:
        break

    if not tweet:
        break

    tweets = html_decode([tweet])

    tweets = process_links(tweets, remove=True)
    tweets = process_emojis(tweets, remove=True)
    tweets = process_names(tweets, remove=True)
    tweets = process_hashtags(tweets, remove=True)
    tweets = clean_non_ascii(tweets)
    tweets = process_case(tweets)
    # Adapted from http://www.smart-words.org/abbreviations/text.html
    # and http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
    tweets = normalization_dictionaries(tweets, ['data/Test_Set_3802_Pairs.txt', 'data/short_abbrev_list.txt'])
    tweets = process_contractions(tweets)

    token_tweets = tokenize(tweets)

    tweets_tags = pos_tagging(token_tweets)

    final_tokens = cs.process_context_similarity(token_tweets, tweets_tags)
    sys.stdout.write("O> ")
    sys.stdout.write(' '.join(final_tokens[0]))
    sys.stdout.write('\n')
