import numpy as np
import re
import codecs
from NLTKUtils import *
from EditDistance import *
from TweetProcess import *
from CorpusProcess import *
import sys


if len(sys.argv) < 3:
    print """
usage: python process_corpus.py <c2v_param> <corpus> [N]"

   c2v_param : path to the .param file  of a pretrained contex2vec model.
               Tested on ukwac : http://irsrv2.cs.biu.ac.il/downloads/context2vec/context2vec.ukwac.model.package.tar.gz

   corpus : path to the corpus file

   N : number of tweet to process. -1 to process all (might take long)
           defaults to 1000

example: python process_corpus.py path/to/context2vec.ukwac.model.params \
         path/to/CorpusBataclan_en.1M.raw.txt
"""
    sys.exit(0)

# Load context2vec

print('***************************  LOAD CONTEXT2VEC  *********************************')

from ContextSimilarity import *
cs = ContextSimilarity(sys.argv[1])

print('*****************************  LOAD CORPUS  ************************************')

# Process input
with codecs.open(sys.argv[2],'r',encoding='utf8') as f:
    data = f.readlines()

N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
data = data[:N]

tweets = html_decode(data)

tweets, cutmarks = reconstruct_tweets(tweets)
tweets = merge_tweets_dp(tweets)
tweets = remove_cut_words(tweets, cutmarks)

print('*****************************  PROCESSING  *************************************')

print('# Processing links')
tweets = process_links(tweets, remove=True)
print('# Processing emojis')
tweets = process_emojis(tweets, remove=True)
print('# Processing names')
tweets = process_names(tweets, remove=True)
print('# Processing hash tags')
tweets = process_hashtags(tweets, remove=True)
print('# Removing non ascii characters')
tweets = clean_non_ascii(tweets)
print('# Changing case')
tweets = process_case(tweets)
print('# Applying Lexical normalisation dictionary')
# Adapted from http://www.smart-words.org/abbreviations/text.html
# and http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
tweets = normalization_dictionaries(tweets, ['data/Test_Set_3802_Pairs.txt', 'data/short_abbrev_list.txt'])
print('# Processing contractions')
tweets = process_contractions(tweets)

print('# NLTK Tokenization')
token_tweets = tokenize(tweets)

print('# NLTK Pos tagging')
tweets_tags = pos_tagging(token_tweets)

print('*******************************  OUTPUT  ***************************************')

final_tokens = cs.process_context_similarity(token_tweets, tweets_tags, prt=True)
