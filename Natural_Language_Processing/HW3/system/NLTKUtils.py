import nltk

def tokenize(tweets):
    token_tweets = []
    for tweet in tweets:
        token_tweets.append(nltk.word_tokenize(tweet))
    return token_tweets

def pos_tagging(token_tweets):
    tags = []
    for tokens in token_tweets:
        tagged = nltk.pos_tag(tokens)
        tags.append([t[1] for t in tagged])
    return tags
