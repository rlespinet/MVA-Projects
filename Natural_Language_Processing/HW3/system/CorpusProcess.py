import numpy as np

class TweetConstructor:

    def __init__(self, max_tweet_len):
        self.max_tweet_len = max_tweet_len

    def begin(self):
        self.tweets = []
        self.cutmarks = []
        self.current = []
        self.current_len = 0

    def end(self):
        self._push_tweet()

    def _push_tweet(self, cut=False):

        if self.current_len > 0:
            self.tweets.append(self.current)
            self.cutmarks.append(cut)
            self.current = []
            self.current_len = 0

    def _add_current(self, line):
        self.current.append(line)
        self.current_len += len(line)

    def _clean_line(self, line):
        line = line.strip()

        rt = False
        cut = False

        if len(line) > 2 and line[:2] == u'RT':
            rt = True
            line = line[2:]

        if len(line) > 1 and line[-1] == u'\u2026':
            cut = True
            line = line[:-1]

        line = line.strip()
        return (line, rt, cut)


    def add_line(self, line):

        line, rt, cut = self._clean_line(line)

        line = line.strip()

        if rt:
            self._push_tweet()

        self._add_current(line)

        if cut:
            self._push_tweet(cut=True)


from HTMLParser import HTMLParser

def html_decode(data):

    processed_data = []

    h = HTMLParser()
    for line in data:
        processed_data.append(h.unescape(line))

    return processed_data

def reconstruct_tweets(data):
    tc = TweetConstructor(140)

    tc.begin()

    for line in data:
        tc.add_line(line)

    tc.end()

    return tc.tweets, tc.cutmarks

def merge_tweets(tweets):
    data = []
    for tweet in tweets:
        data.append((u"\n").join(tweet))
    return data


# Merge the tweets based on a heuristic (dynamic programming approach)
def merge_tweets_dp(tweets):

    max_len = 141
    merged_tweets = []
    for tweet in tweets:

        W = np.zeros((len(tweet), len(tweet)), dtype=float)
        I = -np.ones((len(tweet), len(tweet)), dtype=int)
        L = np.array([len(w) for w in tweet]) + 1
        C = np.hstack((0, np.cumsum(L)))
        C[-1] -= 1

        n = np.ceil(float(C[-1]) / max_len)

        for d in range(0, len(tweet)):
            for i in range(len(tweet) - d):
                if C[i + d + 1] - C[i] <= max_len:
                    W[i, i + d] = np.abs(C[-1] / n - (C[i + d + 1] - C[i]))
                else:
                    I[i, i + d] = np.argmin(W[i, i:(i+d)] + W[(i+1):(i+d+1), i + d]) + 1
                    W[i, i + d] = np.min(W[i, i:(i+d)] + W[(i+1):(i+d+1), i + d])

        merged_tweet = []
        d = 0
        while True:
            i = I[d, -1]
            if i == -1:
                break
            d += i
            merged_tweet.append(u'\n'.join(tweet[:i]))
            tweet = tweet[i:]
        merged_tweet.append(u'\n'.join(tweet))

        merged_tweets.extend(merged_tweet)

    return merged_tweets

def remove_cut_words(tweets, cutmarks):
    cleaned_tweets = []
    for tweet, cut in zip(tweets, cutmarks):
        if cut:
            id = tweet.rfind(' ')
            tweet = tweet[:id]
        cleaned_tweets.append(tweet)
    return cleaned_tweets
