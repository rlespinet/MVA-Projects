import numpy as np
import re

def link_hook(string):
    return "<LINK:" + string.group(0) + ">"

def emoji_hook(string):
    return '<EMOJIS:' + string.group().encode('unicode-escape') + '>'

def name_hook(string):
    return '<NAME:' + string.group(1) + '>'

def hashtag_hook(string):
    return '<HASH:' + string + '>'

def process_links(lines, remove=False):

    process = '' if remove else link_hook

    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub('(https?://\S*)', process, line)
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

def process_names(lines, remove=False):

    process = '' if remove else name_hook

    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub('@(\w+):?', process, line)
        cleaned_lines.append(cleaned_line)
    return cleaned_lines


def process_hashtags(lines, remove=False, heuristic=True):

    cleaned_lines = []
    for line in lines:
        words = re.split('(#\w+)', line)
        for i, word in enumerate(words):
            if len(word) == 0 or word[0] != '#':
                continue

            if heuristic and re.match("^#(([A-Z]+)|([a-zA-Z][a-z]+))$", word):
                words[i] = word[1:]
            elif remove:
                words[i] = ''
            else:
                hashtag_hook(word)

        cleaned_lines.append(''.join(words))
    return cleaned_lines

def process_emojis(lines, remove=False):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    process = '' if remove else emoji_hook

    cleaned_lines = []
    for line in lines:
        cleaned_line = emoji_pattern.sub(process, line)
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

def case_hook(string):
    return string.group(0) if len(string.group(0)) == 1 else string.group(0).lower()

def process_case(lines):
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"[A-Z']+", case_hook, line, re.UNICODE)
        cleaned_lines.append(line)
    return cleaned_lines

def process_case(lines):
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(line.lower())
    return cleaned_lines

def process_contractions(lines):
    cleaned_lines = []
    for line in lines:
        # From stackoverflow
        # https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/19794953#19794953
        line = re.sub(r"won't", "will not", line, re.UNICODE)
        line = re.sub(r"can\'t", "can not", line, re.UNICODE)

        # general
        line = re.sub(r"\'ve", " have", line, re.UNICODE)
        line = re.sub(r"n\'t", " not", line, re.UNICODE)
        line = re.sub(r"\'re", " are", line, re.UNICODE)
#         line = re.sub(r"\'s", " is", line, re.UNICODE) # This one is
        line = re.sub(r"\'d", " would", line, re.UNICODE)
        line = re.sub(r"\'ll", " will", line, re.UNICODE)
        line = re.sub(r"\'t", " not", line, re.UNICODE)
        line = re.sub(r"\'m", " am", line, re.UNICODE)
        cleaned_lines.append(line)
    return cleaned_lines

def clean_non_ascii(lines):

    contraction_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    cleaned_lines = []
    for line in lines:
        cleaned_line = line.encode('ascii',errors='ignore')
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

import csv

def normalization_dictionaries(lines, dictionaries):

    token_lines = []
    for line in lines:
        token_lines.append(re.split("([\w']+)", line))

    for dictionary in dictionaries:
        dic = {}
        with open(dictionary, 'r') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                dic[row[0].strip()] = row[1].strip()

        for tokens in token_lines:
            for i, token in enumerate(tokens):
                if token in dic:
                    tokens[i] = dic[token]
    #                 print token


    cleaned_lines = []
    for tokens in token_lines:
        cleaned_lines.append(''.join(tokens))

    return cleaned_lines
