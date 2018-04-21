import glob
from collections import Counter
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer

categories = ['pos', 'neg']

def extract_tokens(files):
    words = list()
    for file in files:
        with open(file) as f:
            for line in f:
                for word in line.split():
                    if len(word) > 2:
                        words.append(word)
    return words

docs = {}
vocabulary = []
N = 0
for category in categories:
    cat_files = glob.glob('data/imdb1/' + category + '/*.txt')
    cat_words = extract_tokens(cat_files)
    docs[category] = cat_files, cat_words
    vocabulary += cat_words
    N += len(cat_files)

V = len(set(vocabulary))
prior_probs = {}
for category in categories:
    cat_files, cat_words  = docs[category]
    prior_probs[category] = len(cat_files)/N


V = len(vocabulary)

print("Vocabulary Length : %i" %V)
print("Probability of pos : %0.2f" % prior_probs['pos'])
print("Probability of neg : %0.2f" % prior_probs['neg'])


total_counts = Counter(vocabulary)
cat_counts = {}
for category in categories:
    cat_files, cat_words  = docs[category]
    cat_counts[category] = Counter(cat_words)


conditional_probs = {}

for word in vocabulary:
    probs = {}
    for cat in categories:
        probs[cat] = (cat_counts[cat][word] + 1)/(total_counts[word] + V)
    conditional_probs[word] = probs



cat_files, cat_words  = docs['neg']

tokenizer = RegexpTokenizer(r'\w+')
with open(cat_files[160], 'r') as f:
    test_tokens = tokenizer.tokenize(f.read())

filtered_tokens = [t for t in test_tokens if len(t) > 2]
filtered_tokens = ['sad']
print(filtered_tokens)

scores = {}

for cat in categories:
    scores[cat] = np.log(prior_probs[cat])
    for word in filtered_tokens:
        scores[cat] += np.log(conditional_probs[word][cat])

print(scores)
maximum = max(scores, key=scores.get)  # Just use 'min' instead of 'max' for minimum.
print(maximum)




