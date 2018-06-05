"""Following https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/"""

import numpy as np
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt

titles = [
    "The Neatest Little Guide to Stock Market Investing",
    "Investing For Dummies, 4th Edition",
    "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
    "The Little Book of Value Investing",
    "Value Investing: From Graham to Buffett and Beyond",
    "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
    "Investing in Real Estate, 5th Edition",
    "Stock Investing For Dummies",
    "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
]
stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
ignore_chars = ''',:'!'''


class LSA(object):
    def __init__(self, stopwords, ignore_chars):
        self.stopwords = stopwords
        self.ignore_chars = ignore_chars
        self.word_dict = {}
        self.doc_count = 0

    def parse(self, doc):
        """Takes a document, splits it into words, removes the ignored characters and turns everything into
        lowercase so the words can be compared to the stop words. If the word is a stop word, it is ignored
        and we move on to the next word. If it is not a stop word, we put the word in the dictionary, and also
        append the current document number to keep track of which documents the word appears in."""
        words = doc.split()
        for w in words:
            w = w.lower().translate({ord(c): None for c in self.ignore_chars})
            if w in self.stopwords:
                continue
            elif w in self.word_dict:
                self.word_dict[w].append(self.doc_count)
            else:
                self.word_dict[w] = [self.doc_count]
        self.doc_count += 1

    def build(self):
        """All the words (dictionary keys) that are in more than 1 document are extracted and sorted, and a
        matrix is built with the number of rows equal to the number of words (keys), and the number of
        columns equal to the document count. Finally, for each word (key) and document pair the corresponding
        matrix cell is incremented."""
        self.keys = [k for k in self.word_dict.keys() if len(self.word_dict[k]) > 1]
        self.keys.sort()
        self.A = np.zeros([len(self.keys), self.doc_count])
        for i, k in enumerate(self.keys):
            for d in self.word_dict[k]:
                self.A[i, d] += 1

    def printA(self):
        print(self.A, '\n')

    def calcSVD(self):
        self.U, self.s, self.Vt = sp_linalg.svd(self.A)

    def printSVD(self):
        print(self.U, '\n\n', self.s, '\n\n', self.Vt, '\n\n')

    def plot_s(self):
        plt.bar(range(len(self.s)), self.s**2)
        plt.show()


mylsa = LSA(stopwords, ignore_chars)
for t in titles:
    mylsa.parse(t)
mylsa.build()
mylsa.printA()
mylsa.calcSVD()
np.set_printoptions(precision=2)
mylsa.printSVD()
mylsa.plot_s()
