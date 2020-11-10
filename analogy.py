# Tosin Adewumi

import re
import time
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import Word2Vec, KeyedVectors, FastText, fasttext
#from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus
import logging
import os
from multiprocessing import cpu_count

OUTPUT_FILE1 = "L1_out_sv_subgig410.txt"


if __name__ == "__main__":
    # model = KeyedVectors.load_word2vec_format("cc.sv.300.bin", binary=True)
    model = fasttext.load_facebook_vectors("../fastText-0.9.1/sv_1subgiga_410.bin")

    # get vocab size and save
    print("Vocab", len(model.wv.vocab))
    # print("Time elapsed", time_elapsed)
    # evaluate on word analogies
    human_word_sim = model.wv.evaluate_word_pairs("wordsim_yo_dia.tsv")
    analogy_scores = model.wv.evaluate_word_analogies("questions-words-yoruba-dia.txt")
    with open(OUTPUT_FILE1, "w+") as f:
        s = f.write("Vocab: " + str(len(model.wv.vocab)) + "\n" "Analogy Score: " + str(analogy_scores)) + "\n" + "Human Word Similarity:" + str(human_word_sim))
