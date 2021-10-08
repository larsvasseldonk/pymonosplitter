

from src import codeFragment
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter

import pandas as pd
import numpy as np
import keyword
import re
import os
import math
import nltk
import pickle
from pandas.core.frame import DataFrame
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from src.settings import s

from sklearn.decomposition import LatentDirichletAllocation

wnl = WordNetLemmatizer()

class semanticManager:
    def __init__(self, terms):
        self.vocab = terms # self.make_vocab(code)  # ['word1', 'word2', 'word3']
        self.normalise = self.normalise_vocab()
        self.lemmatise = self.lemmatise_vocab()
        self.tfidf_dict = None # {'word1' : 0.12, 'word2', 0.87}

    def normalise_vocab(self):
        '''Normalise the vocab by performing the following actions:          
            -set all characters to lowercase (as a confirmation)
            -strip white space (as a confirmation)
            -remove default stop words (general English stop words) 
        '''
        extra_items = ['self', 'method', 'init', 'staticmethod', 'id', 'function', 'print', 'false', 'true']
        exclude_items = set(keyword.kwlist + nltk.corpus.stopwords.words('english') + extra_items)
        self.vocab = [term.lower() for term in self.vocab if term not in exclude_items]

    def lemmatise_vocab(self):
        # Find the POS tag for each word
        pos_tags = nltk.pos_tag(self.vocab)
        for i, word in enumerate(pos_tags):
            # Word tag
            word_tag = self.pos_tagger(word[1])
            if word_tag is None:
                continue
            else:
                # Sanity check
                if self.vocab[i] == word[0]:
                    lem = wnl.lemmatize(word[0], word_tag)
                    if len(lem) > 1:
                        self.vocab[i] = lem
                    # print("{0:20}{1:20}".format(word[0], wnl.lemmatize(word[0], pos=word_tag)))

    @staticmethod
    def pos_tagger(nltk_tag):
        '''Tag terms. Only take verbs and nouns into consideration.'''
        if nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        elif nltk_tag.startswith('J'):
            return wordnet.ADJ
        else:
            return None

    @staticmethod
    def make_vocab(code):
        '''
        Input: list of code
        output: list of terms
        '''
        # Make set so all words are unique
        exclude_words = set(
            keyword.kwlist + nltk.corpus.stopwords.words('english'))
        vocab = []
        for line in code:
            # Remove all special characters and number
            # Only keep text
            line = re.sub(r'[^A-Za-z]+', r' ', line.strip())
            # If line is not null
            if line:
                words = line.split()
                vocab = vocab + [word.lower() for word in words if word not in exclude_words and len(word) > 1]
        return vocab

    @staticmethod
    def allocate_semantics_to_code_fragments(program):
        '''Give each code fragment and module a semanticManager.'''
        for package in program.packages:
            for module in package.modules:
                for code_fragment in module.code_fragments:
                    code_fragment.add_semanticManager()

    @staticmethod
    def get_semantic_edges(program):
        '''A semantic dependency is there when there is at least one word appearing in both sets.'''
        list_of_code_fragments = program.get_all_code_fragments()
        covered_classes = set()
        covered_modules = set()
        for code_fragment_A in list_of_code_fragments:
            encapsulator_code_fragment_A = code_fragment_A.get_cluster_code_fragment()
            if not encapsulator_code_fragment_A is code_fragment_A:
                continue
            for code_fragment_B in list_of_code_fragments:
                encapsulator_code_fragment_B = code_fragment_B.get_cluster_code_fragment()                
                if not encapsulator_code_fragment_B is code_fragment_B:
                    continue
                # Sanity check
                if code_fragment_A == code_fragment_B:
                    continue
                # Only outgoing edges
                # If the two are both defined in different fragments"
                if encapsulator_code_fragment_A is encapsulator_code_fragment_B:
                    continue
                terms_A = [key for key, value in code_fragment_A.semanticManager.tfidf_dict.items() if value > s.FREQUENCY_THRESHOLD]
                terms_B = [key for key, value in code_fragment_B.semanticManager.tfidf_dict.items() if value > s.FREQUENCY_THRESHOLD]
                overlapping_terms = set(terms_A) & set(terms_B)
                if len(overlapping_terms) > 0:
                    weight = 0
                    for overlapping_term in overlapping_terms:
                        weight += code_fragment_A.semanticManager.tfidf_dict.get(overlapping_term)
                    # Calculating class and module coverage
                    for cf in [encapsulator_code_fragment_A, encapsulator_code_fragment_B]:
                        if cf.type == 'class':
                            covered_classes.add(cf)
                        if cf.type == 'module':
                            covered_modules.add(cf)
                    edge = [encapsulator_code_fragment_A.id, encapsulator_code_fragment_B.id, weight, 1.0, list(overlapping_terms)]
                    # if not encapsulator_code_fragment_A is code_fragment_A:
                    #     edges_without_weight = [i[:2] for i in code_fragment_A.semantic_edges]
                    #     if edge[:2] in edges_without_weight:
                    #         for i, cf_edge in enumerate(code_fragment_A.semantic_edges):
                    #             if edge[:2] == cf_edge[:2]:
                    #                 code_fragment_A.semantic_edges[i][2] += weight
                    #                 code_fragment_A.semantic_edges[i][3] += 1
                    #     else:
                    #         code_fragment_A.semantic_edges.append(edge)
                    # Check if the edge already exists for the encapsulator
                    # IMPORTANT BECAUSE THIS ARE THE ONES THAT WE WILL CLUSTER
                    if edge[:2] in [i[:2] for i in encapsulator_code_fragment_A.semantic_edges]:
                        for i, cf_edge in enumerate(encapsulator_code_fragment_A.semantic_edges):
                            if edge[:2] == cf_edge[:2]:
                                encapsulator_code_fragment_A.semantic_edges[i][2] += weight
                                encapsulator_code_fragment_A.semantic_edges[i][3] += 1
                    else:
                        encapsulator_code_fragment_A.semantic_edges.append(edge)
        all_cfs = program.get_all_code_fragments_to_be_clustered()
        module_cfs = [cf for cf in all_cfs if cf.type == 'module']
        class_cfs = [cf for cf in all_cfs if cf.type == 'class']
        program.semantic_coverage['modules_covered'] = len(covered_modules)
        program.semantic_coverage['module_coverage'] = round(len(covered_modules)/len(module_cfs), 3)
        program.semantic_coverage['classes_covered'] = len(covered_classes)
        program.semantic_coverage['class_coverage'] = round(len(covered_classes)/len(class_cfs),3)

    @staticmethod
    def normalise_semantic_weights(program):
        print('Normalising weights...')
        list_of_code_fragments = program.get_all_code_fragments()
        weights = [edge[2] for cf in list_of_code_fragments for edge in cf.semantic_edges]
        max_weight = max(weights)
        for code_fragment in list_of_code_fragments:
            # TODO: For some reason, some code fragments get updated twice.
            # Figure out why this happens and how to solve it.
            for edge in code_fragment.semantic_edges:
                updated_weight = edge[2]/max_weight
                edge.append(round(updated_weight, 2))
        print('Max weight', max_weight)

    @staticmethod
    def make_bow(code_fragments):
        '''Make a bag of words from a set of vocabs'''
        headers = list()
        bow = list()  # Bag of words
        vocab = list()
        cf_vocabs = list()
        for cf in code_fragments:
            # the vocab of the methods are already in the classes
            if cf.type == 'class' and cf.is_first: 
                cf_vocabs.append(cf.semanticManager.vocab)
            elif cf.type in ['module', 'function']:
                cf_vocabs.append(cf.semanticManager.vocab)
            vocab = vocab + cf.semanticManager.vocab
        vocab = set(vocab)
        idf_dict = semanticManager.compute_idf(vocab, cf_vocabs)
        for cf in code_fragments:
            headers.append(cf.name)
            tfidf = cf.semanticManager.compute_tfidf(idf_dict)
            bow.append(tfidf)
        vocab = list(vocab) # To make sure the order remains the same
        df = pd.DataFrame(bow, columns=vocab, index=headers)
        df.to_excel(os.path.join(s.DATA_PATH, s.program_name + '_bow.xlsx'))
        df = df.to_numpy()
        topic_modelling_lda(df, vocab, headers)
        return bow

    def compute_tf(self):
        '''
        Compute the ratio of number of times the word appears in 
        a document compared to the total number of words in that document.
        '''
        # tf_dict = dict.fromkeys(vocab, 0)
        tf_dict = dict()
        vocab_counter = Counter(self.vocab)
        for word in vocab_counter:
            tf_dict[word] = vocab_counter[word]/float(len(self.vocab))
        return tf_dict

    @staticmethod
    def compute_idf(vocab, cf_vocabs):
        '''
        Compute the weight of rare words across all documents in the corpus. 
        The words that occur rarely in the corpus have a high IDF score.

        input:
            N = number of documents (code fragments in our case)
        '''
        idf_dict = dict.fromkeys(vocab, 0)
        for word in vocab:
            for cf_vocab in cf_vocabs:
                if word in cf_vocab:
                    idf_dict[word] += 1
                     
        for word, val in idf_dict.items():
            if val:
                idf_dict[word] = math.log10(len(cf_vocabs)/float(val))

        return idf_dict

    def compute_tfidf(self, idf_dict):
        # https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
        tfidf_dict = dict()
        for word, val in self.compute_tf().items():
            tfidf_dict[word] = val*idf_dict[word]
        self.tfidf_dict = tfidf_dict
        return tfidf_dict

def topic_modelling_lda(bow, vocab, headers):
    # Change nan's to 0
    bow_no_nas = np.where(np.isnan(bow), 0, bow)
    lda = LatentDirichletAllocation(n_components=5)
    lda.fit(bow_no_nas)
    for idx, topic in enumerate(lda.components_):
        # Get indices of highest lda components
        print ("Topic ", idx, " ".join(vocab[i] for i in topic.argsort()[:-10 - 1:-1]))
    topic_distr = lda.transform(bow_no_nas) # each row represents a code fragment
    df = DataFrame(topic_distr, columns=['topic1', 'topic2', 'topic3', 'topic4', 'topic5'], index=headers)
    df.to_excel(os.path.join(s.DATA_PATH, s.program_name + '_topic_distribution.xlsx'))
    # for i, each in enumerate(topic_distr):
    #     print(headers[i], each)