import networkx as nx
import numpy as np
import data_config
import pickle
from sqlalchemy import create_engine
import logging
from os import path

LOG_NAME = '__data_processing__'
WORD_EMBEDDINGS_PKL = '../data/word_embeddings.pkl'

class TextRank:

    def __init__(self, df):
        self.logger = logging.getLogger(LOG_NAME)
        self.email_df = df
        if path.exists(WORD_EMBEDDINGS_PKL):
            self.logger.info("Loading word embeddings from pkl")
            self.load_word_embeddings()
        else:
            self.logger.info("Creating word embeddings from gLovE")
            self.extract_word_embeddings()
        self.email_df['sentence_vectors'] = self.email_df['tokenized_body'].apply(self.create_sentence_vectors)

    def extract_word_embeddings(self):
        """Creates dictionary of word embeddings."""
        self.word_embeddings = {}
        f = open(data_config.wb_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_embeddings[word] = coefs
        f.close()
        with open(WORD_EMBEDDINGS_PKL, 'wb') as handle:
            pickle.dump(self.word_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_word_embeddings(self):
        """If pkl of word embeddings already exist, then load them."""
        with open(WORD_EMBEDDINGS_PKL, 'rb') as handle:
            self.word_embeddings = pickle.load(handle)

    def create_sentence_vectors(self, clean_sentences):
        """Applies word embeddings to every tokenized email sentence. """
        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((300,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((300,))
            # store v as a list for postgres storage
            v = v.tolist()
            sentence_vectors.append(v)
        return sentence_vectors

    def get_extractive_sentences(self, email_masked_df):
        """Retrieve original sentences and index them. This will be used to generate the extractive summaries. """
        sentences = []
        sentences_list = email_masked_df.extractive_sentences.tolist()
        # flatten list as tuples containting (sentence, dataframe index) to be used to reassociate summary with original email.
        for counter, sublist in enumerate(sentences_list):
            for item in sublist:
                sentences.append((counter, item))
        return sentences

    def subset_emails(self, employee):
        summarization_mask = (self.email_df['employee'] == employee)
        return self.email_df.loc[summarization_mask]

    def get_sentence_vectors(self, email_masked_df):
        """Get premade sentence vectors from dataframe"""
        """Pull out clean tokenized sentences. """
        sentence_vectors = email_masked_df.sentence_vectors.tolist()
        # flatten list
        sentence_vectors = [np.asarray(y, dtype=np.float32) for x in sentence_vectors for y in x]
        return sentence_vectors

    def rank_sentences(self, sentences, sim_result, indexes):
        """Returns a list of sorted scores with the index of the email the extracted sentence came from. """
        num_sen = len(sentences)
        sim_mat = np.zeros([num_sen, num_sen])
        for count, index in enumerate(indexes):
            i = index[0]
            j = index[1]
            sim_mat[i][j] = next(sim_result)

        self.logger.debug("Start pagerank")
        scores = nx.pagerank(nx.from_numpy_array(sim_mat), max_iter = 10000)
        # Pair sentence with it's similarity score then sort. (score, email_index, sentence)
        return list(((scores[i], s[0], s[1]) for i, s in enumerate(sentences)))

    def generate_indexes(self, sen_len, window):
        """Produces indexes for each sentence vector comparison. Limiting total number of comparisons by a window to prevent memory errors."""
        for i in range(sen_len):
            for j in range(sen_len):
                diff = j - i
                if (i != j) and (i < j) and (diff < window):
                    yield (i, j)

    def append_rank_df(self, ranked_sentences, email_masked_df):
        """This will append the TextRank of every sentence to the appropriate location in the email dataframe."""
        email_masked_df['TextRanks'] = np.empty((len(email_masked_df), 0)).tolist()
        for rank in ranked_sentences:
            email_masked_df.TextRanks.iloc[rank[1]] += [rank[0]]
        return email_masked_df

    def insert_db(self, email_masked_df):
        """Insert final dataframe to postgres datagbase."""
        self.logger.info("Push to database: " + str(data_config.postgres_str))
        cnx = create_engine(data_config.postgres_str)
        email_masked_df.to_sql(data_config.table, cnx, if_exists='append')


