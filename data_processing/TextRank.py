"This module takes in a email dataframe, then goes through each inbox to rank each sentence to generate full extractive summaries. "
import networkx as nx
import numpy as np
import data_config
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

from sqlalchemy import create_engine


class TextRank:

    def __init__(self, df):
        self.email_df = df
        # Extract word embeddings.
        self.extract_word_vectors()

    def extract_word_vectors(self):
        # get glove word vectors
        self.word_embeddings = {}
        f = open(data_config.wb_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_embeddings[word] = coefs
        f.close()

    def create_sentence_vectors(self, clean_sentences):
        # Create sentence_vectors
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

    def get_extractive_sentences(self):
        """Retrieve original sentences and index them. This will be used to generate the extracted summaries. """
        # flatten list as tuples containting (sentence, dataframe index) to be used to reassociate summary with original email.
        sentences = []
        sentences_list = self.email_masked_df.extractive_sentences.tolist()
        for counter, sublist in enumerate(sentences_list):
            for item in sublist:
                sentences.append([counter, item])
        self.sentences = sentences

    def subset_emails(self, employee):
        summarization_mask = (self.email_df['employee'] == employee)
        self.email_masked_df = self.email_df.loc[summarization_mask]

    def get_sentence_vectors(self):
        """Get premade sentence vectors from dataframe"""
        """Pull out clean tokenized sentences. """
        self.sentence_vectors = self.email_masked_df.sentence_vectors.tolist()
        # flatten list
        self.sentence_vectors = [np.asarray(y, dtype=np.float32) for x in self.sentence_vectors for y in x]

    def processCosineSim(self, index):
        # Used to calculate sentence similarity
        sen_i = self.reshape_sentence_vectors[index[0]]
        sen_j = self.reshape_sentence_vectors[index[1]]
        return cosine_similarity(sen_i, sen_j)[0, 0]

    def rank_sentences(self):
        """Returns a list of sorted scores with the index of the email the extracted sentence came from. """
        num_sen = len(self.sentences)
        self.reshape_sentence_vectors = []
        for i in range(num_sen):
            self.reshape_sentence_vectors.append(self.sentence_vectors[i].reshape(1, 300))
        indexes = []

        #Create unrolled indexes
        for i in range(num_sen):
            for j in range(num_sen):
                if (i != j) and (i < j):  # Don't compare sentence to itself, or repeat comparisons.
                    indexes.append([i, j])

        #List comprehension attempt
        result = [self.processCosineSim(index) for index in indexes]
        # put result into similarity matrix
        sim_mat = np.zeros([num_sen, num_sen])

        for count, index in enumerate(indexes):
            sim_mat[index[0]][index[1]] = result[count]

        # now generate scores and rank sentences
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        # Pair sentence with it's similarity score then sort. (score, email_index, sentence)
        self.ranked_sentences = list(((scores[i], s[0], s[1]) for i, s in enumerate(self.sentences)))

    def append_rank_df(self):
        #This will append the TextRank of every sentence to the appropriate location in the email dataframe.
        self.email_masked_df['TextRanks'] = np.empty((len(self.email_masked_df), 0)).tolist()
        for rank in self.ranked_sentences:
            self.email_masked_df.TextRanks.iloc[rank[1]] += [rank[0]]

    def insert_db(self):
        cnx = create_engine(data_config.postgres_str)
        self.email_masked_df.to_sql('test_rank_db', cnx, if_exists='append')

    def summarize_emails(self):
        # Creating sentence vectors for each cleaned sentence.
        self.email_df['sentence_vectors'] = self.email_df['tokenized_body'].apply(self.create_sentence_vectors)
        #rank sentences in each inbox
        for employee in self.email_df['employee'].unique():
            self.subset_emails(employee)
            self.get_extractive_sentences()
            self.get_sentence_vectors()
            self.rank_sentences()
            self.append_rank_df()
            self.insert_db()

        print(self.email_masked_df)
        #print(self.email_masked_df.info())

