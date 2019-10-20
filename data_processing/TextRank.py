"This module takes in a email dataframe, then goes through each inbox to rank each sentence to generate full extractive summaries. "
import networkx as nx
import numpy as np
import data_config
from sklearn.metrics.pairwise import cosine_similarity
import gc
import pickle

from sqlalchemy import create_engine

class TextRank:

    def __init__(self, df):
        #Set the full email dataframe and create word vectors.
        self.email_df = df
        print("Extracting word vectors")
        #self.extract_word_embeddings()
        self.load_word_embeddings()
        # Creating sentence vectors for each cleaned sentence.
        self.email_df['sentence_vectors'] = self.email_df['tokenized_body'].apply(self.create_sentence_vectors)

    def extract_word_embeddings(self):
        # get glove word vectors
        self.word_embeddings = {}
        f = open(data_config.wb_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_embeddings[word] = coefs
        f.close()
        #Save to pickle for easy loading
        #self.word_embeddings.to_pickle('../data/word_embeddings.pkl')
        with open('../data/word_embeddings.pkl', 'wb') as handle:
            pickle.dump(self.word_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_word_embeddings(self):
        with open('../data/word_embeddings.pkl', 'rb') as handle:
            self.word_embeddings = pickle.load(handle)

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

    def get_extractive_sentences(self, email_masked_df):
        """Retrieve original sentences and index them. This will be used to generate the extracted summaries. """
        # flatten list as tuples containting (sentence, dataframe index) to be used to reassociate summary with original email.
        sentences = []
        #sentences = np.array([[0, 'initialize']])
        sentences_list = email_masked_df.extractive_sentences.tolist()
        #flatten sentences and keep email number.
        for counter, sublist in enumerate(sentences_list):
            for item in sublist:
                sentences.append([counter, item])
        #try numpy array instead
        return sentences


    def subset_emails(self, employee):
        summarization_mask = (self.email_df['employee'] == employee)
        #self.email_masked_df = self.email_df.loc[summarization_mask]
        return self.email_df.loc[summarization_mask]

    def get_sentence_vectors(self, email_masked_df):
        """Get premade sentence vectors from dataframe"""
        """Pull out clean tokenized sentences. """
        sentence_vectors = email_masked_df.sentence_vectors.tolist()
        # flatten list
        sentence_vectors = [np.asarray(y, dtype=np.float32) for x in sentence_vectors for y in x]
        return sentence_vectors

    #def processCosineSim(self, index):
        # Used to calculate sentence similarity
    #    sen_i = self.sentence_vectors[index[0]].reshape(1, 300)
    #    sen_j = self.sentence_vectors[index[1]].reshape(1, 300)
    #    return cosine_similarity(sen_i, sen_j)[0, 0]

    #def rank_sentences(self, sentensubset_emailsces, sentence_vectors):
    def rank_sentences(self, sentences, sim_result, indexes):
        """Returns a list of sorted scores with the index of the email the extracted sentence came from. """
        #manually garbage collect
        gc.collect()
        num_sen = len(sentences)
        print("Number of sentences: " + str(num_sen))
        sim_mat = np.zeros([num_sen, num_sen])

        # for i in range(num_sen):
        #     for j in range(num_sen):
        #         if (i != j) and (i < j):  # Don't compare sentence to itself, or repeat comparisons.
        #             sen_i = sentence_vectors[i].reshape(1, 300)
        #             sen_j = sentence_vectors[j].reshape(1, 300)
        #             sim_mat[i][j] = cosine_similarity(sen_i, sen_j)[0, 0]
        for count, index in enumerate(indexes):
            i = index[0]
            j = index[1]
            if (i != j) and (i < j):
                sim_mat[i][j] = sim_result[count]
            elif (j < i):
                sim_mat[i][j] = sim_result[count]

        scores = nx.pagerank(nx.from_numpy_array(sim_mat))
        # Pair sentence with it's similarity score then sort. (score, email_index, sentence)
        return list(((scores[i], s[0], s[1]) for i, s in enumerate(sentences)))

    def unroll_rank_indexes(self, sen_len):
        #This returns a list of indexes that need to be calculated for the similarity matrix.
        indexes = []
        for i in range(sen_len):
            for j in range(sen_len):
                if (i != j) and (i < j):
                    indexes.append([i, j])
        return indexes

    def append_rank_df(self, ranked_sentences, email_masked_df):
        #This will append the TextRank of every sentence to the appropriate location in the email dataframe.
        email_masked_df['TextRanks'] = np.empty((len(email_masked_df), 0)).tolist()
        for rank in ranked_sentences:
            email_masked_df.TextRanks.iloc[rank[1]] += [rank[0]]
        return email_masked_df

    def insert_db(self, email_masked_df):
        cnx = create_engine(data_config.postgres_str)
        email_masked_df.to_sql(data_config.table, cnx, if_exists='append')



    def summarize_emails(self):
        #rank sentences in each inbox
        for employee in self.email_df['employee'].unique():
            email_masked_df = self.subset_emails(employee)
            sentences = self.get_extractive_sentences(email_masked_df)
            sentence_vectors = self.get_sentence_vectors(email_masked_df)
            #ranked_sentences = self.rank_sentences(sentences, sentence_vectors)
            #email_masked_df = self.append_rank_df(ranked_sentences, email_masked_df)
            #self.insert_db(email_masked_df)

        #print(self.email_masked_df)
        #print(self.email_masked_df.info())

