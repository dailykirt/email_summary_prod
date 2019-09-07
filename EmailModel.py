import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle

# color scheme to help distinguish summarizaiton text.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class EmailModel:
    def __init__(self):
        # Load model data
        self.ENRON_PICKLE_LOC = "data/dataframes/wrangled_enron_full_df.pkl"
        self.WORD_EMBEDDINGS_LOC = "data/word_embeddings.pkl"
        self.enron_df = pd.read_pickle(self.ENRON_PICKLE_LOC)
        self.word_embeddings = pickle.load(open(self.WORD_EMBEDDINGS_LOC, "rb"))
        self.final_summary = ''
        self.html_summary = []
        self.original_emails = []

    def list_employees(self):
        self.employees = self.enron_df.Employee.unique()
        return self.employees

    def get_timeframe(self, inbox):
        start_date = self.enron_df[self.enron_df.Employee == inbox].Date.min()
        end_date = self.enron_df[self.enron_df.Employee == inbox].Date.max()
        return {'start': str(start_date), 'end': str(end_date)}

    def subset_emails(self, start_date, end_date, person):
        """Outputs a subset of the enron dataset masked by the person and a timeframe. """
        summarization_mask = (self.enron_df['Date'] >= start_date) & (self.enron_df['Date'] <= end_date) & (self.enron_df['Employee'] == person)
        self.enron_masked_df = self.enron_df.loc[summarization_mask]

    def get_extractive_sentences(self):
        """Retrieve original sentences and index them. This will be used to generate the extracted summaries. """
        # flatten list as tuples containting (sentence, dataframe index) to be used to reassociate summary with original email.
        sentences = []
        sentences_list = self.enron_masked_df.Extractive_Sentences.tolist()
        for counter, sublist in enumerate(sentences_list):
            for item in sublist:
                sentences.append([counter, item])
        self.sentences = sentences

    def get_tokenized_sentences(self):
        """Pull out clean tokenized sentences. """
        self.clean_sentences = self.enron_masked_df.Tokenized_Body.tolist()
        # flatten list
        self.clean_sentences = [y for x in self.clean_sentences for y in x]
        self.clean_sentences = self.clean_sentences

    def processCosineSim(self, index):
        # Used to calculate sentence similarity
        sen_i = self.reshape_sentence_vectors[index[0]]
        sen_j = self.reshape_sentence_vectors[index[1]]
        return cosine_similarity(sen_i, sen_j)[0, 0]

    def rank_sentences(self):
        """Returns a list of sorted scores with the index of the email the extracted sentence came from. """
        # Parrallelize function due to slow O(n^2) runtime where n is number of sentence vectors.
        #pool = multiprocessing.Pool(processes=6)
        num_sen = len(self.sentences)
        self.reshape_sentence_vectors = []
        for i in range(num_sen):
            self.reshape_sentence_vectors.append(self.sentence_vectors[i].reshape(1, 300))
        # unroll similarity loop.
        indexes = []
        for i in range(num_sen):
            for j in range(num_sen):
                if (i != j) and (i < j):  # Don't compare sentence to itself, or repeat comparisons.
                    indexes.append([i, j])
        #Now calculate similarities
        result = []
        for index in indexes:
            result.append(self.processCosineSim((index)))
        #result = pool.map(self.processCosineSim, indexes)
        # put result into similarity matrix
        sim_mat = np.zeros([num_sen, num_sen])
        for count, index in enumerate(indexes):
            i = index[0]
            j = index[1]
            if (i != j) and (i < j):
                sim_mat[i][j] = result[count]

        # now generate scores and rank sentences
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        # Pair sentence with it's similarity score then sort.
        self.ranked_sentences = sorted(((scores[i], s[0], s[1]) for i, s in enumerate(self.sentences)), reverse=True)

    def create_sentence_vectors(self):
        """Create sentence_vectors"""
        sentence_vectors = []
        for i in self.clean_sentences:
            if len(i) != 0:
                v = sum([self.word_embeddings.get(w, np.zeros((300,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((300,))
            sentence_vectors.append(v)
        self.sentence_vectors = sentence_vectors

    def display_summary(self):
        # Specify number of sentences as a fraction of total emails.
        sn = (len(self.enron_masked_df) // 10) + 1
        self.html_summary = []
        self.final_summary = ''
        # Generate summary
        for i in range(sn):
            # pull date and subject from original email
            email_date = str(self.enron_masked_df['Date'].iloc[self.ranked_sentences[i][1]])
            email_subject = str(self.enron_masked_df['Subject'].iloc[self.ranked_sentences[i][1]])
            email_from = str(self.enron_masked_df['From'].iloc[self.ranked_sentences[i][1]])
            email_body = str(self.enron_masked_df['Body'].iloc[self.ranked_sentences[i][1]])

            self.final_summary += bcolors.BOLD + "Date: " + email_date + \
                  " Subject: " + email_subject + \
                  " From: " + email_from + bcolors.ENDC + \
                  "\nSummary: " + str(self.ranked_sentences[i][2])

            self.html_summary.append("<br/>" +\
                "Date: " + email_date + \
                " Subject: " + email_subject + \
                " From: " + email_from + "<br/>" + \
                "\nSummary: " + str(self.ranked_sentences[i][2]) + "<br/>"
            )

            self.original_emails.append(email_body)
        print(self.final_summary)

    def summarize_emails(self, start, end, inbox):
        self.subset_emails(start, end, inbox)
        print("Total number of emails to summarize: " + str(len(self.enron_masked_df)))
        self.get_extractive_sentences()
        self.get_tokenized_sentences()
        # Generate sentence vectors
        self.create_sentence_vectors()
        # Create a list of ranked sentences.
        self.rank_sentences()
        # return enron_masked_df, ranked_sentences
        self.display_summary()
