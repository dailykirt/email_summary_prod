import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

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
    def __init__(self, db):
        # Load model data
        self.db = db
        #self.table = 'cleaned_sj'
        self.table = 'full_enron_emails'
        self.final_summary = ''
        self.html_summary = []
        self.original_emails = []

    def list_employees(self):
        query = 'SELECT DISTINCT "employee" FROM ' + self.table
        self.employees = pd.read_sql_query(query, self.db.engine, params=(self.table)).employee.unique()
        return self.employees

    def get_timeframe(self, inbox):
        inbox = '\'' + inbox + '\''
        query = 'SELECT min(date) FROM ' + self.table + ' WHERE employee = ' + inbox
        start_date = pd.read_sql_query(query, self.db.engine).iloc[0][0]
        query = 'SELECT max(date) FROM ' + self.table + ' WHERE employee = ' + inbox
        end_date = pd.read_sql_query(query, self.db.engine).iloc[0][0]
        return {'start': str(start_date), 'end': str(end_date)}

    def subset_emails(self, start_date, end_date, inbox):
        """Outputs a subset of the enron dataset masked by the person and a timeframe. """
        def restore_extractive(text):
            # Restore list of sentences, since postgres stores as strings.
            text = text.split('","')
            text[0] = text[0][2:]
            text[-1] = text[-1][:-2]
            return text

        def restore_sentence_vectors(text):
            sentence_vectors = []
            text = text.split('},{')
            text[0] = text[0][2:]
            text[-1] = text[-1][:-2]
            for vector in text:
                vector = vector.split(',')
                sentence_vectors.append(np.array(vector, dtype=np.float32))
            return sentence_vectors

        #TODO temporarily getting full inbox dataframe, but will need to pull with sql statement.
        query = 'SELECT * FROM ' + self.table + ' WHERE employee = ' + '\'' + inbox + '\''
        self.enron_df = pd.read_sql_query(query, self.db.engine)
        summarization_mask = (self.enron_df['date'] >= start_date) & (self.enron_df['date'] <= end_date) & (self.enron_df['employee'] == inbox)
        self.enron_masked_df = self.enron_df.loc[summarization_mask]
        self.enron_masked_df['extractive_sentences'] = self.enron_masked_df['extractive_sentences'].apply(restore_extractive)
        self.enron_masked_df['sentence_vectors'] = self.enron_masked_df['sentence_vectors'].apply(restore_sentence_vectors)

    def get_extractive_sentences(self):
        """Retrieve original sentences and index them. This will be used to generate the extracted summaries. """
        # flatten list as tuples containting (sentence, dataframe index) to be used to reassociate summary with original email.
        sentences = []
        sentences_list = self.enron_masked_df.extractive_sentences.tolist()
        for counter, sublist in enumerate(sentences_list):
            for item in sublist:
                sentences.append([counter, item])
        self.sentences = sentences

    def get_tokenized_sentences(self):
        """Pull out clean tokenized sentences. """
        self.clean_sentences = self.enron_masked_df.Tokenized_Body.tolist()
        # flatten list
        self.clean_sentences = [y for x in self.clean_sentences for y in x]

    def processCosineSim(self, index):
        # Used to calculate sentence similarity
        sen_i = self.reshape_sentence_vectors[index[0]]
        sen_j = self.reshape_sentence_vectors[index[1]]
        return cosine_similarity(sen_i, sen_j)[0, 0]

    def rank_sentences(self):
        """Returns a list of sorted scores with the index of the email the extracted sentence came from. """
        # Parrallelize function due to slow O(n^2) runtime where n is number of sentence vectors.
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

    def get_sentence_vectors(self):
        """Get premade sentence vectors from dataframe"""
        """Pull out clean tokenized sentences. """
        self.sentence_vectors = self.enron_masked_df.sentence_vectors.tolist()
        # flatten list
        self.sentence_vectors = [np.asarray(y, dtype=np.float32) for x in self.sentence_vectors for y in x]

    def display_summary(self):
        # Specify number of sentences as a fraction of total emails.
        sn = (len(self.enron_masked_df) // 10) + 1
        self.html_summary = []
        self.final_summary = ''
        # Generate summary
        for i in range(sn):
            # pull date and subject from original email
            email_date = str(self.enron_masked_df['date'].iloc[self.ranked_sentences[i][1]])
            email_subject = str(self.enron_masked_df['subject'].iloc[self.ranked_sentences[i][1]])
            email_from = str(self.enron_masked_df['from'].iloc[self.ranked_sentences[i][1]])
            email_body = str(self.enron_masked_df['body'].iloc[self.ranked_sentences[i][1]])

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
        #print("Total number of emails to summarize: " + str(len(self.enron_masked_df)))
        self.get_extractive_sentences()
        self.get_sentence_vectors()
        # Create a list of ranked sentences.
        self.rank_sentences()
        self.display_summary()
