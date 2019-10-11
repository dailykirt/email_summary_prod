import pandas as pd
import numpy as np

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
    def __init__(self, db, table):
        # Load model data
        self.db = db
        self.table = table
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

        def restore_text_rank(text):
            sentence_ranks = []
            text = text.split(',')
            text[0] = text[0][2:]
            text[-1] = text[-1][:-2]
            for rank in text:
                sentence_ranks.append(np.array(rank, dtype=np.float32))
            return sentence_ranks

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
        self.enron_masked_df.loc[:, 'extractive_sentences'] = self.enron_masked_df['extractive_sentences'].apply(restore_extractive)
        self.enron_masked_df.loc[:, 'sentence_vectors'] = self.enron_masked_df['sentence_vectors'].apply(restore_sentence_vectors)
        self.enron_masked_df.loc[:, 'TextRanks'] = self.enron_masked_df['TextRanks'].apply(restore_text_rank)

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

    def create_summary_df(self):
        #This takes the extractive sentences, and pairs them with the TextRank to prepare display
        TextRanks = self.enron_masked_df.TextRanks.tolist()
        TextRanks = [y for x in TextRanks for y in x]

        output_df = pd.DataFrame(list(zip(self.sentences, TextRanks)), columns=['Sentences', 'TextRanks'])
        output_df['TextRanks'] = output_df.TextRanks.astype(float)
        # Specify number of sentences as a fraction of total emails.
        sn = (len(self.enron_masked_df) // 10) + 1
        self.display_top_df = output_df.nlargest(sn, 'TextRanks')

        print(self.display_top_df)

    def display_summary(self):
        # reclear out dispaly summaries
        self.final_summary = ''
        self.html_summary = []
        self.original_emails = []

        for index, row in self.display_top_df.iterrows():
            # pull date and subject from original email
            i = row['Sentences'][0]
            email_date = str(self.enron_masked_df['date'].iloc[i])
            email_subject = str(self.enron_masked_df['subject'].iloc[i])
            email_from = str(self.enron_masked_df['from'].iloc[i])
            email_body = str(self.enron_masked_df['body'].iloc[i])

            self.final_summary += bcolors.BOLD + "Date: " + email_date + \
                             " Subject: " + email_subject + \
                             " From: " + email_from + bcolors.ENDC + \
                             "\nSummary: " + str(row['Sentences'][1])

            #Used to create HTML text that will be displayed.
            self.html_summary.append("<br/>" + \
                                     "Date: " + email_date + \
                                     " Subject: " + email_subject + \
                                     " From: " + email_from + "<br/>" + \
                                     "\nSummary: " + str(row['Sentences'][1]) + "<br/>"
                                     )

            self.original_emails.append(email_body)
        #print(final_summary)


    def retrieve_summaries(self, start, end, inbox):
        self.subset_emails(start, end, inbox)
        self.get_extractive_sentences()
        self.create_summary_df()
        self.display_summary()
        #print(self.enron_masked_df)