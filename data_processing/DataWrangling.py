import email
import re
import sys
from os import listdir
import data_config
import mailparser
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from talon.signature.bruteforce import extract_signature
import logging

LOG_NAME = '__data_processing__'

class DataWrangling:
    def __init__(self):
        self.enron_email_list = []
        self.logger = logging.getLogger(LOG_NAME)

    def process_date(self, date_time):
        """Reformat date to be more pandas readable"""
        try:
            date_time = email.utils.format_datetime(email.utils.parsedate_to_datetime(date_time))
        except:
            date_time = None
        return date_time

    def clean_body(self, mail_body):
        """Contains several email cleaning procedures."""
        delimiters = ["-----Original Message-----", "To:", "From"]
        # Split body by earliest appearing delimiter, with delimiters being indicators of the start of an email being forwarded.
        old_len = sys.maxsize
        for delimiter in delimiters:
            split_body = mail_body.split(delimiter, 1)
            new_len = len(split_body[0])
            if new_len <= old_len:
                old_len = new_len
                final_split = split_body

        if (len(final_split) == 1):
            mail_chain = None
        else:
            mail_chain = final_split[1]
        # The following uses Talon library to try to extract a clean body from signatures of the remaining email body.
        clean_body, sig = extract_signature(final_split[0])
        return {'body': clean_body, 'chain': mail_chain, 'signature': sig}

    def process_email(self, email_path, employee, folder):
        """This parses emails into constituent parts using mailparser to create features to the overall email dataframe. """
        mail = mailparser.parse_from_file(email_path)
        full_body = email.message_from_string(mail.body)
        if full_body.is_multipart():
            return
        else:
            mail_body = full_body.get_payload()
        split_body = self.clean_body(mail_body)
        headers = mail.headers
        date_time = self.process_date(headers.get('Date'))
        email_dict = {
            "employee": employee,
            "email_folder": folder,
            "message_id": headers.get('Message-ID'),
            "date": date_time,
            "from": headers.get('From'),
            "subject": headers.get('Subject'),
            "body": split_body['body'],
            "chain": split_body['chain'],
            "signature": split_body['signature'],
            "full_email_path": email_path  # for debug purposes.
        }
        self.enron_email_list.append(email_dict)

    def remove_stopwords(self, sen):
        """Remove commonly used words that do not add value to the model."""
        stop_words = stopwords.words('english')
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    def tokenize_email(self, text):
        """Tokenizes email bodies, then remove stopwords."""
        clean_sentences = sent_tokenize(text, language='english')
        # removing punctuation, numbers and special characters. Then lowercasing.
        clean_sentences = [re.sub('[^a-zA-Z ]', '', s) for s in clean_sentences]
        clean_sentences = [s.lower() for s in clean_sentences]
        clean_sentences = [self.remove_stopwords(r.split()) for r in clean_sentences]
        return clean_sentences

    def clean_email_df(self):
        """Cleans loaded email bodies throughout dataframe"""
        # Removing strings related to attachments and certain non numerical characters.
        patterns = ["\[IMAGE\]", "-", "_", "\*", "+", "\".\"", "<", ">"]
        for pattern in patterns:
            self.enron_email_list_df['body'] = pd.Series(self.enron_email_list_df['body']).str.replace(pattern, "")
        # Remove multiple spaces.
        self.enron_email_list_df['body'] = self.enron_email_list_df['body'].replace('\s+', ' ', regex=True)
        # Blanks are replaced with NaN. Rows with a 'NaN' in the body will be dropped.
        self.enron_email_list_df = self.enron_email_list_df.replace('', np.NaN)
        self.enron_email_list_df = self.enron_email_list_df.dropna(subset=['body'])
        # Remove duplicate emails
        self.enron_email_list_df = self.enron_email_list_df.drop_duplicates(subset='body')

    def clean_date(self):
        """Set and clean parsed dates. """
        self.enron_email_list_df['date'] = pd.to_datetime(self.enron_email_list_df['date'], utc=True)
        # Keep emails between Jan 1st 1999 and Jan 1st 2003, which is the historical timeframe of the enron dataset.
        self.enron_email_list_df = self.enron_email_list_df[(self.enron_email_list_df.date > '1999-01-01') & (self.enron_email_list_df.date < '2003-01-01')]

    def get_inboxes(self):
        """Gets list of folders from a directory, assumed to contain inboxes."""
        self.mailboxes = listdir(data_config.mail_dir)
        self.logger.info("Mailboxes to be processed: " + str(self.mailboxes))

    def load_mail(self, mailbox):
        """Parses and loads each inbox's email into a dataframe."""
        self.logger.info("Processing mailbox: " + mailbox)
        inbox = listdir(data_config.mail_dir + mailbox)
        for folder in inbox:
            path = data_config.mail_dir + mailbox + "/" + folder
            emails = listdir(path)
            for single_email in emails:
                full_path = path + "/" + single_email
                self.process_email(full_path, mailbox, folder)
        self.enron_email_list_df = pd.DataFrame(self.enron_email_list)

    def wrangle_mailbox(self, mailbox):
        """Takes in a mailbox name to process every email inside the folder. """
        #reset email list for new mailbox.
        self.enron_email_list = []
        self.load_mail(mailbox)
        self.clean_date()
        self.clean_email_df()
        self.enron_email_list_df['extractive_sentences'] = self.enron_email_list_df['body'].apply(sent_tokenize)
        self.enron_email_list_df['tokenized_body'] = self.enron_email_list_df['body'].apply(self.tokenize_email)