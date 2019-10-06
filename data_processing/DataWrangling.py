# Purpose of this file is to wrangle the full enron_email dataset into a pandas dataframe to prepare for the text summarization process.
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


class DataWrangling:
    def __init__(self):
        self.enron_email_list = []

    def process_date(self, date_time):
        try:
            date_time = email.utils.format_datetime(email.utils.parsedate_to_datetime(date_time))
        except:
            date_time = None
        return date_time

    def clean_body(self, mail_body):
        delimiters = ["-----Original Message-----", "To:", "From"]
        # Trying to split string by biggest delimiter.
        old_len = sys.maxsize

        for delimiter in delimiters:
            split_body = mail_body.split(delimiter, 1)
            new_len = len(split_body[0])
            if new_len <= old_len:
                old_len = new_len
                final_split = split_body

        # Then pull chain message
        if (len(final_split) == 1):
            mail_chain = None
        else:
            mail_chain = final_split[1]

        # The following uses Talon to try to get a clean body, and seperate out the rest of the email.
        clean_body, sig = extract_signature(final_split[0])

        return {'body': clean_body, 'chain': mail_chain, 'signature': sig}

    def process_email(self, email_path, employee, folder):
        "This parses emails into consituent parts, to add as features to the overall email dataframe. "
        mail = mailparser.parse_from_file(email_path)
        full_body = email.message_from_string(mail.body)

        # Only getting first payload
        if full_body.is_multipart():
            return
        else:
            mail_body = full_body.get_payload()

        split_body = self.clean_body(mail_body)
        headers = mail.headers
        # Reformating date to be more pandas readable
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

        # Append row to dataframe.
        self.enron_email_list.append(email_dict)

    def load_enron_mail(self):
        # Go through each person's inbox then load up each email to be cleaned and added to the dataframe.
        # mailboxes = listdir(mail_dir)
        # for mailbox in mailboxes:
        mailbox = data_config.mailbox
        inbox = listdir(data_config.mail_dir + mailbox)
        for folder in inbox:
            path = data_config.mail_dir + data_config.mailbox + "/" + folder
            emails = listdir(path)
            for single_email in emails:
                full_path = path + "/" + single_email
                self.process_email(full_path, mailbox, folder)
        self.enron_email_list_df = pd.DataFrame(self.enron_email_list)

    # Email body cleaning at dataframe level
    def clean_email_df(self):
        # Removing strings related to attatchments and certain non numerical characters.
        patterns = ["\[IMAGE\]", "-", "_", "\*", "+", "\".\""]
        for pattern in patterns:
            self.enron_email_list_df['body'] = pd.Series(self.enron_email_list_df['body']).str.replace(pattern, "")

        # Remove multiple spaces.
        self.enron_email_list_df['body'] = self.enron_email_list_df['body'].replace('\s+', ' ', regex=True)

        # Blanks are replaced with NaN in the whole dataframe. Then rows with a 'NaN' in the body will be dropped.
        self.enron_email_list_df = self.enron_email_list_df.replace('', np.NaN)
        self.enron_email_list_df = self.enron_email_list_df.dropna(subset=['body'])

        # Remove all Duplicate emails
        self.enron_email_list_df = self.enron_email_list_df.drop_duplicates(subset='body')

    def clean_date(self):
        # convert date to pandas datetime
        self.enron_email_list_df['date'] = pd.to_datetime(self.enron_email_list_df['date'], utc=True)
        # Keep emails between Jan 1st 1999 and Jan 1st 2003
        self.enron_email_list_df = self.enron_email_list_df[(self.enron_email_list_df.date > '1999-01-01') & (self.enron_email_list_df.date < '2003-01-01')]

    # This function removes stopwords
    def remove_stopwords(self, sen):
        stop_words = stopwords.words('english')
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # This function splits up the body into sentence tokens and removes stop words.
    def tokenize_email(self, text):
        clean_sentences = sent_tokenize(text, language='english')
        # removing punctuation, numbers and special characters. Then lowercasing.
        clean_sentences = [re.sub('[^a-zA-Z ]', '', s) for s in clean_sentences]
        clean_sentences = [s.lower() for s in clean_sentences]
        clean_sentences = [self.remove_stopwords(r.split()) for r in clean_sentences]
        return clean_sentences

    def wrangle_full_enron(self):
        self.load_enron_mail()
        self.clean_date()
        self.clean_email_df()
        # This tokenizing will be the extracted sentences that may be chosen to form the email summaries.
        self.enron_email_list_df['extractive_sentences'] = self.enron_email_list_df['body'].apply(sent_tokenize)
        # Splitting the text in emails into cleaned sentences
        self.enron_email_list_df['tokenized_body'] = self.enron_email_list_df['body'].apply(self.tokenize_email)

