from DataWrangling import DataWrangling
from TextRank import TextRank

dw = DataWrangling()


if __name__ == '__main__':
    dw.wrangle_full_enron()

    print(dw.enron_email_list_df.info())
    tr = TextRank(dw.enron_email_list_df)
    tr.summarize_emails()

