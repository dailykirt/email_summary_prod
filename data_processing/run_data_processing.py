from DataWrangling import DataWrangling
from TextRank import TextRank
import gc

dw = DataWrangling()


if __name__ == '__main__':
    #Wrangle chosen directories
    dw.wrangle_full_enron()
    #force garbage collection.
    gc.collect()
    print(dw.enron_email_list_df.info())
    #Summarize each inbox that was chosen.
    tr = TextRank(dw.enron_email_list_df)
    tr.summarize_emails()

