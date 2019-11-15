from DataWrangling import DataWrangling
from TextRank import TextRank
import gc
import multiprocessing as mp
from functools import partial

from sklearn.metrics.pairwise import cosine_similarity

dw = DataWrangling()

def processCosineSim(sentence_vectors, index):
# Used to calculate sentence similarity
    sen_i = sentence_vectors[index[0]].reshape(1, 300)
    sen_j = sentence_vectors[index[1]].reshape(1, 300)
    return cosine_similarity(sen_i, sen_j)[0, 0]
    #return (index[0], index[1], cosine_similarity(sen_i, sen_j)[0, 0])


def generate_indexes(sen_len):
    # Generator to produce sentence indexes. For very large inboxes,
    # the window limits comparisons to sentences that are close timewise. Attempt to get total comparisons to under a million per 1K emails. 
    window = (10000000 // sen_len) + 1
    print("window: " + str(window))
    for i in range(sen_len):
        for j in range(sen_len):
            diff = j - i
            if (i != j) and (i < j) and (diff < window):
                yield (i, j)

if __name__ == '__main__':
    #Attempt to get number of cpus
    try:
        cpus = mp.cpu_count()
    except NotImplementedError:
        cpus = 2
    pool = mp.Pool(processes=cpus)

    window = 1000
    print("CPUS: " + str(cpus))
    #Wrangle chosen directories
    print("Wrangle enron mailbox")
    #dw.wrangle_full_enron()
    dw.get_inboxes()
    #force garbage collection.
    gc.collect()
    #print(dw.enron_email_list_df.info())
    #Summarize each inbox that was chosen.
    for mailbox in dw.mailboxes:
        dw.wrangle_mailbox(mailbox)
        tr = TextRank(dw.enron_email_list_df)

        #summarize sentences in each inbox
        for employee in tr.email_df['employee'].unique():

            print("Processing: " + str(employee))
            employee_df = tr.subset_emails(employee)

            df_wind = len(employee_df) // window
            #At most, do 1K emails at a time.
            for i in range(0, df_wind+1):
                print("ranking subset: " + str(i))
                index = i * window
                if (i-df_wind == 0):
                    email_masked_df = employee_df.iloc[index:, :]
                else:
                    email_masked_df = employee_df.iloc[index:(index + window), :]

                print(email_masked_df.info())
                print("Get extractive sentences")
                sentences = tr.get_extractive_sentences(email_masked_df)
                num_sen = len(sentences)
                print("Number of sentences: " + str(num_sen))
                print("get sentence vectors. ")
                sentence_vectors = tr.get_sentence_vectors(email_masked_df)
                print("Start multiprocessing pool")
                func = partial(processCosineSim, sentence_vectors)

                indexes = generate_indexes(num_sen)
                #Start multiprocessing similarity matrix
                result = pool.imap(func, indexes, chunksize=(num_sen // cpus))

                #reinitalize generator
                indexes = generate_indexes(len(sentences))
                print("Rank Sentences")
                ranked_sentences = tr.rank_sentences(sentences, result, indexes)
                #print(ranked_sentences)
                print("Append rank and push to database")
                email_masked_df = tr.append_rank_df(ranked_sentences, email_masked_df)
                print(email_masked_df)
                tr.insert_db(email_masked_df)

    pool.close()
    pool.join()
    #tr.summarize_emails()

