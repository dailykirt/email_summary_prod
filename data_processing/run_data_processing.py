from DataWrangling import DataWrangling
from TextRank import TextRank
import gc
import multiprocessing as mp
from functools import partial
import logging
from sklearn.metrics.pairwise import cosine_similarity
import sys

LOG_NAME = '__data_processing__'
LOG_LOC = '../logs/processing.log'
#Limits # of emails to process at a time for memory purposes.
#EMAIL_WINDOW = 1000
EMAIL_WINDOW = 100
SENTENCE_WINDOW = 10000000

def set_logger():
    """Configures the logger."""
    logger = logging.getLogger(LOG_NAME)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename=LOG_LOC, mode='a')
    format = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')
    handler.setFormatter(format)
    logger.addHandler(handler)
    return logger

def get_cpus():
    """Attempt to get number of cpus"""
    try:
        cpus = mp.cpu_count()
    except NotImplementedError:
        cpus = 2
    return cpus

def processCosineSim(sentence_vectors, index):
    """This function is pulled out for multiprocessing purposes. Calculates similarity between sentences."""
    sen_i = sentence_vectors[index[0]].reshape(1, 300)
    sen_j = sentence_vectors[index[1]].reshape(1, 300)
    return cosine_similarity(sen_i, sen_j)[0, 0]

if __name__ == '__main__':
    logger = set_logger()
    logger.info("-----------Start Email Processing-----------")
    cpus = get_cpus()
    logger.info("Number of CPUS: " + str(cpus))
    pool = mp.Pool(processes=cpus)

    dw = DataWrangling()
    dw.get_inboxes()

    for mailbox in dw.mailboxes:
        dw.wrangle_mailbox(mailbox)
        tr = TextRank(dw.enron_email_list_df)
        df_wind = len(tr.email_df) // EMAIL_WINDOW

        #Process window sized chunks of emails at a time to keep within memory constraints.
        for i in range(0, df_wind+1):
            logger.info("Ranking subset: " + str(i))
            index = i * EMAIL_WINDOW
            if (i-df_wind == 0):
                email_masked_df = tr.email_df.iloc[index:, :]
            else:
                email_masked_df = tr.email_df.iloc[index:(index + EMAIL_WINDOW), :]

            logger.info("Starting TextRank Preprocessing")
            sentences = tr.get_extractive_sentences(email_masked_df)
            sentence_vectors = tr.get_sentence_vectors(email_masked_df)
            num_sen = len(sentences)
            window = (SENTENCE_WINDOW // num_sen) + 1

            logger.info("Number of sentences to rank: " + str(num_sen))
            logger.info("Sentence window: " + str(window))

            logger.info("Start multiprocessing pool and create similarity matrix. ")
            func = partial(processCosineSim, sentence_vectors)
            indexes = tr.generate_indexes(num_sen, window)
            result = pool.imap(func, indexes, chunksize=(num_sen // cpus))

            indexes = tr.generate_indexes(num_sen, window)
            try:
                ranked_sentences = tr.rank_sentences(sentences, result, indexes)
                email_masked_df = tr.append_rank_df(ranked_sentences, email_masked_df)
                logger.info(email_masked_df)
                tr.insert_db(email_masked_df)
            except:
                logger.info("PowerIterationFailedConvergence at inbox " + str(mailbox) + "in subset " + str(i))
        gc.collect()

    pool.close()
    pool.join()

