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

#def processCosineSim(sentence_vec):
#    return cosine_similarity(sentence_vec[0], sentence_vec[1])

#def main():

if __name__ == '__main__':
    #Attempt to get number of cpus
    try:
        cpus = mp.cpu_count()
    except NotImplementedError:
        cpus = 2
    pool = mp.Pool(processes=cpus)
    print("CPUS: " + str(cpus))
    #Wrangle chosen directories
    print("Wrangle enron mailbox")
    dw.wrangle_full_enron()
    #force garbage collection.
    gc.collect()
    print(dw.enron_email_list_df.info())
    #Summarize each inbox that was chosen.
    tr = TextRank(dw.enron_email_list_df)

    #summarize sentences in each inbox
    for employee in tr.email_df['employee'].unique():

        print("Pre ranking work")
        email_masked_df = tr.subset_emails(employee)
        sentences = tr.get_extractive_sentences(email_masked_df)
        indexes = tr.unroll_rank_indexes(len(sentences))
        sentence_vectors = tr.get_sentence_vectors(email_masked_df)

        #preprocess indexed sentence vectors
        #full_vectors = []
        #for index in indexes:
        #    sen_i = sentence_vectors[index[0]].reshape(1, 300)
        #    sen_j = sentence_vectors[index[1]].reshape(1, 300)
        #    full_vectors.append([sen_i, sen_j])

        #Start multiprocessing similarity matrix
        print("Start multiprocessing pool")
        pool = mp.Pool(processes=cpus)
        func = partial(processCosineSim, sentence_vectors)
        #result = pool.map(func, indexes)
        #with chunk size 20
        result = pool.imap(func, indexes, 20)
        #test function
        #result = pool.map(processCosineSim, full_vectors)
        pool.close()
        pool.join()
        #ranked_sentences = tr.rank_sentences(sentences, result, indexes)
        ranked_sentences = tr.rank_sentences(sentences, list(result), indexes)
        print("Append rank and push to database")
        email_masked_df = tr.append_rank_df(ranked_sentences, email_masked_df)
        print(email_masked_df)
        tr.insert_db(email_masked_df)
    #tr.summarize_emails()

