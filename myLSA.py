from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Load data
file = r'LarsenBong2016GoldStandard.xls'
gold_items = pd.read_excel(file, sheet_name='Items')
gold_standard = pd.read_excel(file, sheet_name='GoldStandard')
variableIDs = sorted(list(set(gold_items['VariableId'])))
item_corpus = np.asarray(gold_items['Text'])

# Prepare document-term count matrix
# TODO: stem words (?)
vectorizer = CountVectorizer(min_df=1, stop_words='english', lowercase=True, dtype='int32')
dt_matrix = vectorizer.fit_transform(item_corpus)
dt_matrix = dt_matrix.toarray()
feature_names = vectorizer.get_feature_names()
print("Document_term count matrix prepared (items x words).")
# print(pd.DataFrame(dt_matrix, index=item_corpus, columns=vectorizer.get_feature_names()).head(5))

# TODO: doc-strings for all functions


def recreate_construct_identity_gold():
    # Reconstruct construct identity matrix from gold standard.
    # Creates upper triangular with zero diagonal for efficiency.
    construct_identity_gold = np.zeros([len(variableIDs), len(variableIDs)])
    construct_identity_gold = pd.DataFrame(construct_identity_gold, index=variableIDs, columns=variableIDs)
    pool_ids = gold_standard['Poolid'].unique()
    for poolID in pool_ids:
        pool_var_ids = np.asarray(gold_standard['VariableID'][gold_standard['Poolid'] == poolID])
        for ind_1 in range(len(pool_var_ids) - 1):
            var_id_1 = pool_var_ids[ind_1]
            for ind_2 in range(ind_1 + 1, len(pool_var_ids)):
                var_id_2 = pool_var_ids[ind_2]
                construct_identity_gold[var_id_1][var_id_2] = 1
    np.savetxt('construct_identity_gold.txt', construct_identity_gold)
    print("Reconstructed construct identity matrix from gold standard.")
    return construct_identity_gold


def item_vectors_svd(dt_matrix):
    # Apply log entropy and L2 normalization
    # https://radimrehurek.com/gensim/models/logentropy_model.html
    local_weight_matrix = np.log(dt_matrix + 1)
    p_matrix = np.divide(dt_matrix, np.tile(np.sum(dt_matrix, axis=0), (len(dt_matrix), 1)))
    global_weight_matrix = np.tile(1 + np.divide(np.sum(np.multiply(p_matrix, np.nan_to_num(np.log(p_matrix))),
                                                        axis=0), np.log(len(dt_matrix)) + 1), (len(dt_matrix), 1))
    final_weight_matrix = np.multiply(local_weight_matrix, global_weight_matrix)
    dt_matrix_norm = np.multiply(dt_matrix, final_weight_matrix)
    dt_matrix_norm = Normalizer(copy=False).fit_transform(dt_matrix_norm)
    print("Document-term count matrix normalized with log entropy and L2 norm.")

    # Create item vectors with SVD
    t_svd = TruncatedSVD(n_components=300, algorithm='randomized')
    item_vectors = t_svd.fit_transform(dt_matrix_norm)
    print("Item vectors created with SVD")
    return item_vectors


def item_vectors_glove(dt_matrix):
    # TODO: takes some time, could try to vectorize
    # TODO: deal with out of vocabulary words better
    # TODO: <string>:3: UserWarning: DataFrame columns are not unique, some columns will be omitted.
    # Translate items into GloVe vector space by summing GloVe vectors of words in the item multiplied with their
    # frequency and dividing by the total number of words in the item. -> frequency weighted average vectors
    try:
        item_vectors = np.loadtxt('item_vectors_GloVe.txt')
    except FileNotFoundError:
        print("Translating items into GloVe vectors. This will take some time.")
        # Load pre-trained GloVe vectors and use only relevant vectors
        with open('glove.6B/glove.6B.50d.txt', 'r') as file:
            glove_vectors_full = pd.read_table(file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
        glove_vectors_full = glove_vectors_full.transpose().to_dict(orient='list')
        glove_vectors = {}
        ctr_oov = 0
        for word in feature_names:
            try:
                glove_vectors[word] = glove_vectors_full[word]
            except KeyError:
                ctr_oov += 1
        print(ctr_oov, "out of vocabulary words.")

        # Translate items into GloVe vectors
        item_vectors = np.zeros([len(item_corpus), 50])
        for row_ind in range(len(dt_matrix)):
            ctr_oov_occurrences = 0
            for col_ind in range(len(dt_matrix[0])):
                if np.nonzero(dt_matrix[row_ind, col_ind]):
                    try:
                        item_vectors[row_ind] = np.add(item_vectors[row_ind],
                                                       np.asarray(glove_vectors[feature_names[col_ind]])
                                                       * dt_matrix[row_ind, col_ind])
                    except KeyError:
                        ctr_oov_occurrences += dt_matrix[row_ind, col_ind]
            item_vectors[row_ind] = item_vectors[row_ind] / (np.sum(dt_matrix[row_ind] - ctr_oov_occurrences))
            print("Translating items into GloVe vectors.", row_ind + 1, "of", len(dt_matrix), end="\r")
        item_vectors = np.nan_to_num(item_vectors)
        np.savetxt('item_vectors_GloVe.txt', item_vectors)
    print("Items translated into GloVe vectors.")
    return item_vectors


def compute_construct_similarity(item_vectors):
    # Compute cosine similarity of items in the vector space
    item_similarity = np.asarray(np.asmatrix(item_vectors) * np.asmatrix(item_vectors).T)
    # item_similarity = np.triu(item_similarity, 1) # upper triangular, but item_sim_sub needs full matrix
    # np.savetxt('LSA_item_similarity.csv', item_similarity)
    print("Cosine similarity of items computed.")
    print("Number unique item pairs =", np.count_nonzero(np.triu(item_similarity, 1)))

    # Aggregate item similarity to construct similarity by average of similarity of two most
    # similar items between each construct pair. Creates upper triangular with zero diagonal for efficiency.
    # TODO: slight mismatch in number of non-zero elements to expectation... see notes
    print("Aggregating item similarity to construct similarity. This will take some time.")
    construct_similarity = np.zeros([len(variableIDs), len(variableIDs)])
    # n_fields = (len(construct_similarity)**2 - len(construct_similarity)) / 2   # n fields in upper triu for print
    for ind_1 in range(len(variableIDs) - 1):  # rows
        for ind_2 in range(ind_1 + 1, len(variableIDs)):  # columns
            item_indices_1 = np.where(gold_items['VariableId'] == variableIDs[ind_1])[0]
            item_indices_2 = np.where(gold_items['VariableId'] == variableIDs[ind_2])[0]
            item_sim_sub = item_similarity[np.ix_(item_indices_1, item_indices_2)]
            sim_1, sim_2 = np.sort(item_sim_sub, axis=None)[-2:]
            sim_avg = np.average([sim_1, sim_2])
            construct_similarity[ind_1, ind_2] = sim_avg
        # TODO: turn counter into properly weighted percentage
        print("Aggregating construct similarity.", ind_1 + 1, "of", len(variableIDs), end="\r")
        # print("Aggregating construct similarity.",  / n_fields, "%")
    print("Created construct similarity matrix.")
    return construct_similarity


def evaluate(construct_similarity):
    try:
        construct_identity_gold = np.loadtxt('construct_identity_gold.txt')
    except FileNotFoundError:
        construct_identity_gold = recreate_construct_identity_gold()

    # Calculate Receiver Operating Characteristic (ROC) curve on flattened upper triangular of
    # similarity/identity matrix excluding diagonal and lower triangular
    construct_sim_flat = np.asarray([])
    construct_idn_gold_flat = np.asarray([])
    for row_ind in range(len(construct_similarity)):
        construct_sim_flat = np.append(construct_sim_flat, np.asarray(construct_similarity)[
            row_ind, range(row_ind + 1, len(construct_similarity))])
        construct_idn_gold_flat = np.append(construct_idn_gold_flat, np.asarray(construct_identity_gold)[
            row_ind, range(row_ind + 1, len(construct_identity_gold))])
    fpr, tpr, thresholds = roc_curve(construct_idn_gold_flat, construct_sim_flat)
    roc_auc = roc_auc_score(construct_idn_gold_flat, construct_sim_flat)
    return fpr, tpr, roc_auc


# Load or compute construct similarity matrix for LSA
# TODO: produces some NaN entries in item_vectors
try:
    construct_similarity_LSA = np.nan_to_num(np.loadtxt('construct_similarity_LSA.txt'))
except FileNotFoundError:
    construct_similarity_LSA = np.nan_to_num(compute_construct_similarity(item_vectors_svd(dt_matrix)))
    np.savetxt('construct_similarity_LSA.txt', construct_similarity_LSA)

# Load or compute construct similarity matrix for GloVe
# TODO: produces some NaN entries in item_vectors
try:
    construct_similarity_GloVe = np.nan_to_num(np.loadtxt('construct_similarity_GloVe.txt'))
except FileNotFoundError:
    construct_similarity_GloVe = np.nan_to_num(compute_construct_similarity(item_vectors_glove(dt_matrix)))
    np.savetxt('construct_similarity_GloVe.txt', construct_similarity_GloVe)

# construct_similarity_LSA = pd.DataFrame(construct_similarity_LSA, index=variableIDs, columns=variableIDs)
# construct_similarity_GloVe = pd.DataFrame(construct_similarity_GloVe, index=variableIDs, columns=variableIDs)

# Evaluate models
fpr_lsa, tpr_lsa, roc_auc_lsa = evaluate(construct_similarity_LSA)
fpr_glove, tpr_glove, roc_auc_glove = evaluate(construct_similarity_GloVe)
print("ROC AUC LSA =", roc_auc_lsa)
print("ROC AUC GloVe =", roc_auc_glove)

# Plot ROC curves
plt.figure()
plt.grid(True)
plt.plot(fpr_lsa, tpr_lsa)
plt.plot(fpr_glove, tpr_glove)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(["LSA", "GloVe"])
plt.show()
