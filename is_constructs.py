from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def create_dt_matrices():
    """Creates and returns a dictionary of document-term matrices with different processing methods.
    Also returns the feature names (terms) extracted by the vectorizer."""
    # TODO: stem words (?)
    count_vectorizer = CountVectorizer(stop_words='english', lowercase=True, dtype='int32')
    dt_matrix = count_vectorizer.fit_transform(item_corpus)
    dt_matrix = dt_matrix.toarray()
    dt_matrix_l2 = Normalizer(copy=True, norm='l2').fit_transform(dt_matrix)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, norm='l2', use_idf=True, smooth_idf=True)
    dt_matrix_tfidf_l2 = tfidf_vectorizer.fit_transform(item_corpus)
    dt_matrix_tfidf_l2 = dt_matrix_tfidf_l2.toarray()
    feature_names = count_vectorizer.get_feature_names()
    # print(pd.DataFrame(dt_matrix, index=item_corpus, columns=count_vectorizer.get_feature_names()).head(5))

    # Apply log entropy and L2 normalization to count matrix.
    # https://radimrehurek.com/gensim/models/logentropy_model.html
    # Implementation checked manually.
    local_weight_matrix = np.log(dt_matrix + 1)
    p_matrix = np.divide(dt_matrix, np.tile(np.sum(dt_matrix, axis=0), (len(dt_matrix), 1)))
    log_p_matrix = np.log(p_matrix)
    log_p_matrix[np.isneginf(log_p_matrix)] = 0
    global_weight_matrix = np.tile(1 + np.divide(np.sum(np.multiply(p_matrix, log_p_matrix),
                                                        axis=0), np.log(len(dt_matrix) + 1)), (len(dt_matrix), 1))
    final_weight_matrix = np.multiply(local_weight_matrix, global_weight_matrix)
    dt_matrix_log = np.multiply(dt_matrix, final_weight_matrix)
    dt_matrix_log_l2 = Normalizer(copy=True, norm='l2').fit_transform(dt_matrix_log)
    print("Document-term matrices prepared (items x terms).")

    dt_matrices = {
        'dt_matrix': dt_matrix,
        'dt_matrix_l2': dt_matrix_l2,
        'dt_matrix_log': dt_matrix_log,
        'dt_matrix_log_l2': dt_matrix_log_l2,
        'dt_matrix_tfidf_l2': dt_matrix_tfidf_l2,
    }
    return feature_names, dt_matrices


def recreate_construct_identity_gold():
    """Translates the gold standard by Larsen and Bong 2016 into a binary construct identity matrix.
    Creates upper triangular with zero diagonal for efficiency."""
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


def item_vectors_svd(dtm_identifier):
    """Create item vectors with SVD a.k.a. LSA."""
    t_svd = TruncatedSVD(n_components=300, algorithm='randomized')
    item_vectors = t_svd.fit_transform(dt_matrices[dtm_identifier])
    print("LSA item vectors created with SVD.")
    return item_vectors


def item_vectors_glove(dtm_identifier, sub_dict_file, denominator=None, full_dict_file=None):
    """Translate items into pre-trained GloVe vector space and returns the matrix of item vectors. Sums term
    vectors of terms in item with passed document-term matrix as weighting factor. Pass file of relevant dictionary
    or pass dict_file=None and pass the file of the full vector dictionary to create the sub-dictionary."""
    # TODO: deal with out of vocabulary words better
    # Implementation of dict checked.
    if sub_dict_file is not None:
        glove_vector_dict = np.load(sub_dict_file).item()
    elif full_dict_file is not None:
        # Load full pre-trained GloVe vector dictionary and extract relevant term vectors.
        with open(full_dict_file, 'r') as file:
            glove_vectors_full = pd.read_table(file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
        # UserWarning: DataFrame columns are not unique, some columns will be omitted. This is caused by two
        # duplicate NaN indices and reduces the vocabulary to 399998 words.
        glove_vectors_full = glove_vectors_full.transpose().to_dict(orient='list')
        glove_vector_dict = {}
        ctr_oov = 0
        ctr = 0
        for word in feature_names:
            try:
                glove_vector_dict[word] = glove_vectors_full[word]
            except KeyError:
                ctr_oov += 1
            ctr += 1
            print("Creating GloVe vector dictionary of relevant terms.", ctr / len(feature_names) * 100, "%",
                  end="\r")
        print(ctr_oov, "out of vocabulary words.")
        np.save('GloVe_vector_dict.npy', glove_vector_dict)
    else:
        raise FileNotFoundError

    # Translate items into GloVe vectors.
    # Implementation checked manually, but there are multiple ways of doing this.
    dt_matrix = dt_matrices[dtm_identifier]
    item_vectors = np.zeros([len(dt_matrix), len(next(iter(glove_vector_dict.values())))])
    for row_ind in range(len(dt_matrix)):
        ctr_oov = 0
        value_oov = 0
        for col_ind in range(len(dt_matrix[0])):
            if np.nonzero(dt_matrix[row_ind, col_ind]):
                try:
                    item_vectors[row_ind] = np.add(item_vectors[row_ind],
                                                   np.asarray(glove_vector_dict[feature_names[col_ind]])
                                                   * dt_matrix[row_ind, col_ind])
                except KeyError:
                    ctr_oov += 1
                    value_oov += dt_matrix[row_ind, col_ind]
        # TODO: Some error, probably created in the following lines. Test this part of the function.
        item_vectors[row_ind] = {
            'sum': item_vectors[row_ind] / (np.sum(dt_matrix[row_ind])),
            'sum_value-oov': item_vectors[row_ind] / (np.sum(dt_matrix[row_ind]) - value_oov),
            'norm': item_vectors[row_ind] / (np.linalg.norm(dt_matrix[row_ind])),
            'norm_value-oov': item_vectors[row_ind] / (np.linalg.norm(dt_matrix[row_ind]) - value_oov)
        }.get(denominator, item_vectors[row_ind])
        if row_ind % 100 == 0:
            print("Translating items into GloVe vectors.", (row_ind + 1) / len(dt_matrix) * 100, "%", end="\r")
    return item_vectors


def compute_construct_similarity(item_vectors):
    """Computes construct similarities from items in vector space. To aggregate item cosine similarity to construct
    similarity, the average similarity of the two most similar items between each construct pair is taken, as
    established by Larsen and Bong 2016. Creates upper triangular with zero diagonal for efficiency."""
    # Compute cosine similarity of items in the vector space
    item_similarity = np.asarray(np.asmatrix(item_vectors) * np.asmatrix(item_vectors).T)
    print("Cosine similarity of items computed. Number unique item pairs =",
          np.count_nonzero(np.triu(item_similarity, 1)))

    # TODO: slight mismatch in number of non-zero elements to expectation... see notes
    construct_similarity = np.zeros([len(variableIDs), len(variableIDs)])
    n_fields = (len(construct_similarity) ** 2 - len(construct_similarity)) / 2  # n fields in upper triu for print
    ctr = 0  # counter for print
    for ind_1 in range(len(variableIDs) - 1):  # rows
        for ind_2 in range(ind_1 + 1, len(variableIDs)):  # columns
            item_indices_1 = np.where(gold_items['VariableId'] == variableIDs[ind_1])[0]
            item_indices_2 = np.where(gold_items['VariableId'] == variableIDs[ind_2])[0]
            item_sim_sub = item_similarity[np.ix_(item_indices_1, item_indices_2)]
            sim_1, sim_2 = np.sort(item_sim_sub, axis=None)[-2:]
            sim_avg = np.average([sim_1, sim_2])
            construct_similarity[ind_1, ind_2] = sim_avg
            ctr += 1
        if ind_1 % 20 == 0:
            print("Aggregating to construct similarity.", ctr / n_fields * 100, "%", end='\r')
    return construct_similarity


def evaluate(construct_similarity):
    """Evaluates construct similarity matrix against the Larsen and Bong 2016 gold standard in matrix form."""
    print("Evaluating performance.")
    try:
        construct_identity_gold = np.loadtxt('construct_identity_gold.txt')
    except FileNotFoundError:
        construct_identity_gold = recreate_construct_identity_gold()

    # Unwrap upper triangular of similarity and identity matrix, excluding diagonal.
    # Calculate Receiver Operating Characteristic (ROC) curve.
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


# Load data.
file = r'LarsenBong2016GoldStandard.xls'
gold_items = pd.read_excel(file, sheet_name='Items')
gold_standard = pd.read_excel(file, sheet_name='GoldStandard')
variableIDs = sorted(list(set(gold_items['VariableId'])))
item_corpus = np.asarray(gold_items['Text'])

# Create document-term matrices.
feature_names, dt_matrices = create_dt_matrices()

# Load or compute construct similarity matrix for LSA.
# TODO: produced some NaN entries in item_vectors
try:
    construct_similarity_LSA = np.loadtxt('construct_similarity_LSA.txt')
except FileNotFoundError:
    construct_similarity_LSA = np.nan_to_num(compute_construct_similarity(
        item_vectors_svd(dt_matrices['dt_matrix_tfidf_l2'])))
    np.savetxt('construct_similarity_LSA.txt', construct_similarity_LSA)

# Load or compute construct similarity matrix for GloVe.
# TODO: produced some NaN entries in item_vectors
try:
    construct_similarity_GloVe = np.loadtxt('construct_similarity_GloVe.txt')
except FileNotFoundError:
    construct_similarity_GloVe = np.nan_to_num(compute_construct_similarity(
        item_vectors_glove(dt_matrices['dt_matrix_tfidf_l2'], sub_dict_file='GloVe_vector_dict_6B_50d.npy',
                           full_dict_file='glove.6B/glove.6B.50d.txt')))
    np.savetxt('construct_similarity_GloVe.txt', construct_similarity_GloVe)

# Search for best GloVe document projection
denominator_options = [None, 'sum', 'sum_value-oov', 'norm', 'norm_value-oov']
# grid = [[mat, den] for mat in dt_matrices for den in denominator_options]
grid = [['dt_matrix', None], ['dt_matrix', 'sum']]
glove_results = []
ctr = 0
print("Performing grid search on GloVe.")
for dtm_identifier, denominator in grid:
    item_vectors_glove(dtm_identifier, denominator=denominator, sub_dict_file='GloVe_vector_dict_6B_100d.npy')
    construct_similarity_GloVe = np.nan_to_num(compute_construct_similarity(item_vectors_glove))
    fpr_glove, tpr_glove, roc_auc_glove = evaluate(construct_similarity_GloVe)
    glove_results.append([dtm_identifier, denominator, roc_auc_glove])
    ctr += 1
    print("Grid search on GloVe.", ctr / len(grid) * 100, "%")

# construct_similarity_LSA = pd.DataFrame(construct_similarity_LSA, index=variableIDs, columns=variableIDs)
# construct_similarity_GloVe = pd.DataFrame(construct_similarity_GloVe, index=variableIDs, columns=variableIDs)

# Evaluate models
fpr_lsa, tpr_lsa, roc_auc_lsa = evaluate(construct_similarity_LSA)
print("ROC AUC LSA =", roc_auc_lsa)
fpr_glove, tpr_glove, roc_auc_glove = evaluate(construct_similarity_GloVe)
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
