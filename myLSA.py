from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# Load data
file = r'LarsenBong2016GoldStandard.xls'
gold_items = pd.read_excel(file, sheet_name='Items')
gold_standard = pd.read_excel(file, sheet_name='GoldStandard')
corpus = np.asarray(gold_items['Text'])

# Prepare document-term count matrix
# TODO: stem words (?)
vectorizer = CountVectorizer(min_df=1, stop_words='english', lowercase=True, dtype='int32')
dt_matrix = vectorizer.fit_transform(corpus)
dt_matrix = dt_matrix.toarray()
print("Term-document count matrix prepared.")
# print(pd.DataFrame(dt_matrix, index=corpus, columns=vectorizer.get_feature_names()).head(5))

# Apply log entropy and L2 normalization
# https://radimrehurek.com/gensim/models/logentropy_model.html
local_weight_matrix = np.log(dt_matrix + 1)
P_matrix = np.divide(dt_matrix,
                     np.tile(np.sum(dt_matrix, axis=0), (len(dt_matrix), 1)))
global_weight_matrix = np.tile(1 + np.divide(np.sum(np.multiply(P_matrix, np.nan_to_num(np.log(P_matrix))), axis=0),
                                             np.log(len(dt_matrix)) + 1), (len(dt_matrix), 1))
final_weight_matrix = np.multiply(dt_matrix, global_weight_matrix)
dt_matrix = np.multiply(dt_matrix, final_weight_matrix)
dt_matrix = Normalizer(copy=False).fit_transform(dt_matrix)
print("Document-term count matrix normalized with log entropy and L2 norm.")

# Create document vectors with SVD
tSVD = TruncatedSVD(n_components=300, algorithm='randomized')
document_vectors = tSVD.fit_transform(dt_matrix)
print("Document vectors created with SVD")

# Plot squared singular values to estimate number of components
# s = tSVD.singular_values_
# plt.figure()
# plt.bar(range(len(s)), s ** 2)
# plt.show()

# Compute cosine similarity of documents in vector space
item_similarity = np.asarray(np.asmatrix(document_vectors) * np.asmatrix(document_vectors).T)
# item_similarity = np.triu(item_similarity, 1) # upper triangular, but item_sim_sub needs full matrix
# np.savetxt('LSA_document_similarity.csv', item_similarity)
print("Cosine similarity of documents computed.")
print("Number unique document pairs =", np.count_nonzero(np.triu(item_similarity, 1)))

# Aggregate document (e.g. item) similarity to construct similarity by average of similarity of two most
# similar items between each construct pair. Creates upper triangular with zero diagonal for efficiency.
# TODO: slight mismatch in number of non-zero elements to expectation... see notes
variableIDs = sorted(list(set(gold_items['VariableId'])))
construct_similarity = np.zeros([len(variableIDs), len(variableIDs)])
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
# construct_similarity = construct_similarity + np.triu(construct_similarity, 1).T  # mirror diagonally
# np.savetxt('LSA_construct_similarity.csv', item_similarity)
construct_similarity = pd.DataFrame(construct_similarity, index=variableIDs, columns=variableIDs)
print("Created construct similarity matrix.")

# Reconstruct construct identity matrix from gold standard.
# Creates upper triangular with zero diagonal for efficiency.
construct_identity_gold = np.zeros([len(variableIDs), len(variableIDs)])
construct_identity_gold = pd.DataFrame(construct_identity_gold, index=variableIDs, columns=variableIDs)
poolIDs = gold_standard['Poolid'].unique()
for poolID in poolIDs:
    pool_varIDs = np.asarray(gold_standard['VariableID'][gold_standard['Poolid'] == poolID])
    for ind_1 in range(len(pool_varIDs) - 1):
        varID_1 = pool_varIDs[ind_1]
        for ind_2 in range(ind_1 + 1, len(pool_varIDs)):
            varID_2 = pool_varIDs[ind_2]
            construct_identity_gold[varID_1][varID_2] = 1
# construct_identity_gold = construct_identity_gold + np.triu(construct_identity_gold, 1).T  # mirror diagonally
print("Reconstructed construct identity matrix from gold standard.")

# Calculate Receiver Operating Characteristic (ROC) curve on flattened upper triangular of
# similarity/identity matrix excluding diagonal and lower triangular
construct_sim_flat = np.asarray([])
construct_id_gold_flat = np.asarray([])
for row_ind in range(len(construct_similarity)):
    construct_sim_flat = np.append(construct_sim_flat, np.asarray(construct_similarity)[
        row_ind, range(row_ind + 1, len(construct_similarity))])
    construct_id_gold_flat = np.append(construct_id_gold_flat, np.asarray(construct_identity_gold)[
        row_ind, range(row_ind + 1, len(construct_identity_gold))])
fpr, tpr, thresholds = roc_curve(construct_id_gold_flat, construct_sim_flat)
roc_auc = roc_auc_score(construct_id_gold_flat, construct_sim_flat)
print("Area under the curve of ROC =", roc_auc)

# Plot ROC curve
plt.figure()
plt.grid(True)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(["LSA"])
plt.show()
