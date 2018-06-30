from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from stemming.porter2 import stem as stem_porter2
from stemming.paicehusk import stem as stem_paicehusk

import numpy as np
import pandas as pd
import glove
import csv
import matplotlib.pyplot as plt


def info(var):
    if isinstance(var, np.ndarray):
        print("Type:", type(var), "\nShape:", np.shape(var))
    else:
        print("Type:", type(var), "\nLength:", len(var))


def recreate_construct_identity_gold(gold_standard, pool_ids):
    """Translates the gold standard by Larsen and Bong 2016 into a binary construct identity matrix."""
    # Implementation checked 28 June.
    variable_ids = np.sort(np.unique(gold_standard['VariableID'][gold_standard['Poolid'].isin(pool_ids)]))
    construct_identity_gold = np.zeros([len(variable_ids), len(variable_ids)])
    construct_identity_gold = pd.DataFrame(construct_identity_gold, index=variable_ids, columns=variable_ids)
    for pool_id in pool_ids:
        pool_var_ids = np.asarray(gold_standard['VariableID'][gold_standard['Poolid'] == pool_id])
        for ind_1 in range(len(pool_var_ids) - 1):
            var_id_1 = pool_var_ids[ind_1]
            for ind_2 in range(ind_1 + 1, len(pool_var_ids)):
                var_id_2 = pool_var_ids[ind_2]
                indices = np.sort(np.asarray([var_id_1, var_id_2]))  # necessary to get upper triangular
                construct_identity_gold[indices[1]][indices[0]] = 1
    # Mirror the matrix diagonally and fill diagonal with ones.
    construct_identity_gold = np.add(np.asarray(construct_identity_gold), np.asarray(construct_identity_gold.T))
    np.fill_diagonal(construct_identity_gold, 1)
    construct_identity_gold = pd.DataFrame(construct_identity_gold, index=variable_ids, columns=variable_ids)
    return construct_identity_gold


def test_rcig():
    gold_standard = pd.DataFrame([[1, 1],
                                  [1, 3],
                                  [2, 4],
                                  [2, 7],
                                  [2, 8]], columns=['Poolid', 'VariableID'])
    pool_ids = [1, 2]
    result = recreate_construct_identity_gold(gold_standard, pool_ids)
    print(result, "\n")
    info(result)


def load_data(prototype=False, verbose=False):
    """Loads gold_items, pool_ids, variable_ids, construct_identity_gold and funk_papers."""
    # TODO: unit testing
    file = r'LarsenBong2016GoldStandard.xls'
    gold_standard = pd.read_excel(file, sheet_name='GoldStandard')
    gold_items = pd.read_excel(file, sheet_name='Items')
    if prototype:
        try:
            pool_ids = np.loadtxt('pool_ids_prototype.txt')
        except FileNotFoundError:
            if verbose:
                print("No file with prototype pool-IDs found. Drawing new random sample...")
            pool_ids = np.sort(np.random.choice(gold_standard['Poolid'].unique(), size=100, replace=False))
            np.savetxt('pool_ids_prototype.txt', pool_ids)
        variable_ids = np.sort(np.unique(gold_standard['VariableID'][gold_standard['Poolid'].isin(pool_ids)]))
    else:
        pool_ids = np.sort(gold_standard['Poolid'].unique())
        # variable_ids = sorted(list(set(gold_items['VariableId'])))
        variable_ids = np.sort(gold_items['VariableId'].unique())
    gold_items = gold_items.loc[gold_items['VariableId'].isin(variable_ids)]

    # Load the gold standard as matrix DataFrame
    if prototype:
        try:
            construct_identity_gold = np.loadtxt('construct_identity_gold_prototype.txt')
        except FileNotFoundError:
            if verbose:
                print("No construct identity gold matrix file found. Creating new file...")
            construct_identity_gold = recreate_construct_identity_gold(gold_standard, pool_ids)
            np.savetxt('construct_identity_gold_prototype.txt', construct_identity_gold)
    else:
        try:
            construct_identity_gold = np.loadtxt('construct_identity_gold.txt')
        except FileNotFoundError:
            if verbose:
                print("No construct identity gold matrix file found. Creating new file...")
            construct_identity_gold = recreate_construct_identity_gold(gold_standard, pool_ids)
            np.savetxt('construct_identity_gold.txt', construct_identity_gold)

    # Load Funk's data. For now just the paper abstracts.
    file = r'datasetFunk/FunkPapers.xlsx'
    funk_papers = pd.read_excel(file)

    return gold_items, pool_ids, variable_ids, construct_identity_gold, funk_papers


def parse_text(documents, stemmer=None, lower=True, remove_stop_words=True,
               return_config=False, ignore_chars='''.,:;"!?-/()[]{}0123456789''', verbose=False):
    """Parses text with options for removing specified characters, removing stop-words, converting to lower-case
    and stemming (https://pypi.org/project/stemming/1.0/). Available stemming algorithms are 'porter2' and
    'paicehusk'. Paice/Husk seems prone to over-stemming."""
    # TODO: still returns empty '' strings. Also returns "'s".
    # Implementation checked 28 June.
    parsed_docs = []
    error_words = []
    for i in range(len(documents)):
        assert isinstance(documents[i], str), "Document not a string." + str(documents[i])
        if ignore_chars != '':
            documents[i] = documents[i].translate({ord(c): ' ' for c in ignore_chars})
        if lower:
            documents[i] = documents[i].lower()
        parsed_docs.append('')
        for word in documents[i].split():
            if remove_stop_words and word in stop_words.ENGLISH_STOP_WORDS:
                continue
            if word == '':
                continue
            if stemmer is not None:
                try:
                    # TODO: remove default, check if stemmer exists instead
                    parsed_docs[i] += {
                        # 'lovins': stem_lovins(word) + ' ', results in errors with all three algorithms, unknown cause
                        'porter2': stem_porter2(word) + ' ',
                        'paicehusk': stem_paicehusk(word) + ' '
                    }.get(stemmer, word + ' ')
                except ValueError:
                    error_words.append(word)
                    parsed_docs[i] += word + ' '
            else:
                parsed_docs[i] += word + ' '
        parsed_docs[i] = parsed_docs[i].strip()  # remove excess white space
    if verbose and error_words:
        print("ValueError occurred when stemming the following words:", list(set(error_words)))
    parsed_docs = list(filter(None, parsed_docs))
    parsed_docs = np.asarray(parsed_docs)
    parser_config = {'stemmer': stemmer, 'lower': lower, 'remove_stop_words': remove_stop_words,
                     'ignore_chars': ignore_chars}
    if return_config:
        return parsed_docs, parser_config
    else:
        return parsed_docs


def test_pt():
    documents = np.asarray(['It\'s a technologically advanced situation.',
                            'I (Mary) don\'t like the system in this situation.',
                            'I am.',
                            '000 Technological greatness in a system is something.',
                            'Yes, sir (no, sir?): That\'s the question.'])
    stemmer = 'porter2'
    lower = True
    remove_stop_words = True
    return_config = True
    ignore_chars = '''.,:;"!?-/()[]{}0123456789'''
    verbose = True
    result_1, result_2 = parse_text(documents, stemmer=stemmer, lower=lower, remove_stop_words=remove_stop_words,
                                    return_config=return_config, ignore_chars=ignore_chars, verbose=verbose)
    print(result_1, "\n", result_2, "\n")
    info(result_1)
    info(result_2)


def document_term_cooccurrence(corpus, processing='l2'):
    """Creates and returns document-term matrix DataFrame with the specified processing method.
    Also returns the feature names (terms) extracted by the vectorizer. Available processing methods are
    'count', 'l2', 'tfidf_l2' and 'log_l2'."""
    # Implementation checked superficially 28 June.
    count_vectorizer = CountVectorizer(stop_words=None, lowercase=False, dtype='int32')
    dt_matrix = count_vectorizer.fit_transform(corpus)
    dt_matrix = dt_matrix.toarray()
    terms = count_vectorizer.get_feature_names()
    if processing == 'count':
        return pd.DataFrame(dt_matrix, index=corpus, columns=terms), terms
    if processing == 'l2':
        dt_matrix_l2 = Normalizer(copy=True, norm='l2').fit_transform(dt_matrix)
        return pd.DataFrame(dt_matrix_l2, index=corpus, columns=terms), terms
    if processing == 'tfidf_l2':
        tfidf_vectorizer = TfidfVectorizer(stop_words=None, lowercase=False, norm='l2', use_idf=True, smooth_idf=True)
        dt_matrix_tfidf_l2 = tfidf_vectorizer.fit_transform(corpus)
        dt_matrix_tfidf_l2 = dt_matrix_tfidf_l2.toarray()
        return pd.DataFrame(dt_matrix_tfidf_l2, index=corpus, columns=terms), terms
    # print(pd.DataFrame(dt_matrix, index=item_corpus, columns=count_vectorizer.get_feature_names()).head(5))
    if processing == 'log_l2':
        # TODO: Gives RuntimeWarning: divide by zero encountered in log
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
        return pd.DataFrame(dt_matrix_log_l2, index=corpus, columns=terms), terms
    assert False, "chosen processing method not implemented."


def test_dtc():
    # Using parsed test corpus.
    corpus = np.asarray(['it technolog advanc situat',
                         "mari don't like situat",
                         'technolog great',
                         'yes sir sir that question'])
    processing = 'tfidf_l2'
    result_1, result_2 = document_term_cooccurrence(corpus, processing=processing)
    print(result_1, "\n", np.asarray(result_1), "\n", result_2, "\n")
    print(np.linalg.norm(np.asarray(result_1), axis=1))
    info(result_1)
    info(result_2)


def term_term_cooccurrence(dt_matrix):
    """Creates sparse term-term cooccurrence dictionary from dot product of passed document-term matrix."""
    # Implementation checked 30 June.
    # Index terms in corpus, both as index -> term and as term -> index to translate in both directions.
    s = ' '
    terms = dt_matrix.columns.values
    dict_ix_term = {i: terms[i] for i in range(len(terms))}
    dict_term_ix = {v: k for k, v in dict_ix_term.items()}
    terms_ix = [dict_term_ix[term] for term in terms]
    tt_matrix = np.asarray(dt_matrix).T.dot(np.asarray(dt_matrix))
    tt_matrix = pd.DataFrame(tt_matrix, index=terms_ix, columns=terms_ix)
    # Convert term-term co-occurrence matrix to sparse term-term co-occurrence dictionary.
    tt_dict = {i: {} for i in range(len(terms_ix))}
    for i in terms_ix:
        for k in terms_ix:
            if tt_matrix[i][k] != 0:
                try:
                    tt_dict[i][k] += float(tt_matrix[i][k])
                except KeyError:
                    tt_dict[i][k] = float(tt_matrix[i][k])
    return tt_dict, dict_term_ix, dict_ix_term


def test_ttc():
    dt_matrix = np.asarray([[1, 0, 1, 0, 0, 1, 1, 0, 0],
                            [0, 1, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 2, 0, 0, 1, 1]])
    documents = np.asarray(['it technolog advanc situat',
                            'technolog great',
                            'yes sir sir that question'])
    terms = np.asarray(['advanc', 'great', 'it', 'question', 'sir', 'situat', 'technolog', 'that', 'yes'])
    dt_matrix = pd.DataFrame(dt_matrix, index=documents, columns=terms)
    result_1, result_2, result_3 = term_term_cooccurrence(dt_matrix)
    print(result_1, "\n", result_2, "\n", result_3, "\n")
    info(result_1)
    info(result_2)
    info(result_3)


def term_vectors_from_dict(vector_dict, target_terms, normalize=True, verbose=False):
    """Creates term-vector matrix DataFrame of passed terms from passed vector dict."""
    # TODO: deal with OOV words better than just setting a zero vector.
    # Implementation checked 28 June.
    term_vectors = np.zeros([len(target_terms), len(next(iter(vector_dict.values())))])
    i = 0
    ctr_oov = 0
    for term in target_terms:
        try:
            term_vectors[i] = vector_dict[term]
        except KeyError:
            term_vectors[i] = np.zeros(len(next(iter(vector_dict.values()))))
            ctr_oov += 1
        i += 1
    if verbose:
        print("Created term vectors from dictionary.", ctr_oov, "OOV words.")
    if normalize:
        term_vectors = Normalizer(norm='l2', copy=True).fit_transform(term_vectors)
    term_vectors = pd.DataFrame(term_vectors, index=target_terms)
    return term_vectors


def test_tvfd():
    vector_dict = {'it': [0.2, 0.4, -0.1],
                   'technolog': [0.7, -0.9, -0.2],
                   'advanc': [0.6, -0.9, 0],
                   'green': [-0.6, -0.5, -0.4],
                   'lime': [0.3, 0.6, 0.8]
                   }
    target_terms = ['it', 'technolog', 'advanc', 'situat']
    normalize = True
    verbose = True
    result = term_vectors_from_dict(vector_dict, target_terms, normalize=normalize, verbose=verbose)
    print(result, "\n")
    info(result)


def train_vectors_lsa(dt_matrix, n_components=300, return_doc_vectors=False):
    """Train term and item vectors with SVD a.k.a. LSA."""
    # Implementation checked 28 June.
    assert len(dt_matrix) >= n_components, "Number of training documents has to be >= number of components."
    documents = dt_matrix.index.values
    terms = dt_matrix.columns.values
    t_svd = TruncatedSVD(n_components=n_components, algorithm='randomized')
    doc_vectors = t_svd.fit_transform(np.asarray(dt_matrix))
    doc_vectors = pd.DataFrame(doc_vectors, index=documents)
    source_term_vectors = t_svd.components_
    source_term_vectors = pd.DataFrame(source_term_vectors, columns=terms)
    vector_dict = source_term_vectors.to_dict(orient='list')
    if return_doc_vectors:
        return vector_dict, doc_vectors
    else:
        return vector_dict


def test_ttvlsa():
    dt_matrix = np.asarray([[0.61449708, 0., 0., 0.61449708, 0., 0., 0., 0., 0.34984759, 0.34984759, 0., 0.],
                            [0., 0.54848033, 0., 0., 0.54848033, 0.54848033, 0., 0., 0.31226271, 0., 0., 0.],
                            [0., 0., 0.86903011, 0., 0., 0., 0., 0., 0., 0.49475921, 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0.27683498, 0.87754612, 0., 0., 0.27683498, 0.27683498]])
    documents = np.asarray(['it technolog advanc situat',
                            "mari don't like situat",
                            'technolog great',
                            'yes sir sir that question'])
    terms = np.asarray(['advanc', 'don', 'great', 'it', 'like', 'mari', 'question', 'sir', 'situat',
                        'technolog', 'that', 'yes'])
    dt_matrix = pd.DataFrame(dt_matrix, index=documents, columns=terms)
    n_components = 4
    return_doc_vectors = True
    result_1, result_2 = train_vectors_lsa(dt_matrix, n_components=n_components,
                                           return_doc_vectors=return_doc_vectors)
    print(result_1, "\n", result_2, "\n")
    info(result_1)
    info(result_2)


def load_term_vectors_glove(file_name, target_terms, parser_config=None, new_reduce_dict=False, verbose=False):
    """Loads pre-trained GloVe term vectors from file. If option new_reduce_dict=True, load full dictionary and
    reduce it to the passed target_terms, save reduced dict to .npy file. Option to use parser_config to parse
    dictionary keys, but this currently results in multiple vectors being returned for the same stemmed word."""
    # Implementation checked 28 June.
    if not new_reduce_dict:
        vector_dict = np.load(file_name).item()
    else:
        print("Creating GloVe vector-dictionary of relevant terms from full vector file...")
        # Load full pre-trained GloVe vector dictionary and extract relevant term vectors.
        with open(file_name, 'r') as file:
            vectors_full = pd.read_table(file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
        if verbose:
            print("Full GloVe vector file loaded as Pandas DataFrame.")
        # Parse keys of the GloVe vectors with the same parser configuration used on the corpus.
        vectors_full.index = pd.Series(vectors_full.index).replace(np.nan, 'nan')
        # TODO: parsing keys results in multiple vectors being returned for the same term
        if parser_config is not None:
            vectors_full.index = pd.Series(parse_text(vectors_full.index.values, stemmer=parser_config['stemmer'],
                                                      lower=parser_config['lower'], verbose=False,
                                                      return_config=False, remove_stop_words=False,
                                                      ignore_chars=''))
        vector_dict = {}
        ctr = 0
        for term in target_terms:
            try:
                vector_dict[term] = vectors_full.loc[term]
            except KeyError:
                continue  # deal with out of vocabulary words in term_vectors_from_dict(...)
            ctr += 1
            if verbose and ctr % 200 == 0:
                print("Creating GloVe vector dictionary of relevant terms...", ctr / len(target_terms) * 100, "%",
                      end="\r")
        np.save(file_name[:-4] + '_reduced.npy', vector_dict)
    return vector_dict


def test_ltvg():
    file_name = 'glove-pre-trained/glove.6B.50d.txt'
    target_terms = np.asarray(['advanc', 'don', 'great', 'it', 'like', 'mari', 'question', 'sir', 'situat',
                               'technolog', 'that', 'yes'])
    parser_config = {'stemmer': 'porter2', 'lower': True, 'remove_stop_words': True,
                     'ignore_chars': '.,:;"!?-/()[]{}0123456789'}
    parser_config = None
    new_reduce_dict = True
    verbose = True
    result = load_term_vectors_glove(file_name, target_terms, parser_config=parser_config,
                                     new_reduce_dict=new_reduce_dict, verbose=verbose)
    print(result, "\n")
    info(result)


def train_vectors_glove(tt_dict, n_components=300, alpha=0.75, x_max=100.0, step_size=0.05, n_epochs=25,
                        batch_size=64, workers=2, verbose=False):
    """Trains vector dictionary from the passed term-term dictionary with the passed hyperparameters.
    Glove.init()
    cooccurrence dict<int, dict<int, float>> : the co-occurence matrix
    alpha float : (default 0.75) hyperparameter for controlling the exponent for normalized co-occurrence counts.
    x_max float : (default 100.0) hyperparameter for controlling smoothing for common items in co-occurrence matrix.
    d int : (default 50) how many embedding dimensions for learnt vectors
    seed int : (default 1234) the random seed
    Glove.train()
    step_size float : the learning rate for the model
    n_epochs int : the number of iterations over the full dataset
    workers int : number of worker threads used for training
    batch_size int : how many examples should each thread receive (controls the size of the job queue)"""
    # Implementation checked 30 June.
    model = glove.Glove(tt_dict, d=n_components, alpha=alpha, x_max=x_max)
    for epoch in range(n_epochs):
        error = model.train(step_size=step_size, batch_size=batch_size, workers=workers)
        if verbose:
            print("GloVe training epoch %d, error %.3f" % (epoch + 1, error), flush=True)
    vector_matrix = model.W
    vector_dict = {ix: list(vector_matrix[ix]) for ix in tt_dict.keys()}
    return vector_dict


def test_tvg():
    tt_dict = {0: {0: 1, 2: 1, 5: 1, 6: 1}, 1: {1: 1, 6: 1}, 2: {0: 1, 2: 1, 5: 1, 6: 1},
               3: {3: 1, 4: 2, 7: 1, 8: 1}, 4: {3: 2, 4: 4, 7: 2, 8: 2}, 5: {0: 1, 2: 1, 5: 1, 6: 1},
               6: {0: 1, 1: 1, 2: 1, 5: 1, 6: 2}, 7: {3: 1, 4: 2, 7: 1, 8: 1}, 8: {3: 1, 4: 2, 7: 1, 8: 1}}
    n_components = 4
    alpha = 0.75
    x_max = 100.0
    step_size = 0.05
    n_epochs = 25
    batch_size = 1
    workers = 1
    verbose = True
    result = train_vectors_glove(tt_dict, n_components=n_components, alpha=alpha, x_max=x_max,
                                 step_size=step_size, n_epochs=n_epochs, batch_size=batch_size, workers=workers,
                                 verbose=verbose)
    print(result, "\n")
    info(result)


def aggregate_item_similarity(dt_matrix, term_vectors, n_similarities=2, verbose=False):
    """Computes item similarities from terms in vector space. To aggregate term cosine similarity to item
    similarity, the average similarity of the two most similar terms between each item pair is taken. This is
    the same concept as established by Larsen and Bong 2016 for aggregating construct similarity."""
    # TODO: implementation is completely agnostic to dt_matrix processing, e.g. normalizing -> try weighted avg
    # Implementation checked 28 June.
    # Compute cosine term similarity as matrix.
    term_similarity = np.asarray(np.asmatrix(term_vectors) * np.asmatrix(term_vectors).T)
    if verbose:
        print("Cosine similarity of terms computed. Number unique term pairs =",
              np.count_nonzero(np.triu(term_similarity, 1)))

    # Aggregate item similarity from term similarities.
    items = dt_matrix.index.values
    dt_matrix = np.asarray(dt_matrix)
    item_similarity = np.zeros([len(dt_matrix), len(dt_matrix)])
    n_fields = (len(item_similarity) ** 2 - len(item_similarity)) / 2  # n fields in upper triu for print
    ctr = 0  # counter for print
    ctr_one = 0  # counter for item-relationships with only one non-zero term similarity
    ctr_none = 0  # counter for item-relationships with no non-zero term similarity
    for ind_1 in range(len(dt_matrix) - 1):  # rows
        for ind_2 in range(ind_1 + 1, len(dt_matrix)):  # columns
            # Implementation checked manually, excluding exception handling.
            # Get term similarities between the items.
            term_indices_1 = np.where(dt_matrix[ind_1] != 0)[0]
            term_indices_2 = np.where(dt_matrix[ind_2] != 0)[0]
            term_indices_all = []
            for i1 in term_indices_1:
                term_indices_all += [(i1, i2) for i2 in term_indices_2]
            term_sim_sub = [term_similarity[i] for i in term_indices_all]
            try:  # Deals with zero vectors caused by out of vocabulary words.
                # Compute item similarity from average of n highest term similarities.
                sim_avg = np.average(np.sort(term_sim_sub, axis=None)[-np.max([n_similarities, 2]):])
            except ValueError:
                if np.count_nonzero(term_sim_sub) != 0:
                    sim_avg = np.sort(term_sim_sub, axis=None)[-1]
                    ctr_one += 1
                else:
                    sim_avg = 0
                    ctr_none += 1
            item_similarity[ind_1, ind_2] = sim_avg
            ctr += 1
            if verbose and ctr % 60000 == 0:
                print("Aggregating term to item similarity...", ctr / n_fields * 100, "%", end='\r')
    if verbose:
        print("Number of item-relationships with only one non-zero term similarity due to OOV:", ctr_one)
        print("Number of item-relationships with no non-zero term similarity due to OOV:", ctr_none)
    # Mirror lower triangular and fill diagonal of the matrix.
    item_similarity = np.add(item_similarity, item_similarity.T)
    np.fill_diagonal(item_similarity, 1)
    item_similarity = pd.DataFrame(item_similarity, index=items, columns=items)
    return item_similarity


def test_ais():
    dt_matrix = np.asarray([[0.61449708, 0., 0., 0.61449708, 0., 0., 0., 0., 0.34984759, 0.34984759, 0., 0.],
                            [0., 0.54848033, 0., 0., 0.54848033, 0.54848033, 0., 0., 0.31226271, 0., 0., 0.],
                            [0., 0., 0.86903011, 0., 0., 0., 0., 0., 0., 0.49475921, 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0.27683498, 0.87754612, 0., 0., 0.27683498, 0.27683498]])
    items = np.asarray(['it technolog advanc situat',
                        "mari don't like situat",
                        'technolog great',
                        'yes sir sir that question'])
    terms = np.asarray(['advanc', 'don', 'great', 'it', 'like', 'mari', 'question', 'sir', 'situat',
                        'technolog', 'that', 'yes'])
    dt_matrix = pd.DataFrame(dt_matrix, index=items, columns=terms)
    vector_dict = {'advanc': [0.39588465221557745, -1.2575977953455109e-08, 1.249000902703301e-16, 0.48723035562135314],
                   'don': [0.18859491626619804, 0.46382576479578286, -5.84601811404184e-16, -0.232110931026861],
                   'great': [0.47345367495179425, -0.46382576898144157, -4.579669976578771e-16, -0.5826974842832673],
                   'it': [0.39588465221557745, -1.2575978064477411e-08, -6.938893903907228e-18, 0.4872303556213531],
                   'like': [0.18859491626619812, 0.46382576479578286, -5.689893001203927e-16, -0.23211093102686112],
                   'mari': [0.18859491626619812, 0.46382576479578286, -5.689893001203927e-16, -0.23211093102686112],
                   'question': [1.7607443281164592e-16, 1.3183898417423734e-16, 0.27683497845223565,
                                -1.3183898417423734e-16],
                   'sir': [5.551115123125783e-16, 3.677613769070831e-16, 0.877546115093703, -5.689893001203927e-16],
                   'situat': [0.33275791820247796, 0.2640668742960114, -2.706168622523819e-16, 0.1452454665254026],
                   'technolog': [0.49493468508898575, -0.2640668886156101, -2.706168622523819e-16,
                                 -0.05435167411847225],
                   'that': [1.7607443281164592e-16, 1.3183898417423734e-16, 0.27683497845223565,
                            -1.3183898417423734e-16],
                   'yes': [1.7607443281164592e-16, 1.3183898417423734e-16, 0.27683497845223565,
                           -1.3183898417423734e-16]}
    term_vectors = term_vectors_from_dict(vector_dict, terms)
    n_similarities = 2
    verbose = True
    result = aggregate_item_similarity(dt_matrix, term_vectors, n_similarities=n_similarities, verbose=verbose)
    print(result, "\n")
    info(result)


def aggregate_construct_similarity(item_similarity, gold_items, variable_ids, n_similarities=2, verbose=False):
    """Computes construct similarities from items in vector space. To aggregate item cosine similarity to construct
    similarity, the average similarity of the two most similar items between each construct pair is taken, as
    established by Larsen and Bong 2016. Creates upper triangular with zero diagonal for efficiency."""
    # Implementation checked 28 June. There might be a problem in the indices to fully fill triu, but seems solved.
    item_similarity = np.asarray(item_similarity)
    variable_ids = np.sort(variable_ids)
    construct_similarity = np.zeros([len(variable_ids), len(variable_ids)])
    n_fields = (len(construct_similarity) ** 2 - len(construct_similarity)) / 2  # n fields in upper triu for print
    ctr = 0  # counter for print
    for ind_1 in range(len(variable_ids) - 1):  # rows
        for ind_2 in range(ind_1 + 1, len(variable_ids)):  # columns
            # Implementation checked manually.
            # Get item similarities between the constructs.
            item_indices_1 = np.where(gold_items['VariableId'] == variable_ids[ind_1])[0]
            item_indices_2 = np.where(gold_items['VariableId'] == variable_ids[ind_2])[0]
            # Combine item-indices so they fill the upper triangular of the construct similarity matrix.
            item_indices_all = []
            for i1 in item_indices_1:
                item_indices_all += [(i1, i2) for i2 in item_indices_2]
            item_sim_sub = [item_similarity[i] for i in item_indices_all]
            # Compute construct similarity from average of n highest item similarities.
            sim_avg = np.average(np.sort(item_sim_sub, axis=None)[-np.max([n_similarities, 2]):])
            construct_similarity[ind_1, ind_2] = sim_avg
            ctr += 1
            if verbose and ctr % 5000 == 0:
                print("Aggregating item to construct similarity...", ctr / n_fields * 100, "%", end='\r')
    construct_similarity = pd.DataFrame(construct_similarity, index=variable_ids, columns=variable_ids)
    return construct_similarity


def test_acs():
    item_similarity = np.asarray([[1.00000000e+00, 8.59243068e-01, 8.90522750e-01, 2.30422117e-16],
                                  [8.59243068e-01, 1.00000000e+00, 1.81708876e-01, -5.31647944e-18],
                                  [8.90522750e-01, 1.81708876e-01, 1.00000000e+00, -1.50979114e-17],
                                  [2.30422117e-16, -5.31647944e-18, -1.50979114e-17, 1.00000000e+00]])
    items = np.asarray(['it technolog advanc situat',
                        "mari don't like situat",
                        'technolog great',
                        'yes sir sir that question'])
    item_similarity = pd.DataFrame(item_similarity, index=items, columns=items)
    gold_items = pd.DataFrame([[1, 1],
                               [3, 1],
                               [6, 4],
                               [7, 9]], columns=['ItemId', 'VariableId'])
    variable_ids = [4, 1, 9]
    n_similarities = 2
    verbose = True
    result = aggregate_construct_similarity(item_similarity, gold_items=gold_items, variable_ids=variable_ids,
                                            n_similarities=n_similarities, verbose=verbose)
    print(result, "\n")
    info(result)


def evaluate(construct_similarity, construct_identity_gold):
    """Evaluates construct similarity matrix against the Larsen and Bong 2016 gold standard in matrix form."""
    # TODO: adjust to change in method recreate_construct_identity_gold(...)
    # Implementation checked 30 June.
    # Unwrap upper triangular of similarity and identity matrix, excluding diagonal.
    # Calculate Receiver Operating Characteristic (ROC) curve.
    construct_similarity = np.asarray(construct_similarity)
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


def test_e():
    variable_ids = [1, 2, 4, 9]
    construct_similarity = pd.DataFrame([[0.00000000e+00, 8.59243068e-01, 8.90522750e-01, 2.30422117e-16],
                                         [0.00000000e+00, 0.00000000e+00, 1.81708876e-01, -5.31647944e-18],
                                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.50979114e-17],
                                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                        index=variable_ids, columns=variable_ids)
    construct_identity_gold = pd.DataFrame([[1, 1, 1, 0],
                                            [1, 1, 1, 1],
                                            [1, 1, 1, 0],
                                            [0, 1, 0, 1]],
                                           index=variable_ids, columns=variable_ids)
    result_1, result_2, result_3 = evaluate(construct_similarity, construct_identity_gold)
    print(result_1, "\n", result_2, "\n", result_3, "\n")
    info(result_1)
    info(result_2)


# Define central parameters.
prototype = True
stemmer = 'porter2'
ignore_chars = '''.,:;"!?-/()[]{}0123456789'''
dtm_processing = 'tfidf_l2'
use_doc_vectors_lsa = False

# Load data.
print("Loading data...")
gold_items, pool_ids, variable_ids, construct_identity_gold, funk_papers = load_data(prototype=prototype,
                                                                                     verbose=True)

# Parse texts.
print("Parsing texts...")
corpus_items = parse_text(np.asarray(gold_items['Text']), stemmer=stemmer, lower=True,
                          remove_stop_words=True, return_config=False,
                          ignore_chars=ignore_chars, verbose=True)
# corpus_abstracts = parse_text(np.asarray(funk_papers['Abstract']), stemmer=stemmer, lower=True,
#                               remove_stop_words=True, return_config=False,
#                               ignore_chars=ignore_chars, verbose=True)

# Create document-term matrices and term-term dictionary.
print("Creating document-term matrices (docs x terms)...")
dtm_items, terms_items = document_term_cooccurrence(corpus_items, processing=dtm_processing)
# dtm_abstracts, terms_abstracts = create_dt_matrix(corpus_abstracts, processing=dtm_processing)
ttd_items, dict_term_ix, dict_ix_term = term_term_cooccurrence(dtm_items)

# Compute construct similarity matrix with LSA.
print("Computing construct similarity matrix with LSA...")
vector_dict_lsa, item_vectors_lsa = train_vectors_lsa(dtm_items, n_components=300, return_doc_vectors=True)
if use_doc_vectors_lsa:
    item_similarity_lsa = pd.DataFrame(np.asmatrix(item_vectors_lsa) * np.asmatrix(item_vectors_lsa).T,
                                       index=gold_items, columns=gold_items)
else:
    term_vectors_lsa = term_vectors_from_dict(vector_dict_lsa, terms_items, normalize=True, verbose=True)
    item_similarity_lsa = aggregate_item_similarity(dtm_items, term_vectors_lsa, n_similarities=2, verbose=True)
construct_similarity_lsa = aggregate_construct_similarity(item_similarity_lsa, gold_items, variable_ids,
                                                          n_similarities=2, verbose=True)

# Compute construct similarity matrix with pre-trained GloVe.
print("Computing construct similarity matrix with pre-trained GloVe...")
vector_dict_glove = load_term_vectors_glove(file_name='glove-pre-trained/glove.6B.300d.txt',
                                            target_terms=terms_items, parser_config=None,
                                            new_reduce_dict=True, verbose=True)
term_vectors_glove = term_vectors_from_dict(vector_dict_glove, terms_items, normalize=True, verbose=True)
item_similarity_glove = aggregate_item_similarity(dtm_items, term_vectors_glove, n_similarities=2, verbose=True)
construct_similarity_glove = aggregate_construct_similarity(item_similarity_glove, gold_items, variable_ids,
                                                            n_similarities=2, verbose=True)

# Compute construct similarity matrix with self-trained GloVe.
print("Computing construct similarity matrix with self-trained GloVe...")
pass

# Evaluate models.
print("Evaluating performance...")
fpr_lsa, tpr_lsa, roc_auc_lsa = evaluate(construct_similarity_lsa, construct_identity_gold)
print("ROC AUC LSA =", roc_auc_lsa)
fpr_glove, tpr_glove, roc_auc_glove = evaluate(construct_similarity_glove, construct_identity_gold)
print("ROC AUC GloVe =", roc_auc_glove)

# Plot ROC curves.
plt.figure()
plt.grid(True)
plt.plot(fpr_lsa, tpr_lsa)
plt.plot(fpr_glove, tpr_glove)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(["LSA", "GloVe"])
plt.show()


# TODO: -----------------------------------------
# TODO: ---------------- ARCHIVE ----------------


def items_vector_average_glove(dtm_identifier, vector_dict, denominator=None):
    # TODO: completely out of date
    """Translate items into pre-trained GloVe vector space and returns the matrix of item vectors. Sums term
    vectors of terms in item with passed document-term matrix as weighting factor and divides vector according
    to passed denominator mode."""
    # Translate items into GloVe vectors.
    # Implementation checked manually, but there are multiple ways of doing this.
    dt_matrix = dtm_items[dtm_identifier]
    item_vectors = np.zeros([len(dt_matrix), len(next(iter(vector_dict.values())))])
    for row_ind in range(len(dt_matrix)):
        ctr_oov = 0
        value_oov = 0
        for col_ind in range(len(dt_matrix[0])):
            if np.nonzero(dt_matrix[row_ind, col_ind]):
                try:
                    item_vectors[row_ind] = np.add(item_vectors[row_ind],
                                                   np.asarray(vector_dict[terms_items[col_ind]])
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
        if row_ind % 250 == 0:
            print("Translating items into GloVe vectors.", (row_ind + 1) / len(dt_matrix) * 100, "%", end="\r")
    return item_vectors


def term_term_cooccurrence_scratch(corpus):
    """Creates sparse term-term cooccurrence dictionary from passed corpus. Compared to using the dot product
    of the dt_matrix, this implementation has the disadvantage of being harder to normalize."""
    # Implementation checked 30 June.
    # Index terms in corpus, both as index -> term and as term -> index to translate in both directions.
    s = ' '
    terms = np.unique(s.join(corpus).split())
    dict_ix_term = {i: terms[i] for i in range(len(terms))}
    dict_term_ix = {v: k for k, v in dict_ix_term.items()}
    # Translate corpus to indices.
    corpus = np.asarray([[dict_term_ix[term] for term in corpus[i].split()] for i in range(len(corpus))])
    # Build the sparse term co-occurrence dictionary.
    tt_dict = {i: {} for i in range(len(terms))}
    for paragraph in corpus:
        for ix_center_term in range(len(paragraph)):
            for window_term in paragraph[:ix_center_term] + paragraph[ix_center_term + 1:]:
                try:
                    tt_dict[paragraph[ix_center_term]][window_term] += 1
                except KeyError:
                    tt_dict[paragraph[ix_center_term]][window_term] = 1
    return tt_dict, dict_term_ix, dict_ix_term


def test_ttco():
    corpus = np.asarray(['it technolog advanc situat',
                         "mari don't like situat",
                         'technolog great',
                         'yes sir sir that question'])
    processing = 'count'
    result_1, result_2, result_3 = term_term_cooccurrence(corpus, processing=processing)
    print(result_1, "\n", result_2, "\n", result_3, "\n")
    info(result_1)
    info(result_2)
    info(result_3)


# Convert similarity matrices to Pandas data frames with labelling.
construct_similarity_lsa = pd.DataFrame(construct_similarity_lsa, index=variable_ids, columns=variable_ids)
construct_similarity_glove = pd.DataFrame(construct_similarity_glove, index=variable_ids, columns=variable_ids)

# NOTES: Load the results from 'GloVe_search_results.npy'. Best results around 0.63.
# Parameter search for GloVe document projection by weighted item vectors.
# Uses old item_vectors_glove(...) function.
denominator_options = [None, 'sum', 'sum_value-oov', 'norm', 'norm_value-oov']
grid = [[mat, den] for mat in dtm_items for den in denominator_options]
# grid = [['dt_matrix', None], ['dt_matrix', 'sum']]
glove_results = []
ctr = 0
print("Performing grid search on GloVe.")
for dtm_identifier, denominator in grid:
    item_vectors = items_vector_average_glove(dtm_identifier, denominator=denominator)
construct_similarity_glove = np.nan_to_num(aggregate_construct_similarity(item_vectors))
fpr_glove, tpr_glove, roc_auc_glove = evaluate(construct_similarity_glove)
glove_results.append([dtm_identifier, denominator, roc_auc_glove])
ctr += 1
print("New GloVe search result:", dtm_identifier, denominator, roc_auc_glove)
print("Grid search on GloVe.", ctr / len(grid) * 100, "%\n")
np.save('GloVe_search_results.npy', np.asarray(glove_results))
glove_results = np.load('GloVe_search_results.npy')
