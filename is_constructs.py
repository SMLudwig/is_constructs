from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from scipy.stats.stats import pearsonr

from stemming.porter2 import stem as stem_porter2
from stemming.paicehusk import stem as stem_paicehusk

import numpy as np
import pandas as pd
import editdistance
import glove
import csv
import os.path
import warnings
import matplotlib.pyplot as plt


def info(var):
    if isinstance(var, np.ndarray):
        print("Type:", type(var), "\nShape:", np.shape(var))
    else:
        print("Type:", type(var), "\nLength:", len(var))


def recreate_construct_identity_gold(gold_standard, pool_ids, full_var_ids=None):
    """Translates the gold standard by Larsen and Bong 2016 into a binary construct identity matrix.
    Pass full_var_ids if not prototyping, since not all variable ids are present in pools."""
    # Implementation checked 28 June.
    if full_var_ids is not None:
        variable_ids = full_var_ids
    else:
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
    full_var_ids = None
    result = recreate_construct_identity_gold(gold_standard, pool_ids, full_var_ids=full_var_ids)
    print(result, "\n")
    info(result)


def load_data(prototype=False, max_editdistance=1, verbose=False):
    """Load data. construct_authors are indexed by the matching construct ID in Funk's dataset. Use funk2gold to
    translate the IDs to matching gold IDs."""
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
            construct_identity_gold = pd.read_pickle('construct_identity_gold_prototype.df')
        except FileNotFoundError:
            if verbose:
                print("No construct identity gold matrix file found. Creating new file...")
            construct_identity_gold = recreate_construct_identity_gold(gold_standard, pool_ids)
            construct_identity_gold.to_pickle('construct_identity_gold_prototype.df')
    else:
        try:
            construct_identity_gold = pd.read_pickle('construct_identity_gold.df')
        except FileNotFoundError:
            if verbose:
                print("No construct identity gold matrix file found. Creating new file...")
            construct_identity_gold = recreate_construct_identity_gold(gold_standard, pool_ids,
                                                                       full_var_ids=variable_ids)
            construct_identity_gold.to_pickle('construct_identity_gold.df')

    # Load Funk's data on papers and constructs.
    file = r'datasetFunk/FunkPapers.xlsx'
    funk_papers = pd.read_excel(file)
    file = r'datasetFunk/FunkConstructs.xlsx'
    funk_constructs = pd.read_excel(file)
    # Get unique construct IDs from Larsen's and Funk's dataset.
    gold_construct_ids = np.unique(gold_items['VariableId'])
    funk_construct_ids = np.unique(funk_constructs['ConstructID'])

    # Calculate construct distances between constructs in Larsen's and Funk's datasets.
    # TODO: unit testing
    # TODO: painfully slow, probably since it has to search in DataFrames every iteration.
    try:
        construct_distances = pd.read_pickle('construct_editdistances.df')
    except FileNotFoundError:
        print("No construct editdistance file found. Creating new file...")
        if prototype:
            warnings.warn("Computing distances in prototype mode. Remember to delete file for full mode.")
        # Remove ignore characters from construct names to make them more comparable. Add parsed names to DataFrames.
        ignore_chars = '''.,:;"'!?-/()[]{}&%0123456789'''
        gold_names_parsed = [' '.join(name.translate({ord(c): ' ' for c in ignore_chars}).lower().split())
                             for name in gold_items['VariableName']]
        gold_items['VariableNameParse'] = gold_names_parsed
        funk_names_parsed = [' '.join(name.translate({ord(c): ' ' for c in ignore_chars}).lower().split())
                             for name in funk_constructs['ConstructName']]
        funk_constructs['ConstructNameParse'] = funk_names_parsed

        # TODO: could create two dicts ID: ConstructName to speed this up.
        # Evaluate distances and fill new DataFrame.
        construct_distances = pd.DataFrame(np.zeros([len(gold_construct_ids), len(funk_construct_ids)]),
                                           index=gold_construct_ids, columns=funk_construct_ids)
        ctr = 0
        for gold_id in gold_construct_ids:
            for funk_id in funk_construct_ids:
                gold_name = gold_items.loc[gold_items['VariableId'] == gold_id, 'VariableNameParse'].iloc[0]
                funk_name = funk_constructs.loc[funk_constructs['ConstructID'] == funk_id, 'ConstructNameParse'].iloc[0]
                distance = editdistance.eval(gold_name, funk_name)
                construct_distances[funk_id][gold_id] = distance  # DataFrames access columns first, then rows.
            ctr += 1
            if verbose and ctr % 30 == 0:
                print("Relating gold constructs to Funk's constructs:", ctr / len(gold_construct_ids) * 100, "%",
                      flush=True)
        construct_distances.to_pickle('construct_editdistances.df')

    # Create construct ID translation dictionary between Larsen' and Funk's dataset. Simply uses the first match.
    # TODO: deal with multiple matches.
    funk2gold = {}
    ctr = 0
    for funk_id in funk_construct_ids:
        for gold_id in gold_construct_ids:
            # Check whether the gold_id has already been linked to a funk_id. This can happen with multiple matches.
            if gold_id in funk2gold.values():
                continue
            try:
                if construct_distances[funk_id][gold_id] <= max_editdistance:
                    funk2gold[funk_id] = gold_id
                    # Break to go to the next funk_id, so that every id gets only matched once.
                    break
            except KeyError:
                print("Check whether editdistances were created on prototype or full dataset.")
                raise
        ctr += 1
        if verbose and ctr % 200 == 0:
            print("Creating construct ID translation dictionary:", ctr / len(funk_construct_ids) * 100, "%",
                  flush=True)
    if verbose:
        print("Related", len(funk_construct_ids), "Funk constructs to", len(gold_construct_ids), "gold constructs.")
        print(len(funk2gold), "matches found with Levenshtein distance <=", max_editdistance, "\n")
    gold2funk = {g: f for f, g in funk2gold.items()}

    # Get authors of the constructs in Funk's dataset.
    # Implementation checked 4 July.
    construct_authors = {}
    for construct_id in funk_construct_ids:
        # Get PaperID related to a specific ConstructID
        paper_id = funk_constructs.loc[funk_constructs['ConstructID'].isin([construct_id])]['PaperID']
        # Get Author of specific PaperID
        authors = funk_papers.loc[funk_papers['PaperID'].isin([paper_id])]['Author']
        construct_authors[construct_id] = np.asarray(authors)[0]

    return gold_items, pool_ids, variable_ids, construct_identity_gold, funk_papers, funk_constructs, \
           construct_authors, construct_distances, funk2gold, gold2funk


def test_ld():
    prototype = False
    verbose = True
    gold_items, pool_ids, variable_ids, construct_identity_gold, funk_papers, funk_constructs, \
    construct_authors, construct_distances, funk2gold, gold2funk = load_data(prototype=prototype, verbose=verbose)

    for i in np.unique(gold_items['VariableId'].head(200)):
        try:
            print(gold_items.loc[gold_items['VariableId'].isin([i]), 'VariableName'].iloc[0], ":",
                  construct_authors[gold2funk[i]])
        except KeyError:
            pass


def parse_text(documents, stemmer=None, lower=True, remove_stop_words=True,
               return_config=False, ignore_chars='''.,:;"'!?-/()[]{}&%0123456789''', verbose=False):
    """Parses text with options for removing specified characters, removing stop-words, converting to lower-case
    and stemming (https://pypi.org/project/stemming/1.0/). Available stemming algorithms are 'porter2' and
    'paicehusk'. Paice/Husk seems prone to over-stemming."""
    # TODO: still returns empty '' strings. Also returns "'s".
    # Implementation checked 28 June.
    parsed_docs = []
    error_words = []
    for i in range(len(documents)):
        assert isinstance(documents[i], str), "Document not a string:" + str(documents[i])
        if ignore_chars != '':
            documents[i] = ' '.join(documents[i].translate({ord(c): ' ' for c in ignore_chars}).split())
        if lower:
            documents[i] = documents[i].lower()
        parsed_docs.append('')
        for word in documents[i].split():
            if remove_stop_words and word in stop_words.ENGLISH_STOP_WORDS:
                continue
            # TODO: what does this do?
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
        parsed_docs[i] = ' '.join(parsed_docs[i].split())  # remove excess white space
    if verbose and error_words:
        print("ValueError occurred when stemming the following words:", list(set(error_words)), "\n")
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
    ignore_chars = '''.,:;"'!?-/()[]{}0123456789'''
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
    dt_matrix = count_vectorizer.fit_transform(corpus).toarray()
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


def term_term_cooccurrence(dt_matrix, verbose=False):
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
    ctr = 0
    for i in terms_ix:
        for k in terms_ix:
            if tt_matrix[i][k] != 0:
                try:
                    tt_dict[i][k] += float(tt_matrix[i][k])
                except KeyError:
                    tt_dict[i][k] = float(tt_matrix[i][k])
        ctr += 1
        if verbose and ctr % 300 == 0:
            print("Building term-term cooccurrence dictionary:", ctr / len(terms_ix) * 100, "%", flush=True)
    # if verbose:
    #     print("\n")
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
    verbose = True
    result_1, result_2, result_3 = term_term_cooccurrence(dt_matrix, verbose=verbose)
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
    assert len(dt_matrix) >= n_components, \
        "n docs must be >= n components. " + str(len(dt_matrix)) + " < " + str(n_components)
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


def load_term_vectors_glove(file_name, target_terms, new_reduce_dict=False,
                            ignore_chars='''.,:;"'!?_-/()[]{}&%0123456789''', verbose=False):
    """Loads pre-trained GloVe term vectors from file. If option new_reduce_dict=True, load full dictionary and
    reduce it to the passed target_terms, save reduced dict to .npy file."""
    # Implementation unchecked.
    if not new_reduce_dict:
        vector_dict = np.load(file_name).item()
    else:
        print("Creating GloVe vector-dictionary of relevant terms from full vector file...")
        if not os.path.isfile(file_name[:-4] + '.h5'):
            print("No HDF5 file found. Creating new file...")
            # Convert full vector file to pandas hdf5 file.
            hdf = pd.HDFStore(file_name[:-4] + '.h5')
            for chunk in pd.read_table(file_name, chunksize=100000, sep=' ', index_col=0,
                                       quoting=csv.QUOTE_NONE):
                for i in chunk.index.values:
                    if i in ignore_chars:  # Skip ignore characters like '/', '.', ...
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # Some NaturalNameWarnings for names like 'and', 'in', ...
                        try:
                            hdf.put(i, chunk.loc[i])
                        except ValueError:
                            if verbose:
                                print("ValueError encountered for", i)
                            continue
            if verbose:
                print("Full GloVe vector file converted to pandas HDF5 file.")

        # TODO: load HDF5 or read properly from existing.

        vector_dict = {}
        ctr = 0
        for term in target_terms:
            try:
                vector_dict[term] = pd.read_hdf(file_name[:-4] + '.h5', key=term)
            except KeyError:
                continue  # deal with out of vocabulary words in term_vectors_from_dict(...)
            ctr += 1
            if verbose and ctr % 200 == 0:
                print("Creating GloVe vector dictionary of relevant terms...", ctr / len(target_terms) * 100, "%",
                      end="\r")
        np.save(file_name[:-4] + '_reduced.npy', vector_dict)
        hdf.close()
    return vector_dict


def test_ltvg():
    file_name = 'glove-pre-trained/glove.6B.50d.txt'
    target_terms = np.asarray(['advanc', 'don', 'great', 'it', 'like', 'mari', 'question', 'sir', 'situat',
                               'technolog', 'that', 'yes'])
    new_reduce_dict = True
    verbose = True
    result = load_term_vectors_glove(file_name, target_terms, new_reduce_dict=new_reduce_dict, verbose=verbose)
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
    epoch_loss = []
    for epoch in range(n_epochs):
        error = model.train(step_size=step_size, batch_size=batch_size, workers=workers)
        epoch_loss.append(error)
        if verbose:
            print("GloVe training epoch %d, error %.5f" % (epoch + 1, error), flush=True)
    # if verbose:
    #     print("\n")
    vector_matrix = model.W
    vector_dict = {ix: list(vector_matrix[ix]) for ix in tt_dict.keys()}
    return vector_dict, np.asarray(epoch_loss)


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


def vector_average(dt_matrix, term_vectors, weighting=False):
    """Compute vector average of term vectors to form item vectors. If weighting=True, weights are values in the
    dt_matrix, mean is computed by dividing weighted sum of the vectors by sum of the weights."""
    # Implementation checked 6 July.
    doc_vectors = pd.DataFrame(np.zeros([len(dt_matrix), len(term_vectors.iloc[0])]), index=dt_matrix.index.values)
    for i in range(len(dt_matrix)):
        doc_term_vectors = term_vectors.loc[dt_matrix.columns.values[dt_matrix.iloc[i] > 0]]
        if weighting:
            # Take weighted mean, dividing by sum of weights.
            weights = dt_matrix.iloc[i][dt_matrix.iloc[i] > 0]
            for w_ix in weights.index.values:
                doc_term_vectors.loc[w_ix] = doc_term_vectors.loc[w_ix] * weights.loc[w_ix]
            doc_vectors.iloc[i] = np.sum(np.asarray(doc_term_vectors), axis=0) / np.sum(weights)
        else:
            doc_vectors.iloc[i] = np.mean(np.asarray(doc_term_vectors), axis=0)
    return doc_vectors


def test_va():
    dt_matrix = np.asarray([[0.61449708, 0., 0., 0.61449708, 0., 0., 0., 0., 0.34984759, 0.34984759, 0., 0.],
                            [0., 0.54848033, 0., 0., 0.54848033, 0.54848033, 0., 0., 0.31226271, 0., 0., 0.],
                            [0., 0., 0.86903011, 0., 0., 0., 0., 0., 0., 0.49475921, 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0.27683498, 0.87754612, 0., 0., 0.27683498, 0.27683498]])
    items = np.asarray(['it technolog advanc situat',
                        "mari don't like situat",
                        'technolog great',
                        'yes sir sir that question'])
    terms = np.asarray(['advanc', "don't", 'great', 'it', 'like', 'mari', 'question', 'sir', 'situat',
                        'technolog', 'that', 'yes'])
    dt_matrix = pd.DataFrame(dt_matrix, index=items, columns=terms)
    vector_dict = {'advanc': [0.39588465221557745, -1.2575977953455109e-08, 1.249000902703301e-16, 0.48723035562135314],
                   "don't": [0.18859491626619804, 0.46382576479578286, -5.84601811404184e-16, -0.232110931026861],
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
    term_vectors = term_vectors_from_dict(vector_dict, terms, normalize=False)
    weighting = False
    result = vector_average(dt_matrix, term_vectors, weighting=weighting)
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
            if verbose and ctr % 100000 == 0:
                print("Aggregating term to item similarity...", ctr / n_fields * 100, "%", end='\r')
    if verbose:
        print("Number of item-relationships with only one non-zero term similarity due to OOV:", ctr_one)
        print("Number of item-relationships with no non-zero term similarity due to OOV:", ctr_none, "\n")
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


def aggregate_construct_similarity(constituent_similarity, gold_items, variable_ids, construct_authors=None,
                                   n_similarities=2, verbose=False):
    """Computes construct similarities from items or authors in vector space. To aggregate constituent
    cosine similarity to construct similarity, the average similarity of the two most similar constituents
    between each construct pair is taken, as established by Larsen and Bong 2016 with items.
    Creates upper triangular with zero diagonal for efficiency."""
    # Implementation checked 4 July.
    authors = constituent_similarity.index.values
    constituent_similarity = np.asarray(constituent_similarity)
    variable_ids = np.sort(variable_ids)
    construct_similarity = np.zeros([len(variable_ids), len(variable_ids)])
    n_fields = (len(construct_similarity) ** 2 - len(construct_similarity)) / 2  # n fields in upper triu for print
    ctr = 0  # counter for print
    for ind_1 in range(len(variable_ids) - 1):  # rows
        for ind_2 in range(ind_1 + 1, len(variable_ids)):  # columns
            if construct_authors is not None:
                # Get author similarity indices between constructs.
                try:
                    constit_ix_1 = np.where(authors == construct_authors[gold2funk[variable_ids[ind_1]]])[0]
                    constit_ix_2 = np.where(authors == construct_authors[gold2funk[variable_ids[ind_2]]])[0]
                except KeyError:
                    # TODO: deal with constructs that have unknown authors better
                    construct_similarity[ind_1, ind_2] = 0
                    break
            # Following implementation checked manually.
            else:
                # Get item similarity indices between the constructs.
                constit_ix_1 = np.where(gold_items['VariableId'] == variable_ids[ind_1])[0]
                constit_ix_2 = np.where(gold_items['VariableId'] == variable_ids[ind_2])[0]
            # Combine item-indices so they fill the upper triangular of the construct similarity matrix.
            item_indices_all = []
            for i1 in constit_ix_1:
                item_indices_all += [(i1, i2) for i2 in constit_ix_2]
            item_sim_sub = [constituent_similarity[i] for i in item_indices_all]
            # Compute construct similarity from average of n highest item similarities.
            sim_avg = np.average(np.sort(item_sim_sub, axis=None)[-np.max([n_similarities, 2]):])
            construct_similarity[ind_1, ind_2] = sim_avg
            ctr += 1
            if verbose and ctr % 10000 == 0:
                print("Aggregating constituent to construct similarity...", ctr / n_fields * 100, "%", end='\r')
    # Set nan values to 0. Origin unknown.
    construct_similarity = np.nan_to_num(construct_similarity)
    construct_similarity = pd.DataFrame(construct_similarity, index=variable_ids, columns=variable_ids)
    return construct_similarity


def test_acs():
    # Last test 28 June.
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
    # Implementation checked 4 July.
    # Unwrap upper triangular of similarity and identity matrix, excluding diagonal.
    # Calculate Receiver Operating Characteristic (ROC) curve.
    construct_similarity = np.asarray(construct_similarity)
    construct_identity_gold = np.asarray(construct_identity_gold)
    triu_indices = np.triu_indices(len(construct_similarity), k=1)
    try:
        construct_sim_flat = construct_similarity[triu_indices]
        construct_idn_gold_flat = construct_identity_gold[triu_indices]
    except IndexError:
        # Occurs when already flattened arrays are passed.
        construct_sim_flat = construct_similarity
        construct_idn_gold_flat = construct_identity_gold
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
prototype = False
stemmer = 'porter2'
ignore_chars = '''.,:;"'!?_-/()[]{}&%0123456789'''
dtm_processing = 'tfidf_l2'
glove_pretrained_filename = 'glove-pre-trained/glove.6B.300d.txt'
glove_new_reduce_dict = False
verbose = True

# Load data.
print("Loading data...")
gold_items, pool_ids, variable_ids, construct_identity_gold, funk_papers, funk_constructs, construct_authors, \
construct_editdistances, funk2gold, gold2funk = load_data(prototype=prototype, max_editdistance=1, verbose=verbose)
var_ids_authors = np.sort(list(gold2funk.keys()))
construct_identity_gold_authors = construct_identity_gold.loc[var_ids_authors, var_ids_authors]
triu_indices = np.triu_indices(len(var_ids_authors), k=1)

# Process corpus texts.
print("Parsing texts...")
corpus_items = parse_text(np.asarray(gold_items['Text']), stemmer=stemmer, lower=True,
                          remove_stop_words=True, return_config=False,
                          ignore_chars=ignore_chars, verbose=verbose)
# corpus_abstracts = parse_text(np.asarray(funk_papers['Abstract']), stemmer=stemmer, lower=True,
#                               remove_stop_words=True, return_config=False,
#                               ignore_chars=ignore_chars, verbose=True)
corpus_authors = np.unique(list(construct_authors.values()))  # Options: .asarray or .unique
# corpus_ authors = parse_text(np.unique(list(construct_authors.values())), stemmer=None, lower=True,
#                              remove_stop_words=False, return_config=False,
#                              ignore_chars=ignore_chars, verbose=True)

# Create document-term matrices and term-term dictionary.
print("Creating document-term matrices (docs x terms)...")
dtm_items, terms_items = document_term_cooccurrence(corpus_items, processing=dtm_processing)
# dtm_abstracts, terms_abstracts = document_term_cooccurrence(corpus_abstracts, processing=dtm_processing)
dtm_authors, terms_authors = document_term_cooccurrence(corpus_authors, processing=dtm_processing)
ttd_items, dict_term_ix_items, dict_ix_term_items = term_term_cooccurrence(dtm_items, verbose=verbose)
ttd_authors, dict_term_ix_authors, dict_ix_term_authors = term_term_cooccurrence(dtm_authors, verbose=verbose)

# Compute construct similarity matrix with LSA on item corpus.
print("Computing construct similarity matrix with LSA...")
use_doc_vectors_lsa = True
lsa_aggregation = False
vector_dict_lsa, item_vectors_lsa = train_vectors_lsa(dtm_items, n_components=300, return_doc_vectors=True)
if use_doc_vectors_lsa:
    # Use document-vectors.
    item_similarity_lsa = pd.DataFrame(np.asmatrix(item_vectors_lsa) * np.asmatrix(item_vectors_lsa).T,
                                       index=gold_items, columns=gold_items)
else:
    term_vectors_lsa = term_vectors_from_dict(vector_dict_lsa, terms_items, normalize=True, verbose=verbose)
    if lsa_aggregation:
        # Term to item vector aggregation.
        item_similarity_lsa = aggregate_item_similarity(dtm_items, term_vectors_lsa, n_similarities=2, verbose=verbose)
    else:
        # Term vector averaging.
        item_vectors_lsa_avg = vector_average(dtm_items, term_vectors_lsa, weighting=False)
        item_similarity_lsa = pd.DataFrame(np.asarray(
            np.asmatrix(item_vectors_lsa_avg) * np.asmatrix(item_vectors_lsa_avg).T),
            index=item_vectors_lsa_avg.index.values, columns=item_vectors_lsa_avg.index.values)
construct_similarity_lsa = aggregate_construct_similarity(item_similarity_lsa, gold_items, variable_ids,
                                                          n_similarities=2, verbose=verbose)
fpr_lsa, tpr_lsa, roc_auc_lsa = evaluate(construct_similarity_lsa, construct_identity_gold)
print("ROC AUC LSA =", roc_auc_lsa, "\n")

# Compare item vector and item similarity aggregation methods.
# TODO: include new result with weighted vector averaging
term_vectors_lsa = term_vectors_from_dict(vector_dict_lsa, terms_items, normalize=True, verbose=verbose)
item_vectors_lsa_dvec = item_vectors_lsa
item_vectors_lsa_avg = vector_average(dtm_items, term_vectors_lsa, weighting=False)
item_similarity_lsa_dvec = np.asarray(np.asmatrix(item_vectors_lsa_dvec) * np.asmatrix(item_vectors_lsa_dvec).T)
item_similarity_lsa_avg = np.asarray(np.asmatrix(item_vectors_lsa_avg) * np.asmatrix(item_vectors_lsa_avg).T)
item_similarity_lsa_agg = aggregate_item_similarity(dtm_items, term_vectors_lsa, n_similarities=2,
                                                    verbose=verbose)
item_similarity_methods = pd.DataFrame(
    np.asarray(np.asmatrix([np.asarray(item_similarity_lsa_dvec)[np.triu_indices(len(item_similarity_lsa), k=1)],
                            np.asarray(item_similarity_lsa_avg)[np.triu_indices(len(item_similarity_lsa), k=1)],
                            np.asarray(item_similarity_lsa_agg)[np.triu_indices(len(item_similarity_lsa), k=1)]]).T),
    columns=['LSA dvec', 'LSA avg', 'LSA agg'])
print("Pearson correlation item-vectors LSA from document-vectors and from term-vector average.")
print(pearsonr(np.asarray(item_vectors_lsa_dvec).flatten(), np.asarray(item_vectors_lsa_avg).flatten()))
print("Correlation table for item similarity methods.")
print(item_similarity_methods.corr())
plt.scatter(np.asarray(item_vectors_lsa_dvec).flatten(), np.asarray(item_vectors_lsa_avg).flatten())

# Compute construct similarity matrix with pre-trained GloVe on item corpus.
print("Computing construct similarity matrix with pre-trained GloVe...")
vector_dict_preglove = load_term_vectors_glove(file_name=glove_pretrained_filename,
                                               target_terms=terms_items, parser_config=None,
                                               new_reduce_dict=glove_new_reduce_dict, verbose=verbose)
term_vectors_preglove = term_vectors_from_dict(vector_dict_preglove, terms_items, normalize=True, verbose=verbose)
item_similarity_preglove = aggregate_item_similarity(dtm_items, term_vectors_preglove, n_similarities=2,
                                                     verbose=verbose)
construct_similarity_preglove = aggregate_construct_similarity(item_similarity_preglove, gold_items, variable_ids,
                                                               n_similarities=2, verbose=verbose)
fpr_preglove, tpr_preglove, roc_auc_preglove = evaluate(construct_similarity_preglove, construct_identity_gold)
print("ROC AUC pre-trained GloVe =", roc_auc_preglove, "\n")

# Perform grid search on GloVe self-trained on item corpus with unweighted vector average for speed.
# You can train GloVe with best parameters and item similarity aggregation instead of vector average afterwards.
try:
    glove_results = pd.read_csv('GloVe_search_results.csv', index_col=0).values.tolist()
except FileNotFoundError:
    glove_results = []
search_alpha = [0.55, 0.575, 0.625, 0.65]
search_x_max = [20, 30, 70, 80]
search_step_size = [0.001, 0.0025, 0.005, 0.0075]
search_n_epochs = [75]
search_weighting = [False]
search_grid = [[alpha, x_max, step_size, n_epochs, weighting] for alpha in search_alpha for x_max in search_x_max
               for step_size in search_step_size for n_epochs in search_n_epochs for weighting in search_weighting]
search_early_stopping = 0.99  # ROC AUC for early stopping of grid search.
ctr = 0
print("Performing grid search on GloVe self-trained on item corpus...\n")
for alpha, x_max, step_size, n_epochs, weighting in search_grid:
    try:
        print("alpha =", alpha, "x_max =", x_max, "step_size =", step_size,
              "n_epochs =", n_epochs, "weighting =", weighting)
        vector_dict_trglove, loss_glove_items = train_vectors_glove(ttd_items, n_components=300, alpha=alpha,
                                                                    x_max=x_max,
                                                                    step_size=step_size, n_epochs=n_epochs,
                                                                    batch_size=64,
                                                                    workers=2, verbose=verbose)  # Train vectors.
        if np.sum(np.isnan(loss_glove_items)) > 0:
            print("Encountered nan loss with following parameters:")
            print("alpha =", alpha, "x_max =", x_max, "step_size =", step_size,
                  "n_epochs =", n_epochs, "weighting =", weighting, "\n")
            continue
        vector_dict_trglove = {dict_ix_term_items[key]: value for key, value in
                               vector_dict_trglove.items()}  # Translate indices.
        term_vectors_trglove = term_vectors_from_dict(vector_dict_trglove, terms_items, normalize=True, verbose=verbose)
        item_vectors_trglove = vector_average(dtm_items, term_vectors_trglove, weighting=weighting)
        item_similarity_trglove = pd.DataFrame(np.asarray(
            np.asmatrix(item_vectors_trglove) * np.asmatrix(item_vectors_trglove).T),
            index=item_vectors_trglove.index.values, columns=item_vectors_trglove.index.values)
        construct_similarity_trglove = aggregate_construct_similarity(item_similarity_trglove, gold_items, variable_ids,
                                                                      n_similarities=2, verbose=verbose)
        fpr_trglove, tpr_trglove, roc_auc_trglove = evaluate(construct_similarity_trglove, construct_identity_gold)
        ctr += 1
        print("Result for GloVe with alpha =", alpha, "x_max =", x_max, "step_size =", step_size,
              "n_epochs =", n_epochs, "weighting =", weighting)
        print("ROC AUC =", roc_auc_trglove, "GloVe training loss =", loss_glove_items[-1], "\n")
        print("Grid search on GloVe.", ctr / len(search_grid) * 100, "%\n")
        glove_results.append([alpha, x_max, step_size, n_epochs, weighting, roc_auc_trglove, loss_glove_items[-1]])
        if roc_auc_trglove >= search_early_stopping:
            print("Early stopping: ROC AUC", roc_auc_trglove, ">=", search_early_stopping)
            break
    except:
        print("Encountered some error. Continuing search with next parameter set...\n")
        continue
print("Grid search results:")
glove_results = pd.DataFrame(np.asarray(glove_results), columns=['alpha', 'x_max', 'step_size', 'n_epochs',
                                                                 'weighting', 'roc_auc', 'training_loss'])
print(glove_results)
# Print best GloVe configuration.
glove_results_best = pd.DataFrame(
    np.asarray(glove_results)[np.where(np.asarray(glove_results)[:, -2] == np.max(np.asarray(glove_results)[:, -2]))],
    columns=['alpha', 'x_max', 'step_size', 'n_epochs', 'weighting', 'roc_auc', 'training_loss'])
print("Best result:")
print(glove_results_best, "\n")
# Save grid search results.
glove_results.to_csv('GloVe_search_results.csv')

# Plot GloVe grid search results.
if verbose:
    plt.figure(figsize=(10, 6))
    # Training alpha.
    x_plt = np.unique(glove_results['alpha'])
    y_plt = [np.mean(glove_results.loc[glove_results['alpha'].isin([x]), 'roc_auc']) for x in x_plt]
    e_plt = [np.std(glove_results.loc[glove_results['alpha'].isin([x]), 'roc_auc'], axis=0) for x in x_plt]
    plt.subplot(2, 3, 1)
    plt.errorbar(x_plt, y_plt, e_plt, fmt='o', capsize=4)
    plt.xlabel('alpha')
    plt.ylabel('mean roc_auc')
    # Training x_max.
    x_plt = np.unique(glove_results['x_max'])
    y_plt = [np.mean(glove_results.loc[glove_results['x_max'].isin([x]), 'roc_auc']) for x in x_plt]
    e_plt = [np.std(glove_results.loc[glove_results['x_max'].isin([x]), 'roc_auc'], axis=0) for x in x_plt]
    plt.subplot(2, 3, 2)
    plt.errorbar(x_plt, y_plt, e_plt, fmt='o', capsize=4)
    plt.xlabel('x_max')
    plt.ylabel('mean roc_auc')
    plt.title('GloVe prototype hyperparameter search')
    # Training step size.
    x_plt = np.unique(glove_results['step_size'])
    y_plt = [np.mean(glove_results.loc[glove_results['step_size'].isin([x]), 'roc_auc']) for x in x_plt]
    e_plt = [np.std(glove_results.loc[glove_results['step_size'].isin([x]), 'roc_auc'], axis=0) for x in x_plt]
    plt.subplot(2, 3, 3)
    plt.errorbar(x_plt, y_plt, e_plt, fmt='o', capsize=4)
    plt.xlabel('step_size')
    plt.ylabel('mean roc_auc')
    # Number of training epochs.
    x_plt = np.unique(glove_results['n_epochs'])
    y_plt = [np.mean(glove_results.loc[glove_results['n_epochs'].isin([x]), 'roc_auc']) for x in x_plt]
    e_plt = [np.std(glove_results.loc[glove_results['n_epochs'].isin([x]), 'roc_auc'], axis=0) for x in x_plt]
    plt.subplot(2, 3, 4)
    plt.errorbar(x_plt, y_plt, e_plt, fmt='o', capsize=4)
    plt.xlabel('n_epochs')
    plt.ylabel('mean roc_auc')
    # Weighting in vector averaging.
    x_plt = np.unique(glove_results['weighting'])
    y_plt = [np.mean(glove_results.loc[glove_results['weighting'].isin([x]), 'roc_auc']) for x in x_plt]
    e_plt = [np.std(glove_results.loc[glove_results['weighting'].isin([x]), 'roc_auc'], axis=0) for x in x_plt]
    plt.subplot(2, 3, 5)
    plt.bar(x_plt, y_plt, yerr=e_plt, capsize=4)
    plt.xlim(-0.5, 1.5)
    plt.xlabel('weighting')
    plt.ylabel('mean roc_auc')
    # Vector training loss.
    x_plt = np.unique(glove_results['training_loss'])
    y_plt = [np.mean(glove_results.loc[glove_results['training_loss'].isin([x]), 'roc_auc']) for x in x_plt]
    plt.subplot(2, 3, 6)
    plt.scatter(x_plt, y_plt)
    plt.xlabel('training_loss')
    plt.ylabel('mean roc_auc')
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show(block=False)

# Compute construct similarity matrix with self-trained GloVe on item corpus.
print("Computing construct similarity matrix with self-trained GloVe...")
glove_aggregation = True
vector_dict_trglove, loss_glove_items = train_vectors_glove(ttd_items, n_components=300, alpha=0.55, x_max=80.0,
                                                            step_size=0.0075, n_epochs=75, batch_size=64, workers=2,
                                                            verbose=verbose)  # Train vectors.
if verbose:
    pass
    # plt.figure()
    # plt.plot(range(len(loss_glove_items)), loss_glove_items)
    # plt.legend(["GloVe training loss"])
    # plt.show()
vector_dict_trglove = {dict_ix_term_items[key]: value for key, value in
                       vector_dict_trglove.items()}  # Translate indices.
term_vectors_trglove = term_vectors_from_dict(vector_dict_trglove, terms_items, normalize=True, verbose=verbose)
if glove_aggregation:
    item_similarity_trglove = aggregate_item_similarity(dtm_items, term_vectors_trglove, n_similarities=2,
                                                        verbose=verbose)
else:
    item_vectors_trglove = vector_average(dtm_items, term_vectors_trglove, weighting=False)
    # Compute item similarity and set negative values to 0.
    item_similarity_trglove = pd.DataFrame(np.asarray(
        np.asmatrix(item_vectors_trglove) * np.asmatrix(item_vectors_trglove).T).clip(min=0),
        index=item_vectors_trglove.index.values, columns=item_vectors_trglove.index.values)
construct_similarity_trglove = aggregate_construct_similarity(item_similarity_trglove, gold_items, variable_ids,
                                                              n_similarities=2, verbose=verbose)
fpr_trglove, tpr_trglove, roc_auc_trglove = evaluate(construct_similarity_trglove, construct_identity_gold)
print("ROC AUC self-trained GloVe =", roc_auc_trglove, "\n")

# Compute construct similarity based on normalized author co-occurrence matrix without creating a semantic space.
author_similarity = np.asarray(dtm_authors).dot(np.asarray(dtm_authors).T)
author_similarity = pd.DataFrame(author_similarity, index=corpus_authors, columns=corpus_authors)
construct_similarity_authors = aggregate_construct_similarity(author_similarity, gold_items, var_ids_authors,
                                                              construct_authors=construct_authors,
                                                              n_similarities=2, verbose=verbose)
fpr_auth, tpr_auth, roc_auc_auth = evaluate(construct_similarity_authors,
                                            construct_identity_gold_authors)
print("ROC AUC authors =", roc_auc_auth, "\n")
# Pearson correlation coefficient between construct similarity computed with reduced LSA items and with authors.
print("Pearson correlation and significance between LSA on items and co-occurr authors:\n",
      pearsonr(np.asarray(construct_similarity_lsa.loc[var_ids_authors, var_ids_authors])[triu_indices],
               np.asarray(construct_similarity_authors)[triu_indices]), "\n")

# TODO: below 0.5 performance might be due to possibly unsorted indices in various places. using doc-vectors
# TODO (cntd.): gives above 0.5 performance. Using unique authors as corpus increases correlation but
# TODO (cntd.): degrades performance.
# Compute construct similarity matrix with LSA on author corpus.
vector_dict_lsa_authors, coauthors_vectors_lsa = train_vectors_lsa(dtm_authors, n_components=100,
                                                                   return_doc_vectors=True)
coauthor_similarity_lsa = pd.DataFrame(np.asarray(coauthors_vectors_lsa).dot(coauthors_vectors_lsa.T),
                                       index=coauthors_vectors_lsa.index.values,
                                       columns=coauthors_vectors_lsa.index.values)
# author_vectors_lsa = term_vectors_from_dict(vector_dict_lsa_authors, terms_authors, normalize=True, verbose=verbose)
# coauthor_similarity_lsa = pd.DataFrame(aggregate_item_similarity(dtm_authors, author_vectors_lsa,
#                                                                n_similarities=2, verbose=verbose),
#                                      index=dtm_authors.index.values, columns=dtm_authors.index.values)
construct_similarity_lsa_authors = aggregate_construct_similarity(coauthor_similarity_lsa, gold_items, var_ids_authors,
                                                                  construct_authors=construct_authors,
                                                                  n_similarities=2, verbose=verbose)
fpr_lsa_auth, tpr_lsa_auth, roc_auc_lsa_auth = evaluate(construct_similarity_lsa_authors,
                                                        construct_identity_gold_authors)
print("ROC AUC LSA authors =", roc_auc_lsa_auth, "\n")
triu_indices = np.triu_indices(len(var_ids_authors), k=1)
# Pearson correlation coefficient between construct similarity computed with reduced LSA items and with LSA authors.
print("Pearson correlation and significance between LSA on items and LSA on authors:\n",
      pearsonr(np.asarray(construct_similarity_lsa.loc[var_ids_authors, var_ids_authors])[triu_indices],
               np.asarray(construct_similarity_lsa_authors)[triu_indices]), "\n")

# Compute construct similarity matrix with GloVe on author corpus. Does item-aggregation work with authors?
vector_dict_glove_authors, loss_glove_authors = train_vectors_glove(ttd_authors, n_components=100, alpha=0.75,
                                                                    x_max=100.0, step_size=0.05, n_epochs=25,
                                                                    batch_size=64, workers=2, verbose=verbose)
vector_dict_glove_authors = {dict_ix_term_authors[key]: value for key, value in
                             vector_dict_glove_authors.items()}  # Translate indices.
author_vectors_glove = term_vectors_from_dict(vector_dict_glove_authors, terms_authors, normalize=True, verbose=verbose)
coauthor_similarity_glove = pd.DataFrame(aggregate_item_similarity(dtm_authors, author_vectors_glove,
                                                                   n_similarities=2, verbose=verbose),
                                         index=dtm_authors.index.values, columns=dtm_authors.index.values)
construct_similarity_glove_authors = aggregate_construct_similarity(coauthor_similarity_glove, gold_items,
                                                                    var_ids_authors,
                                                                    construct_authors=construct_authors,
                                                                    n_similarities=2, verbose=verbose)
fpr_glove_auth, tpr_glove_auth, roc_auc_glove_auth = evaluate(construct_similarity_glove_authors,
                                                              construct_identity_gold_authors)
print("ROC AUC GloVe authors =", roc_auc_glove_auth, "\n")

# Perform linear regression on self-trained GloVe.
lin_reg_trglove_X = np.asarray(np.asarray(construct_similarity_trglove)[triu_indices]).reshape(-1, 1)
regression_y = np.asarray(construct_identity_gold_authors)[triu_indices].reshape(-1, 1)
lin_reg_trglove = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
lin_reg_trglove = lin_reg_trglove.fit(lin_reg_trglove_X, regression_y)
print("Linear regression (trGloVe) coefficients:", lin_reg_trglove.coef_)
lin_reg_trglove_r2 = lin_reg_trglove.score(lin_reg_trglove_X, regression_y)
print("Linear regression (trGloVe) R-squared:", lin_reg_trglove_r2)

# Perform linear regression on all construct similarity measures.
similarities_flat_all = np.asarray(np.asmatrix([np.asarray(construct_similarity_lsa)[triu_indices],
                                                np.asarray(construct_similarity_preglove)[triu_indices],
                                                np.asarray(construct_similarity_trglove)[triu_indices],
                                                np.asarray(construct_similarity_authors)[triu_indices],
                                                np.asarray(construct_similarity_lsa_authors)[triu_indices]]).T)
regression_y = np.asarray(construct_identity_gold_authors)[triu_indices]
lin_reg_all = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
lin_reg_all = lin_reg_all.fit(similarities_flat_all, regression_y.reshape(-1, 1))
print("Linear regression (all) coefficients:", lin_reg_all.coef_)
lin_reg_all_r2 = lin_reg_all.score(similarities_flat_all, regression_y.reshape(-1, 1))
print("Linear regression (all) R-squared:", lin_reg_all_r2, "\n")
lin_reg_all_prediction = lin_reg_all.predict(similarities_flat_all)
fpr_linreg_all, tpr_linreg_all, roc_auc_linreg_all = evaluate(lin_reg_all_prediction,
                                                              np.asarray(construct_identity_gold_authors)[
                                                                  triu_indices])
print("ROC AUC linreg all =", roc_auc_linreg_all, "\n")

# Perform logistic regression on trGloVe.
log_reg_trglove = LogisticRegression(fit_intercept=False, class_weight=None, multi_class='ovr',
                                     solver='newton-cg').fit(np.asarray(
    construct_similarity_trglove)[triu_indices].reshape(-1, 1), regression_y)
print("Logistic regression (trGloVe) coefficients:", log_reg_trglove.coef_)
log_reg_trglove_r2 = log_reg_trglove.score(np.asarray(construct_similarity_trglove)[triu_indices].reshape(-1, 1),
                                           regression_y)
print("Logistic regression (trGloVe) R-squared:", log_reg_trglove_r2, "\n")
log_reg_trglove_prediction = log_reg_trglove.predict_proba(np.asarray(
    construct_similarity_trglove)[triu_indices].reshape(-1, 1))
fpr_log_trglove, tpr_log_trglove, roc_auc_log_trglove = evaluate(log_reg_trglove_prediction[:, 1],
                                                                 np.asarray(construct_identity_gold_authors)[
                                                                     triu_indices])
print("ROC AUC logreg trGloVe =", roc_auc_log_trglove, "\n")

# Perform logistic regression on all construct similarity measures.
log_reg_all = LogisticRegression(fit_intercept=False, class_weight=None, multi_class='ovr',
                                 solver='newton-cg').fit(similarities_flat_all, regression_y)
print("Logistic regression (all) coefficients:", log_reg_all.coef_)
log_reg_all_r2 = log_reg_all.score(similarities_flat_all, regression_y)
print("Logistic regression (all) R-squared:", log_reg_all_r2, "\n")
log_reg_all_prediction = log_reg_all.predict_proba(similarities_flat_all)
fpr_log_all, tpr_log_all, roc_auc_log_all = evaluate(log_reg_all_prediction[:, 1],
                                                     np.asarray(construct_identity_gold_authors)[triu_indices])
print("ROC AUC logreg all =", roc_auc_log_all, "\n")

# Take mean of similarity measures and evaluate.
fpr_mean_all, tpr_mean_all, roc_auc_mean_all = evaluate(np.mean(similarities_flat_all, axis=1),
                                                        np.asarray(construct_identity_gold_authors)[triu_indices])
print("ROC AUC mean all =", roc_auc_mean_all, "\n")

# Take mean of LSA, preGloVe and trGloVe and evaluate.
similarities_flat_lsa_glove = similarities_flat_all[:, 0:3]
fpr_mean_lsa_glove, tpr_mean_lsa_glove, roc_auc_mean_lsa_glove = evaluate(
    np.mean(similarities_flat_lsa_glove, axis=1), np.asarray(construct_identity_gold_authors)[triu_indices])
print("ROC AUC mean LSA GloVe =", roc_auc_mean_lsa_glove, "\n")

# Construct correlation matrix between all construct similarities.
all_similarities_gold = np.asarray(np.asmatrix([np.asarray(construct_similarity_lsa)[triu_indices],
                                                np.asarray(construct_similarity_preglove)[triu_indices],
                                                np.asarray(construct_similarity_trglove)[triu_indices],
                                                np.asarray(construct_similarity_authors)[triu_indices],
                                                np.asarray(construct_similarity_lsa_authors)[triu_indices],
                                                np.asarray(construct_identity_gold_authors)[triu_indices]]).T)
all_similarities_gold = pd.DataFrame(all_similarities_gold, columns=['LSA', 'preGloVe', 'trGloVe', 'Authors',
                                                                     'LSA authors', 'gold'])
all_similarity_correlations = all_similarities_gold.corr()
print("Correlations between all construct similarity measures:")
print(all_similarity_correlations, "\n")

if verbose:
    # Plot ROC curves.
    plt.figure()
plt.grid(True)
plt.plot(fpr_lsa, tpr_lsa)
plt.plot(fpr_preglove, tpr_preglove)
plt.plot(fpr_trglove, tpr_trglove)
plt.plot(fpr_auth, tpr_auth)
plt.plot(fpr_lsa_auth, tpr_lsa_auth)
plt.plot(fpr_mean_all, tpr_mean_all)
plt.plot(fpr_log_all, tpr_log_all)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(["LSA", "preGloVe", "trGloVe", "authors", "LSA authors", "mean all", "log-reg all"])
plt.show()


# TODO: -----------------------------------------
# TODO: ---------------- ARCHIVE ----------------


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


def test_ttcs():
    corpus = np.asarray(['it technolog advanc situat',
                         "mari don't like situat",
                         'technolog great',
                         'yes sir sir that question'])
    processing = 'count'
    result_1, result_2, result_3 = term_term_cooccurrence_scratch(corpus)
    print(result_1, "\n", result_2, "\n", result_3, "\n")
    info(result_1)
    info(result_2)
    info(result_3)
