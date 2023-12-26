"""
pyLDAvis Prepare
===============
Main transformation functions for preparing LDAdata to the visualization's data structures
"""
import testSmallThing as ts
import json
import logging
import numpy as np
import pandas as pd
import areaOverlapOfCircle as overCircle
import drawTheCiecle
from collections import namedtuple
from joblib import Parallel, delayed, cpu_count
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
import math as Math
from pyLDAvis.utils import NumPyEncoder
circle_prop = 0.25
mdswidth = 530
mdsheight = 530
barwidth = 530
barheight = 530
termwidth = 90
mdsarea = mdsheight * mdswidth
def _chunks(lambda_seq, n):
    """ Yield successive n-sized chunks from lambda_seq.
    """
    for i in range(0, len(lambda_seq), n):
        yield lambda_seq[i:i + n]

def _job_chunks(lambda_seq, n_jobs):
    n_chunks = n_jobs
    if n_jobs < 0:
        # so, have n chunks if we are using all n cores/cpus
        n_chunks = cpu_count() + 1 - n_jobs

    return _chunks(lambda_seq, n_chunks)

def _pcoa(pair_dists, n_components=2):
    """Principal Coordinate Analysis,
    aka Classical Multidimensional Scaling
    """
    # code referenced from skbio.stats.ordination.pcoa
    # https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py

    # pairwise distance matrix is assumed symmetric
    pair_dists = np.asarray(pair_dists, np.float64)

    # perform SVD on double centred distance matrix
    n = pair_dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = - H.dot(pair_dists ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eig(B)

    # Take first n_components of eigenvalues and eigenvectors
    # sorted in decreasing order
    ix = eigvals.argsort()[::-1][:n_components]
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    # replace any remaining negative eigenvalues and associated eigenvectors with zeroes
    # at least 1 eigenvalue must be zero
    eigvals[np.isclose(eigvals, 0)] = 0
    if np.any(eigvals < 0):
        ix_neg = eigvals < 0
        eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
        eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

    return np.sqrt(eigvals) * eigvecs

def _find_relevance(log_ttd, log_lift, R, lambda_):
    relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
    return relevance.T.apply(lambda topic: topic.nlargest(R).index)

def _find_relevance_chunks(log_ttd, log_lift, R, lambda_seq):
    return pd.concat([_find_relevance(log_ttd, log_lift, R, seq) for seq in lambda_seq])

def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def js_PCoA(distributions):
    """Dimension reduction via Jensen-Shannon Divergence & Principal Coordinate Analysis
    (aka Classical Multidimensional Scaling)

    Parameters
    ----------
    distributions : array-like, shape (`n_dists`, `k`)
        Matrix of distributions probabilities.

    Returns
    -------
    pcoa : array, shape (`n_dists`, 2)
    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    return _pcoa(dist_matrix)
def _topic_info(topic_term_dists, topic_proportion, term_frequency, term_topic_freq,
                vocab, lambda_step, R, n_jobs, start_index=1):
    # marginal distribution over terms (width of blue bars)
    term_proportion = term_frequency / term_frequency.sum()

    # compute the distinctiveness and saliency of the terms:
    # this determines the R terms that are displayed when no topic is selected.
    # TODO(msusol): Make flake8 test pass here with 'unused' variables.
    tt_sum = topic_term_dists.sum()
    topic_given_term = pd.eval("topic_term_dists / tt_sum")
    log_1 = np.log(pd.eval("topic_given_term.T / topic_proportion"))
    kernel = pd.eval("topic_given_term * log_1.T")
    distinctiveness = kernel.sum()
    saliency = term_proportion * distinctiveness
    # Order the terms for the "default" view by decreasing saliency:
    default_term_info = pd.DataFrame({
        'saliency': saliency,
        'Term': vocab,
        'Freq': term_frequency,
        'Total': term_frequency,
        'Category': 'Default'})
    default_term_info = default_term_info.sort_values(
        by='saliency', ascending=False).head(R).drop('saliency', axis=1)
    # Rounding Freq and Total to integer values to match LDAvis code:
    default_term_info['Freq'] = np.floor(default_term_info['Freq'])
    default_term_info['Total'] = np.floor(default_term_info['Total'])
    ranks = np.arange(R, 0, -1)
    default_term_info['logprob'] = default_term_info['loglift'] = ranks
    default_term_info = default_term_info.reindex(columns=[
        "Term", "Freq", "Total", "Category", "logprob", "loglift"
    ])

    # compute relevance and top terms for each topic
    log_lift = np.log(pd.eval("topic_term_dists / term_proportion")).astype("float64")
    log_ttd = np.log(pd.eval("topic_term_dists")).astype("float64")
    lambda_seq = np.arange(0, 1 + lambda_step, lambda_step)

    def topic_top_term_df(tup):
        new_topic_id, (original_topic_id, topic_terms) = tup
        term_ix = topic_terms.unique()
        df = pd.DataFrame({'Term': vocab[term_ix],
                           'Freq': term_topic_freq.loc[original_topic_id, term_ix],
                           'Total': term_frequency[term_ix],
                           'Category': 'Topic%d' % new_topic_id,
                           'logprob': log_ttd.loc[original_topic_id, term_ix].round(4),
                           'loglift': log_lift.loc[original_topic_id, term_ix].round(4),
                           })
        return df.reindex(columns=[
            "Term", "Freq", "Total", "Category", "logprob", "loglift"
        ])

    top_terms = pd.concat(Parallel(n_jobs=n_jobs)
                          (delayed(_find_relevance_chunks)(log_ttd, log_lift, R, ls)
                          for ls in _job_chunks(lambda_seq, n_jobs)))
    topic_dfs = map(topic_top_term_df, enumerate(top_terms.T.iterrows(), start_index))
    return pd.concat([default_term_info] + list(topic_dfs))

def _topic_coordinates(topic_term_dists, topic_proportion, start_index=1):
    K = topic_term_dists.shape[0]
    mds_res = js_PCoA(topic_term_dists)
    assert mds_res.shape == (K, 2)
    mds_df = pd.DataFrame({'x': mds_res[:, 0], 'y': mds_res[:, 1],
                           'topics': range(start_index, K + start_index),
                           'cluster': 1, 'Freq': topic_proportion * 100})
    # note: cluster (should?) be deprecated soon. See: https://github.com/cpsievert/LDAvis/issues/26
    return mds_df
def _df_with_names(data, index_name, columns_name):
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df
def _series_with_name(data, name):
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)
def _token_table(topic_info, term_topic_freq, vocab, term_frequency, start_index=1):
    # last, to compute the areas of the circles when a term is highlighted
    # we must gather all unique terms that could show up (for every combination
    # of topic and value of lambda) and compute its distribution over topics.

    # term-topic frequency table of unique terms across all topics and all values of lambda
    # from Avison_groupLDAdivideTF_IDF import returnArtcleAndTopic
    # topic, aricle = returnArtcleAndTopic()
    term_ix = topic_info.index.unique()
    term_ix = np.sort(term_ix)

    top_topic_terms_freq = term_topic_freq[term_ix]
    # use the new ordering for the topics
    K = len(term_topic_freq)
    top_topic_terms_freq.index = range(start_index, K + start_index)
    top_topic_terms_freq.index.name = 'Topic'

    # we filter to Freq >= 0.5 to avoid sending too much data to the browser
    token_table = pd.DataFrame({'Freq': top_topic_terms_freq.unstack()})\
        .reset_index().set_index('term').query('Freq >= 0.5')

    token_table['Freq'] = token_table['Freq'].round()
    # token_table["myCategory"]=topic
    # token_table["myArticle"]=aricle
    token_table['Term'] = vocab[token_table.index.values].values
    # Normalize token frequencies:
    token_table['Freq'] = token_table.Freq / term_frequency[token_table.index]
    return token_table.sort_values(by=['Term', 'Topic'])
class PreparedData(namedtuple('PreparedData', ['topic_coordinates', 'topic_info', 'token_table',
                                               'R', 'lambda_step', 'plot_opts', 'topic_order'])):

    def sorted_terms(self, topic=1, _lambda=1):
        """Returns a dataframe using _lambda to calculate term relevance of a given topic."""
        tdf = pd.DataFrame(self.topic_info[self.topic_info.Category == 'Topic' + str(topic)])
        if _lambda < 0 or _lambda > 1:
            _lambda = 1
        stdf = tdf.assign(relevance=_lambda * tdf['logprob'] + (1 - _lambda) * tdf['loglift'])
        return stdf.sort_values('relevance', ascending=False)

    def to_dict(self):

        return {'mdsDat': self.topic_coordinates.to_dict(orient='list'),
                'tinfo': self.topic_info.to_dict(orient='list'),
                'token.table': self.token_table.to_dict(orient='list'),
                'R': self.R,
                'lambda.step': self.lambda_step,
                'plot.opts': self.plot_opts,
                'topic.order': self.topic_order}

    def to_json(self):
        return json.dumps(self.to_dict(), cls=NumPyEncoder)

def prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency,
            R=30, lambda_step=0.01, mds=js_PCoA, n_jobs=-1,
            plot_opts=None, sort_topics=True, start_index=1):
    topic_term_dist_cols = [
        pd.Series(topic_term_dist, dtype="float64")
        for topic_term_dist in topic_term_dists
    ]
    doc_topic_dists = _df_with_names(doc_topic_dists, 'doc', 'topic')
    topic_freq = doc_topic_dists.mul(doc_lengths, axis="index").sum()
    topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
    topic_order = topic_proportion.index
    start_index=1
    topic_term_dists = pd.concat(topic_term_dist_cols, axis=1).T
    topic_term_dists = _df_with_names(topic_term_dists, 'topic', 'term')
    topic_term_dists = topic_term_dists.iloc[topic_order]
    term_topic_freq = (topic_term_dists.T * topic_freq).T
    term_frequency = np.sum(term_topic_freq, axis=0)
    vocab = _series_with_name(vocab, 'vocab')
    topic_coordinates = _topic_coordinates(topic_term_dists, topic_proportion, start_index)
    dist=topic_coordinates.to_dict(orient='list')

    x_max_value = max(dist['x'])
    x_min_value = min(dist['x'])
    y_max_value = max(dist['y'])
    y_min_value = min(dist['y'])
    R = min(R, len(vocab))
    topic_info = _topic_info(topic_term_dists, topic_proportion,
                             term_frequency, term_topic_freq, vocab, lambda_step, R,
                             n_jobs, start_index)
    token_table = _token_table(topic_info, term_topic_freq, vocab, term_frequency, start_index)
    client_topic_order = [x + start_index for x in topic_order]
    terms_array = topic_info['Term'].values
    freqs_array = topic_info['Freq'].values
    totals_array = topic_info['Total'].values
    category_array = topic_info['Category'].values
    n=PreparedData(topic_coordinates, topic_info,
                        token_table, R, lambda_step, plot_opts, client_topic_order)
    m=n.to_dict()
    q=n.sorted_terms()

    ts.getFreq_Term(terms_array,freqs_array,totals_array,category_array)
    for i in range(len(dist['x'])):
        drawTheCiecle.draw(x_min_value,x_max_value,y_min_value,y_max_value,dist['x'][i],dist['y'][i],dist['Freq'][i])
        # result = [(x, y, Math.sqrt((z)*mdswidth*mdsheight*circle_prop/Math.pi)) for x, y, z in zip(dist['x'], dist['y'], dist['Freq'])]
    # overCircle.getArea(result)

    i=0
