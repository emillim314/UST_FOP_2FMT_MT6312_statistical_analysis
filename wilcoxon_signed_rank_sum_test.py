
import csv
from scipy import stats
from scipy.stats import wilcoxon, rankdata
from statistics import mean, median, stdev, quantiles
import numpy as np
from numpy import (asarray, compress)
from scipy.stats import (find_repeats, distributions)
from math import sqrt
from tabulate import tabulate

FILENAMES = ['data/DataAnalysis_COVID.csv','data/DataAnalysis_Vaccine.csv']


def copied_wilcoxon(x, y=None, zero_method="wilcox", correction=False):
    """
    Calculate the Wilcoxon signed-rank test.
    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.
    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but splitting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    Returns
    -------
    statistic : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    pvalue : float
        The two-sided p-value for the test.
    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if y is None:
        d = asarray(x)
    else:
        x, y = map(asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    if zero_method == "wilcox":
        # Keep all non-zero differences
        d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Warning: sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = min(r_plus, r_minus)
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * distributions.norm.sf(abs(z))

    return T, prob, z

def read_data(filename):
    out = []
    with open(filename) as csv_file_object:
        csv_file_data = csv.reader(csv_file_object)
        next(csv_file_data)
        for row in csv_file_data:
            row_value = []
            for col in row:
                row_value.append(float(col))
            out.append(row_value)
    return out

def get_data_column(data):
    out0 = []
    out1 = []
    for datum in data:
        out0.append(datum[0])
        out1.append(datum[1])
    return out0, out1

def get_differences(pres, posts):
    out = []
    for pre, post in zip(pres, posts):
        out.append(pre-post)
    return out

def get_counts(values):
    negative, zero, positive = 0, 0, 0
    for value in values:
        if value < 0:
            negative += 1
        elif value == 0:
            zero += 1
        else:
            positive += 1
    return negative, zero, positive

def get_is_positive(values):
    out = []
    for value in values:
        if value < 0:
            out.append(-1)
        else:
            out.append(1)
    return out


def get_absolute(values):
    out = []
    for value in values:
        if value < 0:
            out.append(-value)
        else:
            out.append(value)
    return out

## PROBLEM WITH RANKING
def get_ranks(values):
    return rankdata(values, method='average')
    #out = [0] * len(values)
    #for i, x in enumerate(sorted(range(len(values)), key=lambda y: values[y])):
    #    out[x] = i+1
    #return out

def get_signed_ranks(ranks, signs):
    out = []
    for rank, sign in zip(ranks, signs):
        out.append(rank*sign)
    return out

def get_signed_sums(values):
    negatives = []
    positives = []
    for value in values:
        if value < 0:
            negatives.append(value)
        elif value > 0:
            positives.append(value)
    return negatives, positives

def print_descriptive(data):
    print(tabulate([['post',data['descriptive']['post']['N'],data['descriptive']['post']['mean'],
    data['descriptive']['post']['stdev'],data['descriptive']['post']['min'],data['descriptive']['post']['max'],
    data['descriptive']['post']['25th percentile'],data['descriptive']['post']['median'],data['descriptive']['post']['75th percentile']],
    ['pre',data['descriptive']['pre']['N'],data['descriptive']['pre']['mean'],
    data['descriptive']['pre']['stdev'],data['descriptive']['pre']['min'],data['descriptive']['pre']['max'],
    data['descriptive']['pre']['25th percentile'],data['descriptive']['pre']['median'],data['descriptive']['pre']['75th percentile']]],headers=['', 'N', 'mean', 'stdev', 'min', 'max', '25th percentile', 'median', '75th percentile']))

def print_ranks(data):
    print(tabulate([['negative ranks',data['ranks']['negative ranks']['N'],data['ranks']['negative ranks']['mean rank'],data['ranks']['negative ranks']['sum of ranks']],
    ['positive ranks',data['ranks']['positive ranks']['N'],data['ranks']['positive ranks']['mean rank'],data['ranks']['positive ranks']['sum of ranks']],
    ['ties',data['ranks']['ties']['N'],data['ranks']['ties']['mean rank'],data['ranks']['ties']['sum of ranks']],
    ['total',data['ranks']['total']['N'],data['ranks']['total']['mean rank'],data['ranks']['total']['sum of ranks']]],headers=['','N','mean rank','sum of ranks']))

def print_wilcoxon(data):
    print(tabulate([['statistic',data['test statistics']['statistic']],['Z',data['test statistics']['Z']],['p-value',data['test statistics']['pvalue']]]))

def print_all(data, filename):
    print(filename, end='\n\n')
    print_descriptive(data)
    print()
    print_ranks(data)
    print()
    print_wilcoxon(data)

def get_wilcoxon_data(pre, post):
    if len(pre) != len(post):
        raise Exception('pre-test and post-test sizes do not match')
    N = len(pre)
    pre_mean = mean(pre)
    post_mean = mean(post)
    pre_stdev = stdev(pre)
    post_stdev = stdev(post)
    pre_min = min(pre)
    post_min = min(post)
    pre_max = max(pre)
    post_max = max(post)
    pre_25, pre_mid, pre_75 = quantiles(pre, n=4)
    post_25, post_mid, post_75 = quantiles(post, n=4)
    differences = get_differences(pre, post)
    N_neg, N_tie, N_pos = get_counts(differences)
    differences = list(filter(lambda x: x!=0, differences))
    positives = get_is_positive(differences)
    absolutes = get_absolute(differences)
    ranks = get_ranks(absolutes)
    signed_ranks = get_signed_ranks(ranks, positives)
    negative_ranks, positive_ranks = get_signed_sums(signed_ranks)
    negative_sum_rank = -sum(negative_ranks)
    positive_sum_rank = sum(positive_ranks)
    negative_mean = -mean(negative_ranks)
    positive_mean = mean(positive_ranks)
    N_total = N_neg + N_tie + N_pos
    wilcoxon_stat, wilcoxon_pval, wilcoxon_z = copied_wilcoxon(pre, post)
    return {
        'descriptive':{
            'pre':{
                'N':N,
                'mean':pre_mean,
                'stdev':pre_stdev,
                'min':pre_min,
                'max':pre_max,
                '25th percentile':pre_25,
                'median':pre_mid,
                '75th percentile':pre_75
            },
            'post':{
                'N':N,
                'mean':post_mean,
                'stdev':post_stdev,
                'min':post_min,
                'max':post_max,
                '25th percentile':post_25,
                'median':post_mid,
                '75th percentile':post_75
            }
        },
        'ranks':{
            'negative ranks':{
                'N':N_neg,
                'mean rank':negative_mean,
                'sum of ranks':negative_sum_rank
            },
            'positive ranks':{
                'N':N_pos,
                'mean rank':positive_mean,
                'sum of ranks':positive_sum_rank
            },
            'ties':{
                'N':N_tie,
                'mean rank':'',
                'sum of ranks':''
            },
            'total':{
                'N':N_total,
                'mean rank':'',
                'sum of ranks':''
            }
        },
        'test statistics':{
            'statistic':wilcoxon_stat,
            'Z': wilcoxon_z,
            'pvalue':wilcoxon_pval
        }
    }

for filename in FILENAMES:
    data1, data2 = get_data_column(read_data(filename))
    print('[BEGIN]')
    print_all(get_wilcoxon_data(data1,data2), filename)
    print('[END]',end='\n\n')
