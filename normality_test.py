
import csv
from scipy.stats import shapiro
from scipy.stats import kstest
from statsmodels.stats.diagnostic import lilliefors
from tabulate import tabulate


FILENAMES = ['data/DataAnalysis_COVID.csv','data/DataAnalysis_Vaccine.csv']

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

def get_difference_column(data):
    out = []
    for datum in data:
        out.append(datum[2])
    return out

def check_normality(data):
    kolmogorov_data = kstest(data, 'norm')
    shapiro_data = shapiro(data)
    lilliefors_data = lilliefors(data)
    df = len(data)
    return {
        'Kolmogorov-Smirnov':{
            'statistic': kolmogorov_data.statistic,
            'df': df,
            'pvalue': kolmogorov_data.pvalue
        },
        'Lilliefors':{
            'statistic': lilliefors_data[0],
            'df': df,
            'pvalue': lilliefors_data[1]
        },
        'Shapiro-Wilk':{
            'statistic': shapiro_data.statistic,
            'df': df,
            'pvalue': shapiro_data.pvalue
        }
    }

def print_data(testname, data):
    print('[{} Test]'.format(testname))
    print(tabulate([[data[testname]['statistic'], data[testname]['df'], data[testname]['pvalue']]], headers=['statistic', 'df', 'p-value']))
    print()

def print_all_data(data, filename):
    print('[BEGIN]')
    print(filename, end='\n\n')
    for testname in data:
        print_data(testname, data)
    print('[END]', end='\n\n')

for filename in FILENAMES:
    data = get_difference_column(read_data(filename))
    normality_data = check_normality(data)
    print_all_data(normality_data, filename)
