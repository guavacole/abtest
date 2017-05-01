import random
from itertools import groupby, combinations, islice
import scipy as sp
import numpy as np
import scipy.stats
from scipy.stats import norm
import math

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def normal_diff(data_a, data_b):
    m_diff = np.mean(data_a) - np.mean(data_b)
    v = np.var(data_a) + np.var(data_b)
    return 1 - norm.cdf((-1 * m_diff) / math.sqrt(v))


def product_gen(products, frequency, scale=100):
   assert len(products) == len(frequency)
   frequency = sorted(frequency)
   assert frequency[-1] == scale

   dict = {}
   mean = 0
   index = 1
   for p, f in zip(products, frequency):
       while index <= f:
           dict[index] = p
           index = index + 1
           mean = mean + p
   print("True mean: ", mean/scale)
   while True:
       yield dict[random.randint(1,scale)]

def traffic_gen(ratio, a, b, size):
    a_sample = []
    b_sample = []
    for i in range(size):
        if random.random() > ratio: a_sample.append(next(a))
        else: b_sample.append(next(b))

    return a_sample, b_sample

def experiment(a, b):
    exposure_fixed = .5
    exposure_elastic = .5
    sum_fixed = 0
    sum_elastic = 0

    def fixed(data_a, data_b):
        if normal_diff(data_a, data_b) > 0.5: return .95
        else: return .05

    def elastic(data_a, data_b):
        confidence = normal_diff(data_a, data_b)
        if confidence > 0.5: return confidence
        else: return 1 - confidence

    for batch in range(10):
        fixed_data_a, fixed_data_b = traffic_gen(exposure_fixed, a, b, 1000)
        elastic_data_a, elastic_data_b = traffic_gen(exposure_elastic, a, b, 1000)

        sum_fixed += sum(fixed_data_a + fixed_data_b)
        sum_elastic += sum(elastic_data_a + elastic_data_b)

        exposure_fixed = fixed(fixed_data_a, fixed_data_b)
        exposure_elastic = elastic(elastic_data_a, elastic_data_b)

    return sum_fixed - sum_elastic


products = [product_gen([0, 50, 60, 65], [95, 96, 98, 100]),
            product_gen([0, 52, 61, 65], [94, 96, 98, 100]),
            product_gen([0, 49, 60, 68], [93, 97, 98, 100]),
            product_gen([0, 50, 60, 45], [91, 96, 98, 100]),
            product_gen([0, 44, 60, 65], [93, 95, 99, 100]),
            product_gen([0, 45, 64, 72], [94, 95, 96, 100]),
            product_gen([0, 53, 60, 65], [91, 96, 97, 100]),
            product_gen([0, 51, 59, 65], [92, 93, 98, 100]),
            product_gen([0, 50, 64, 65], [94, 95, 98, 100]),
            product_gen([0, 50, 63, 70], [94, 95, 99, 100])]

product_pairs = combinations(products, 2)

results = [experiment(a, b) for a, b in product_pairs]

print("fixed win: ", len(list(filter(lambda x: x > 0, results))))
print("elastic win: ", len(list(filter(lambda x: x < 0, results))))
print("overall: ", sum(results))


