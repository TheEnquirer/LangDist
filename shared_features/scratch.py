from pycldf import Dataset
import pandas as pd
import rich
from rich import print, inspect
from rich.progress import track
from collections import defaultdict
from geopy.distance import geodesic
import itertools
import numpy as np
import matplotlib.pyplot as plt


DATASET_PATH = "/Users/huxley/dataset_storage/langdist_data/cldf-datasets-wals-878ea47/cldf/StructureDataset-metadata.json"
wals = Dataset.from_metadata(DATASET_PATH)

###############################
# TODO
# * [x] use TF-IDF to weight the params
# * wrap it all in a class with a method to calculate similarity between languages
###############################

# for c in wals.components:
#     print(c)

# for i, val in enumerate(wals["ValueTable"]):
#     if i == 100:
#         print(val)
#         break

# for v in track(wals.objects('ValueTable')):
#     print(v.language.name)

# for l in track(wals.objects('LanguageTable')):
#     print(l)

# print(wals.objects('LanguageTable')[12].__dict__)

# for language in wals.objects('LanguageTable'):
#     if len(language.id) != 3: # skip macro langs
#         continue

#     print(language)
#     # get all the values for this language


#     break

#################################################

langs = defaultdict(list)

for i, v in track(enumerate(wals.objects('ValueTable'))):
    # print(v)
    # inspect(v)
    langs[v.language.id].append(v)


eng_params = [x.parameter for x in langs["eng"]]
fre_params = [x.parameter for x in langs["fre"]]
overlap = set(eng_params) & set(fre_params)

def generate_feature_weights():
    """
    feature weights are defined as:
        for a given parameter, the number of times that parameter is in a given state
        divied by the total number of times that parameter appears
        so it's the probability of a given parameter being in a given state
        and then we invert this to weight less common parameter states more heavily
    @return: dict[str, dict[str, float]], the weights for each parameter, in the form:
        {
            "param": {
                "state": weight,
                ...
            },
            ...
        }
    """

    # get the total number of times each parameter appears
    param_counts = defaultdict(int)

    for lang in langs.values():
        for param in lang:
            param_counts[param.parameter] += 1

    # get the total number of times each parameter is in a given state
    param_state_counts = defaultdict(lambda: defaultdict(int))

    for lang in langs.values():
        for param in lang:
            param_state_counts[param.parameter][param.typed_value] += 1

    # calculate the weights
    weights = defaultdict(lambda: defaultdict(float))

    for param, states in param_state_counts.items():
        total = param_counts[param]
        for state, count in states.items():
            # weights[param][state] = count / total
            # weights[param][state] = 1 / (count / total) # TODO is this the right way to invert the weights?
            # we still want it to be normalized
            # weights[param][state] = 1 / (count / total) / sum([1 / (count / total) for count in states.values()])
            # weights[param][state] = 1 - (count / total)
            weights[param][state] = 1 - (count / total)

    return weights

weights = generate_feature_weights()

def similarity(a, b):
    """
    find the normalized weighted number of overlapping features between languages with ids a and b
    @param a: str
    @param b: str
    @return: similarity: float, number of shared parameters to compare with: int
    """

    overlap = set([x.parameter for x in langs[a]]) & set([x.parameter for x in langs[b]])

    if len(overlap) == 0:
        return 0, 0

    similarity = 0
    normalization = 0
    # normalization shouldn't just be the total number of parameters
    # it should be the sum of the weights of the parameters

    # construct a dict of parameter: value for each language
    a_dict = {x.parameter: x.typed_value for x in langs[a]}
    b_dict = {x.parameter: x.typed_value for x in langs[b]}

    for param in overlap:
        if a_dict[param] == b_dict[param]:
            similarity += weights[param][a_dict[param]]

        # find the normalization factor
        # which should be the expected value of this parameter
        # where the probability of a given parameter being in a given state is 1 - weight

        normalization += sum([weights[param][state] * (1 - weights[param][state]) for state in weights[param].keys()])
        # normalization += 1

    # similarity /= len(overlap)
    similarity /= normalization

    return similarity, len(overlap)


def get_top_langs(n):
    """
    get the top n languages with the most parameters
    """
    lang_params = [(k, len(v)) for k, v in langs.items()]
    lang_params.sort(key=lambda x: x[1], reverse=True)
    return lang_params[:n]


def distance_metric(similarity_metric, languages, plot=False):
    """
    compares the similarity_metric to the geographical distance between languages
    @param similarity_metric: function, the similarity metric to use
    @param languages: list[str], the languages to compare
    """

    l_table = wals.objects('LanguageTable')

    distances = []
    sims = []

    for a, b in itertools.combinations(languages, 2):
        # d = geodesic(l_table[a].lonlat[::-1], l_table[b].lonlat[::-1]).km

        # TODO should this be squared? cus exponential falloff
        d = geodesic(l_table[a].lonlat[::-1], l_table[b].lonlat[::-1]).km ** 0.5
        s, n = similarity_metric(a, b)

        distances.append(d)
        sims.append(1-s)


    # find the correlation between the distances and the similarities
    correlation = np.corrcoef(distances, sims)[0, 1]

    if plot:
        plt.scatter(distances, sims)
        plt.xlabel("Distance (km)")
        plt.ylabel("Dis-Similarity")

        m, b = np.polyfit(distances, sims, 1)
        plt.plot(distances, m*np.array(distances) + b, color="red")

        plt.show()

    return correlation


