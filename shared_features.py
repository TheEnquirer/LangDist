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
import time
import logging
logging.basicConfig(level=logging.INFO)

###############################
# TODO
# * [x] use TF-IDF to weight the params
# * [x] wrap it all in a class with a method to calculate similarity between languages
###############################

DATASET_PATH = "/Users/huxley/dataset_storage/langdist_data/cldf-datasets-wals-878ea47/cldf/StructureDataset-metadata.json"

class SharedFeatures:
    def __init__(self, dataset_path = DATASET_PATH):

        start = time.time()

        self.wals = Dataset.from_metadata(dataset_path)
        self.langs = defaultdict(list)

        for i, v in track(enumerate(self.wals.objects('ValueTable'))):
            self.langs[v.language.id].append(v)

        self.weights = self.generate_feature_weights()

        logging.info(f"Initialzed SharedFeatures similarity with {len(self.langs)} languages in {time.time() - start:.2f} seconds")

    def generate_feature_weights(self):
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

        for lang in self.langs.values():
            for param in lang:
                param_counts[param.parameter] += 1

        # get the total number of times each parameter is in a given state
        param_state_counts = defaultdict(lambda: defaultdict(int))

        for lang in self.langs.values():
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

    def similarity(self, a, b):
        """
        find the normalized weighted number of overlapping features between languages with ids a and b
        @param a: str
        @param b: str
        @return: similarity: float, number of shared parameters to compare with: int
        """

        if self.langs[a] == [] or self.langs[b] == []:
            raise ValueError("Language not found")

        overlap = set([x.parameter for x in self.langs[a]]) & set([x.parameter for x in self.langs[b]])

        if len(overlap) == 0:
            logging.warning(f"No overlapping parameters for {a} and {b}")
            return 0, 0

        similarity = 0
        normalization = 0
        # normalization shouldn't just be the total number of parameters
        # it should be the sum of the weights of the parameters

        # construct a dict of parameter: value for each language
        a_dict = {x.parameter: x.typed_value for x in self.langs[a]}
        b_dict = {x.parameter: x.typed_value for x in self.langs[b]}

        for param in overlap:
            if a_dict[param] == b_dict[param]:
                similarity += self.weights[param][a_dict[param]]

            # find the normalization factor
            # which should be the expected value of this parameter
            # where the probability of a given parameter being in a given state is 1 - weight

            normalization += sum([self.weights[param][state] * (1 - self.weights[param][state]) for state in self.weights[param].keys()])
            # normalization += 1

        # similarity /= len(overlap)
        similarity /= normalization

        return similarity, len(overlap)


    def get_most_featured_langs(self, n):
        """
        get the top n languages with the most parameters
        """
        lang_params = [(k, len(v)) for k, v in self.langs.items()]
        lang_params.sort(key=lambda x: x[1], reverse=True)
        return lang_params[:n]


    def distance_metric(self, similarity_metric, languages, plot=False):
        """
        compares the similarity_metric to the geographical distance between languages
        @param similarity_metric: function, the similarity metric to use
        @param languages: list[str], the languages to compare
        """

        print("update it")

        l_table = self.wals.objects('LanguageTable')

        distances = []
        sims = []

        for a, b in itertools.combinations(languages, 2):
            # d = geodesic(l_table[a].lonlat[::-1], l_table[b].lonlat[::-1]).km

            # TODO should this be squared? cus exponential falloff
            try:
                d = geodesic(l_table[a].lonlat[::-1], l_table[b].lonlat[::-1]).km ** 0.5
                s, n = similarity_metric(a, b)

                distances.append(d)
                sims.append(1-s)
            except:
                logging.error(f"Error for {a} and {b}")

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

    def code_to_name(self, code):
        return self.wals.objects('LanguageTable')[code].name

if __name__ == "__main__":
    sf = SharedFeatures()

    v = sf.distance_metric(sf.similarity, [x[0] for x in sf.get_most_featured_langs(15)], plot=True)
    print(v)

    # print([(*x , sf.code_to_name(x[0])) for x in sf.get_most_featured_langs(15)])


