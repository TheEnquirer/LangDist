from pycldf import Dataset
import pandas as pd
import rich
from rich import print, inspect
from rich.progress import track
from collections import defaultdict


DATASET_PATH = "/Users/huxley/dataset_storage/langdist_data/cldf-datasets-wals-878ea47/cldf/StructureDataset-metadata.json"
wals = Dataset.from_metadata(DATASET_PATH)

# TODO
# * use TF-IDF to weight the params
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


def calc_overlap(a, b):
    """
    find the number of overlapping features between languages with ids a and b
    @param a: str
    @param b: str
    @return: similarity: float, number of shared parameters to compare with: int
    """

    overlap = set([x.parameter for x in langs[a]]) & set([x.parameter for x in langs[b]])

    if len(overlap) == 0:
        return 0, 0

    similarity = 0

    # construct a dict of parameter: value for each language
    a_dict = {x.parameter: x.typed_value for x in langs[a]}
    b_dict = {x.parameter: x.typed_value for x in langs[b]}

    for param in overlap:
        if a_dict[param] == b_dict[param]:
            similarity += 1

    similarity /= len(overlap)

    return similarity, len(overlap)


def get_top_langs(n):
    """
    get the top n languages with the most parameters
    """
    lang_params = [(k, len(v)) for k, v in langs.items()]
    lang_params.sort(key=lambda x: x[1], reverse=True)
    return lang_params[:n]


