import numpy as np
from conllu import parse, parse_incr, parse_tree
from io import open
from rich import print, inspect
from rich.progress import track
import edist.ted as ted
import edist.tree_edits as tree_edits
import itertools
import time
import functools
import logging
logging.basicConfig(level=logging.INFO)

DATA_DIR = "/Users/huxley/dataset_storage/langdist_data/PUD/"

class SharedStructure:
    def __init__(self, dataset_path = DATA_DIR):

        self.NORMALIZATION_CONSTANT = 27
        # self.NORMALIZATION_CONSTANT = 1.08

        self.dataset_path = dataset_path

        # TODO this should probably go somewhere else
        # and also be objects, but :shrug:
        self.lang_mapping = [
            ("ar_pud-ud-test.conllu", "ams", "Arabic"),
            ("hi_pud-ud-test.conllu", "hin", "Hindi"),
            ("pt_pud-ud-test.conllu", "por", "Portuguese"),
            ("cs_pud-ud-test.conllu", "cze", "Czech"),
            ("id_pud-ud-test.conllu", "ind", "Indonesian"),
            ("ru_pud-ud-test.conllu", "rus", "Russian"),
            ("de_pud-ud-test.conllu", "ger", "German"),
            ("is_pud-ud-test.conllu", "ice", "Icelandic"),
            ("sv_pud-ud-test.conllu", "swe", "Swedish"),
            ("en_pud-ud-test.conllu", "eng", "English"),
            ("it_pud-ud-test.conllu", "ita", "Italian"),
            ("th_pud-ud-test.conllu", "tha", "Thai"),
            ("es_pud-ud-test.conllu", "spa", "Spanish"),
            ("tr_pud-ud-test.conllu", "tur", "Turkish"),
            ("fi_pud-ud-test.conllu", "fin", "Finnish"),
            ("ja_pud-ud-test.conllu", "jpn", "Japanese"),
            ("ja_pudluw-ud-test.conllu", "jpn_uw", "Japanese (UW)"),
            ("zh_pud-ud-test.conllu", "mnd", "Mandarin"),
            ("fr_pud-ud-test.conllu", "fre", "French"),
            ("ko_pud-ud-test.conllu", "kor", "Korean"),
            ("pl_pud-ud-test.conllu", "pol", "Polish"),
        ]

    @functools.cache
    def get_sentences(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            return parse_tree(data)

    @functools.cache
    def code_to_file(self, code):
        for file, c, lang in self.lang_mapping:
            if c == code:
                return file

        raise ValueError("Language not found")

    # @functools.cache
    def conllu_to_tree(self, sent):
        """
        @param sent: a TokenTree object representing the root of a sentence
        @return a tuple (nodes, adj) where nodes is a list of UPOS tags
        """

        nodes = []
        adj = []

        def traverse(node, parent_idx=None):
            """
            recursive function to traverse the Conllu tree.

            @param node: the current TokenTree node being processed.
            @param parent_idx: index of the parent node in the nodes list.
            """
            current_idx = len(nodes)

            nodes.append(node.token['upos']) # TODO do we want `deprel`? or some combo?
            # nodes.append(node.token['deprel']) # TODO do we want `deprel`? or some combo?

            if parent_idx is not None:
                adj[parent_idx].append(current_idx)
            adj.append([])  # initialize children list for the current node

            # recursively process each child
            for child in node.children:
                traverse(child, current_idx)

        traverse(sent)

        return nodes, adj

    @functools.cache
    def similarity(self, a, b):
        """
        a and b are 3 letter language codes, like "eng" or "spa"
        """

        # load the file
        # parse all the sentences into trees
        # loop through and find pairwise edit distance
        # return the average

        a_sents = self.get_sentences(self.dataset_path + str(self.code_to_file(a)))
        b_sents = self.get_sentences(self.dataset_path + str(self.code_to_file(b)))

        a_trees = [self.conllu_to_tree(sent) for sent in a_sents]
        b_trees = [self.conllu_to_tree(sent) for sent in b_sents]

        cumulative_ted = 0

        for idx, tree in enumerate(a_trees):

            # normalize by avg num nodes
            # n = (len(a_trees[idx][0]) + len(b_trees[idx][0])) / 2
            # cumulative_ted += ted.standard_ted(*a_trees[idx], *b_trees[idx]) / n

            cumulative_ted += ted.standard_ted(*a_trees[idx], *b_trees[idx])

        avg_ted = cumulative_ted / len(a_trees)

        return np.exp(-(1/self.NORMALIZATION_CONSTANT) * avg_ted), avg_ted

    def get_max_raw_editdistance(self):
        # loop through all pairs of languages
        # find the similarity
        # return the max

        max_raw = 0
        max_pair = None

        for a, b in track(list(itertools.combinations([c for _, c, _ in self.lang_mapping], 2))):
            normalized, raw = self.similarity(a, b)
            if raw > max_raw:
                max_raw = raw
                max_pair = (a, b)

        return max_raw, max_pair

# def get_max_similarity():
#     # loop through all pairs of languages
#     # find the similarity
#     # return the max

#     max_similarity = 0
#     max_pair = None

#     for a, b in itertools.combinations([c for _, c, _ in lang_mapping], 2):
#         sim = similarity(a, b)
#         if sim > max_similarity:
#             max_similarity = sim
#             max_pair = (a, b)

#     return max_similarity, max_pair

# get_max_similarity()

# eng_t = conllu_to_tree(eng[9])
# spa_t = conllu_to_tree(spa[9])
# jpn_t = conllu_to_tree(jpn[9])

#  # ted.standard_ted(*eng_t, *spa_t)
# aligment = ted.standard_ted_backtrace(*eng_t, *jpn_t)

# tree_edits.alignment_to_script(aligment, *eng_t, *jpn_t)

if __name__ == "__main__":
    ss = SharedStructure()

    # print(ss.similarity("eng", "spa"))

    # for a, b in track(list(itertools.combinations([c for _, c, _ in ss.lang_mapping], 2))):
    #     print(a, b, ss.similarity(a, b))

    # print the similarity for all languages w. eng

    for lang in track([c for _, c, _ in ss.lang_mapping]):
        print(lang, ss.similarity("eng", lang))

    # print(ss.get_max_raw_editdistance())



