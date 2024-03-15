import numpy as np
from conllu import parse, parse_incr, parse_tree
from io import open
from rich import print, inspect
from rich.progress import track
import edist.ted as ted
import edist.tree_edits as tree_edits
import itertools

DATA_DIR = "/Users/huxley/dataset_storage/langdist_data/PUD/"

NORMALIZATION_CONSTANT = 27

lang_mapping = [
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


def get_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
        return parse_tree(data)

def code_to_file(code):
    for file, c, lang in lang_mapping:
        if c == code:
            return file

    raise ValueError("Language not found")

def similarity(a, b):
    """
    a and b are 3 letter language codes, like "eng" or "spa"
    """
   # load the file
   # parse all the sentences into trees
   # loop through and find pairwise edit distance
   # return the average


    a_sents = get_sentences(DATA_DIR + str(code_to_file(a)))
    b_sents = get_sentences(DATA_DIR + str(code_to_file(b)))

    a_trees = [conllu_to_tree(sent) for sent in a_sents]
    b_trees = [conllu_to_tree(sent) for sent in b_sents]

    cumulative_ted = 0

    for idx, tree in enumerate(a_trees):
        cumulative_ted += ted.standard_ted(*a_trees[idx], *b_trees[idx])

    avg_ted = cumulative_ted / len(a_trees)

    return np.exp(-(1/NORMALIZATION_CONSTANT) * avg_ted), avg_ted


def conllu_to_tree(sent):
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
        if parent_idx is not None:
            adj[parent_idx].append(current_idx)
        adj.append([])  # initialize children list for the current node

        # recursively process each child
        for child in node.children:
            traverse(child, current_idx)

    traverse(sent)

    return nodes, adj

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



