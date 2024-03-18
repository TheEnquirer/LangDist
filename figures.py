import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from shared_features import SharedFeatures
from shared_structure import SharedStructure

from pyvis.network import Network

from Bio import Phylo, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, DistanceMatrix


def sf_barchart():
    # a barchart of the sf similarity in rank order all the top 15 most featured languages against english
    sf = SharedFeatures()

    codes = [x[0] for x in sf.get_most_featured_langs(15)]
    codes.remove('eng')

    # get the similarity of each language to english
    sims = []

    for code in codes:
        sims.append((sf.similarity('eng', code)[0], code))

    # sort the languages by similarity to english
    sims.sort(reverse=True, key=lambda x: x[0])

    # plot the barchart
    fig, ax = plt.subplots()
    ax.bar([sf.code_to_name(x[1]) for x in sims], [x[0] for x in sims])
    ax.set_xlabel('Language')
    ax.set_ylabel('Shared Feature Similarity to English')
    ax.set_title('Shared Feature Similarity of top 15 most measured languages to English')
    plt.xticks(rotation=45)
    # the bottom is getting cut off
    plt.tight_layout()
    plt.show()

def ss_barchart():
    ss = SharedStructure()
    sf = SharedFeatures()
    codes = [x[1] for x in ss.lang_mapping]
    codes.remove("eng")

    # get the similarity of each language to english
    sims = []

    for code in codes:
        sims.append((ss.similarity('eng', code)[0], code))

    # sort the languages by similarity to english
    sims.sort(reverse=True, key=lambda x: x[0])

    names = []

    for x in sims:
        if x[1] == 'ams':
            names.append('Arabic')
            continue
        try:
            names.append(sf.code_to_name(x[1]))
        except:
            names.append(x[1])

    # plot the barchart
    fig, ax = plt.subplots()
    ax.bar(names, [x[0] for x in sims])
    ax.set_xlabel('Language')
    ax.set_ylabel('Shared Structure Similarity to English')
    ax.set_title('Shared Structure Similarity of all PUD languages to English')
    plt.xticks(rotation=45)
    # the bottom is getting cut off
    plt.tight_layout()
    # set the y axis scale to range from the min to the max similarity
    plt.ylim(sims[-1][0]  - 0.1, sims[0][0] + 0.1)

    plt.show()


def force_directed_graph():
    sf = SharedFeatures()
    nodes = [x[0] for x in sf.get_most_featured_langs(15)]
    edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]

    G = nx.Graph()

    for node in nodes:
        G.add_node(node)

    for edge in edges:
        weight, _ = sf.similarity(edge[0], edge[1])
        # if weight > 0.3:
        G.add_edge(edge[0], edge[1], weight=(weight * 10)**2)

    pos = nx.spring_layout(G, weight='weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15, connectionstyle='arc3,rad=0.1')

    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()


def similarity_matrix():
    sf = SharedFeatures()
    codes = [x[0] for x in sf.get_most_featured_langs(25)]

    # sort the codes by geographical distance to english
    codes.sort(key=lambda x: sf.geographical_distance('eng', x))

    sims = np.zeros((len(codes), len(codes)))

    for i in range(len(codes)):
        for j in range(i, len(codes)):
            sims[i, j], _ = sf.similarity(codes[i], codes[j])
            sims[j, i] = sims[i, j]

    fig, ax = plt.subplots()
    cax = ax.matshow(sims, cmap='viridis', interpolation='nearest')

    ax.set_xticks(np.arange(len(codes)))
    ax.set_yticks(np.arange(len(codes)))

    ax.set_xticklabels([sf.code_to_name(x) for x in codes])
    ax.set_yticklabels([sf.code_to_name(x) for x in codes])

    plt.xticks(rotation=90)

    # show a scale
    cbar = plt.colorbar(cax)
    cbar.set_label('Shared Feature Similarity')

    # make the plot look nice
    plt.title('Shared Feature Similarity of top 25 most measured languages')
    plt.tight_layout()


    plt.show()

def similarity_matrix_ss():
    ss = SharedStructure()
    sf = SharedFeatures()

    codes = [x[1] for x in ss.lang_mapping]

    # sort the codes by geographical distance to english
    def distance(x):
        if x == 'jpn_uw':
            return 100 ** 2
        try:
            return sf.geographical_distance('eng', x)
        except:
            return 0

    codes.sort(key=distance)

    sims = np.zeros((len(codes), len(codes)))

    for i in range(len(codes)):
        for j in range(i, len(codes)):
            sims[i, j], _ = ss.similarity(codes[i], codes[j])
            sims[j, i] = sims[i, j]

    fig, ax = plt.subplots()
    cax = ax.matshow(sims, cmap='viridis', interpolation='nearest')

    ax.set_xticks(np.arange(len(codes)))
    ax.set_yticks(np.arange(len(codes)))

    names = []

    for x in codes:
        if x == 'ams':
            names.append('Arabic')
            continue
        try:
            names.append(sf.code_to_name(x))
        except:
            names.append(x)

    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    plt.xticks(rotation=90)

    # show a scale
    cbar = plt.colorbar(cax)
    cbar.set_label('Shared Structure Similarity')

    # make the plot look nice
    plt.title('Shared Structure Similarity of all PUD languages')
    plt.tight_layout()


    plt.show()


def pyvis_network():
    sf = SharedFeatures()

    nodes = [x[0] for x in sf.get_most_featured_langs(15)]

    net = Network()

    for node in nodes:
        net.add_node(node, label=sf.code_to_name(node))

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            weight, _ = sf.similarity(nodes[i], nodes[j])
            if weight > 0.35:
                net.add_edge(nodes[i], nodes[j], value=weight ** 2, title=str(weight))

    # net.set_options("""
    # {
    #   "physics": {
    #     "barnesHut": {
    #       "gravitationalConstant": -80000,
    #       "centralGravity": 0.3,
    #       "springLength": 95,
    #       "springConstant": 0.04,
    #       "damping": 0.09,
    #       "avoidOverlap": 0.1
    #     },
    #     "maxVelocity": 50,
    #     "minVelocity": 0.1,
    #     "solver": "barnesHut",
    #     "timestep": 0.5,
    #     "adaptiveTimestep": true
    #   }
    # }
    # """)

    # net.set_options("""
    # {
    #   "physics": {
    #     "barnesHut": {
    #       "gravitationalConstant": -10000,
    #       "centralGravity": 0.3,
    #       "springLength": 95,
    #       "springConstant": 0.04,
    #       "damping": 0.09,
    #       "avoidOverlap": 0.1
    #     },
    #     "maxVelocity": 50,
    #     "minVelocity": 0.1,
    #     "solver": "barnesHut",
    #     "timestep": 0.5,
    #     "adaptiveTimestep": true
    #   }
    # }
    # """)

    # net.set_options("""
    # {
    #   "nodes": {
    #     "shape": "dot",
    #     "scaling": {
    #       "min": 10,
    #       "max": 30
    #     },
    #   },
    #   "edges": {
    #     "color": {
    #       "inherit": true
    #     },
    #     "smooth": false  // This ensures edges are straight
    #   },
    #   "physics": {
    #     "enabled": false,  // Disable physics
    #     "stabilization": false
    #   },
    # }""")
    # net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)

    net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -10,
      "centralGravity": 0.00,
      "springLength": 50,
      "springStrength": 0.15,
      "damping": 2.5,
      "avoidOverlap": 5
    },
    "edges": {
      "smooth": false
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": { "iterations": 150 }
  }
}
                    """)
    net.show("my_network.html", notebook=False)


def calc_UPGMA():
    sf = SharedFeatures()
    codes = [x[0] for x in sf.get_most_featured_langs(15)]

    # sort the codes by geographical distance to english
    codes.sort(key=lambda x: sf.geographical_distance('eng', x))

    # calcualte a distance matrix

    dists = np.zeros((len(codes), len(codes)))

    for i in range(len(codes)):
        for j in range(i, len(codes)):
            dists[i, j] = sf.geographical_distance(codes[i], codes[j])
            dists[j, i] = dists[i, j]

    constructor = DistanceTreeConstructor()

    # convert dists into a DistanceMatrix object
    dists = DistanceMatrix(dists)

    UGMATree = constructor.upgma(dists)

    Phylo.draw(UGMATree)


if __name__ == "__main__":
    # sf_barchart()
    # force_directed_graph()
    # ss_barchart()
    # similarity_matrix()
    # similarity_matrix_ss()

    # pyvis_network()
    calc_UPGMA()
