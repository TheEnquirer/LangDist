import numpy as np
import matplotlib.pyplot as plt
import itertools
from rich import print, inspect
from rich.progress import track
from geopy.distance import geodesic

from shared_features import SharedFeatures
from shared_structure import SharedStructure

sf = SharedFeatures()
ss = SharedStructure()

# make a scatterplot of (ss.similarity, sf.similarity) for language pairs

pairs = list(itertools.combinations([x[1] for x in ss.lang_mapping if x[1] != "jpn_uw"], 2))

points = []
for pair in track(pairs):
    try:
        points.append((
            1-ss.similarity(*pair)[0],
            1-sf.similarity(*pair)[0]
        ))
    except:
        print(f"err on {pair}")



# points = [[0.48788934, 0.44432187], [0.56903416, 0.60719308], [0.58549578, 0.32131144], [0.6118101 , 0.51712731], [0.59976209, 0.48641451], [0.55877969, 0.28690338], [0.51541763, 0.49363248], [0.61009039, 0.39380019], [0.5884088 , 0.40275485], [0.54972708, 0.51353879], [0.50397822, 0.45936739], [0.56667861, 0.50558575], [0.53369758, 0.24993058], [0.58895388, 0.42704169], [0.39316791, 0.17349907], [0.48893852, 0.31873644], [0.53202007, 0.48872213], [0.51414021, 0.33165762], [0.60794755, 0.59600654], [0.47513888, 0.38885964], [0.49813264, 0.35598411], [0.50427696, 0.26889048], [0.49961078, 0.52562675], [0.48367923, 0.44740322], [0.45156451, 0.38784728], [0.51186022, 0.41139407], [0.49465789, 0.49843037], [0.46106225, 0.41121589], [0.43621082, 0.29621504], [0.47276911, 0.47935431], [0.51123499, 0.45310334], [0.51685133, 0.49670387], [0.43574254, 0.35674402], [0.4621222 , 0.34928709], [0.45433252, 0.49404033], [0.47994965, 0.4724955 ], [0.50465064, 0.55154245], [0.57598903, 0.3181133 ], [0.6182565 , 0.41043422], [0.58190711, 0.67738408], [0.61451255, 0.49621369], [0.493633  , 0.4955408 ], [0.64820028, 0.54956704], [0.68958407, 0.64658533], [0.68150994, 0.89507202], [0.5026733 , 0.51358033], [0.70393161, 0.80452148], [0.49955527, 0.36950154], [0.56863387, 0.51156217], [0.38321919, 0.2039243 ], [0.47016731, 0.57140674], [0.64680934, 0.71057617], [0.47082949, 0.31242891], [0.58782069, 0.59097451], [0.61761568, 0.2995322 ], [0.65519996, 0.61188984], [0.61362556, 0.35246951], [0.53934092, 0.53734506], [0.64063443, 0.4864763 ], [0.60812771, 0.53075988], [0.56203835, 0.35327487], [0.51244825, 0.28954191], [0.57294648, 0.4360585 ], [0.56237151, 0.3016608 ], [0.64803225, 0.45627584], [0.39970159, 0.27056549], [0.51351221, 0.29844629], [0.54534681, 0.35422012], [0.53858238, 0.28494089], [0.68762028, 0.59938084], [0.62969575, 0.36710448], [0.60029545, 0.30790726], [0.53958068, 0.38547919], [0.6682131 , 0.31818064], [0.67507875, 0.41463442], [0.58987074, 0.45842248], [0.55449155, 0.57880784], [0.60379622, 0.37464855], [0.56158058, 0.24294993], [0.64355953, 0.33834506], [0.40599785, 0.3164468 ], [0.5282698 , 0.48329156], [0.56592354, 0.35085819], [0.54230542, 0.28855381], [0.63411908, 0.44300411], [0.60889398, 0.64271875], [0.53619399, 0.65831241], [0.63828974, 0.53442579], [0.6118101 , 0.62857831], [0.56214245, 0.67303296], [0.51375951, 0.36131507], [0.57500855, 0.63248148], [0.55578688, 0.43836484], [0.64716878, 0.62162061], [0.39965718, 0.35135865], [0.51362633, 0.40106776], [0.55313779, 0.64535744], [0.53324314, 0.40455267], [0.67780958, 0.7467749 ], [0.51434972, 0.70149   ], [0.64714481, 0.63736075], [0.63935444, 0.64659643], [0.59432241, 0.56844703], [0.49706373, 0.27049036], [0.60554304, 0.53794566], [0.52470141, 0.37359628], [0.60502743, 0.50834256], [0.38905404, 0.33080829], [0.49004441, 0.34097805], [0.58201488, 0.69673701], [0.49737679, 0.37613105], [0.61251296, 0.49341873], [0.56278824, 0.7843334 ], [0.53355923, 0.62272563], [0.48673423, 0.51375493], [0.46661104, 0.32128374], [0.49288397, 0.52996461], [0.51011907, 0.26564857], [0.56525321, 0.46251156], [0.38013751, 0.32222367], [0.47248903, 0.37875513], [0.47368051, 0.59256589], [0.49710055, 0.35931963], [0.54359241, 0.60367887], [0.72932952, 0.7878258 ], [0.6161078 , 0.58699747], [0.53998052, 0.31549265], [0.63170464, 0.6133329 ], [0.5556428 , 0.36353778], [0.6593383 , 0.52466392], [0.40271808, 0.26913301], [0.51444498, 0.39765055], [0.59375037, 0.59742833], [0.52911179, 0.30881507], [0.64752842, 0.51728731], [0.65301959, 0.71364739], [0.52801551, 0.41676886], [0.65765548, 0.60106195], [0.53188215, 0.36872664], [0.62036675, 0.56386653], [0.3924841 , 0.3033799 ], [0.50194774, 0.44340378], [0.6163132 , 0.61067225], [0.5038849 , 0.37842514], [0.6204127 , 0.5745034 ], [0.48832321, 0.48550105], [0.65777728, 0.83600603], [0.48942771, 0.38181252], [0.55148086, 0.55647759], [0.37779355, 0.17521654], [0.46135264, 0.45365689], [0.62813511, 0.79230703], [0.46280733, 0.24917937], [0.57547727, 0.57726738], [0.49924083, 0.41293859], [0.48443221, 0.25861827], [0.52343976, 0.40346063], [0.36771598, 0.34994828], [0.46309882, 0.55956228], [0.47799828, 0.35248092], [0.47063771, 0.37242532], [0.51953826, 0.36710423], [0.49951827, 0.35866651], [0.56149739, 0.52539844], [0.38234022, 0.28137372], [0.46900205, 0.38672198], [0.62797228, 0.70990043], [0.47069001, 0.3293113 ], [0.58597305, 0.56076723], [0.60527397, 0.47494993], [0.43445337, 0.4935852 ], [0.52840677, 0.34820577], [0.47729066, 0.42348721], [0.60471379, 0.43954707], [0.5695191 , 0.36913295], [0.41242385, 0.34943164], [0.54502374, 0.37534536], [0.53508304, 0.50514349], [0.58499724, 0.3876503 ], [0.65029229, 0.56433011], [0.41027569, 0.46504065], [0.37269309, 0.26536058], [0.42782671, 0.62459618], [0.40189856, 0.2591836 ], [0.45581572, 0.29574442], [0.52680442, 0.53735731], [0.51520769, 0.40678837], [0.45056215, 0.30810792], [0.56095695, 0.5438044 ], [0.54112168, 0.33695266]]

points = np.array(points)

if False:
    plt.scatter(points[:, 0], points[:, 1])
# make a line of best fit
    m, b = np.polyfit(points[:, 0], points[:, 1], 1)
    plt.plot(points[:, 0], m*points[:, 0] + b, color='red')
    plt.title('Feature Similarity vs Structure Similarity')
    plt.xlabel('Structure Similarity')
    plt.ylabel('Feature Similarity')
    plt.show()

# cor = np.corrcoef(points[:, 0], points[:, 1])[0, 1]
# print(cor)
# exit()

# get the correlation between the two
distances = []


for pair in pairs:
    try:
        distances.append(sf.geographical_distance(*pair) ** 0.5)
        # distances.append(sf.geographical_distance(*pair))
    except:
        print(f"err on {pair}")

if False:
    pairs = list(itertools.combinations([x[0] for x in sf.get_most_featured_langs(15)], 2))
    sims = [(1-sf.similarity(*pair)[0]) for pair in pairs]
    # make a 2d scatter plot of (sf.similarity, distance)
    # and a line of best fit

    distances = [sf.geographical_distance(*pair) ** 0.5 for pair in pairs]

    plt.scatter(sims, distances)
    plt.xlabel('Feature Dis-Similarity')
    plt.ylabel('Root Geographical Distance')
    plt.title('Feature Dis-Similarity vs Root Geographical Distance')

    m, b = np.polyfit(sims, distances, 1)
    plt.plot(sims, m*np.array(sims) + b, color="red")




    plt.show()

# cor = np.corrcoef(sims, distances)[0, 1]
# print(cor)

if True:
# make a 3d scatterplot of (ss.similarity, sf.similarity, distance)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], distances)
    ax.set_xlabel('Structure Similarity')
    ax.set_ylabel('Feature Similarity')
    ax.set_zlabel('Geographical Distance')

# 3-d line of best fit (lmao)
    A = np.vstack([points[:, 0], points[:, 1], np.ones(len(points))]).T
    m, c, d = np.linalg.lstsq(A, distances, rcond=None)[0]
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = m*X + c*Y + d
    ax.plot_surface(X, Y, Z, alpha=0.5)


    # make another plane with the other two axes


    plt.show()

if False:
    # make a 2d scatterplot of (ss.similarity, distance)
    # and a line of best fit




    plt.xlabel('Structure Dis-Similarity')
    plt.ylabel('Root Geographical Distance')
    plt.title('Structure Dis-Similarity vs Root Geographical Distance')

    m, b = np.polyfit(points[:, 0], distances, 1)
    plt.plot(points[:, 0], m*np.array(points[:, 0]) + b, color="red")

    plt.scatter(points[:, 0], distances)


    plt.show()



# cor = np.corrcoef(points[:, 0], distances)[0, 1]

# get the correlation between ss similarity and distance, and sf similarity and distance

ss_distance_cor = np.corrcoef(points[:, 0], distances)[0, 1]
sf_distance_cor = np.corrcoef(points[:, 1], distances)[0, 1]
sf_ss_cor = np.corrcoef(points[:, 0], points[:, 1])[0, 1]

print(f"Correlation between structure similarity and geographical distance: {ss_distance_cor}")
print(f"Correlation between feature similarity and geographical distance: {sf_distance_cor}")
print(f"Correlation between structure similarity and feature similarity: {sf_ss_cor}")

# graph of geographical distance vs structural similarity
if False:
    plt.scatter(distances, points[:, 0])
    plt.xlabel('Geographical Distance')
    plt.ylabel('Structure Similarity')
    plt.title('Geographical Distance vs Structure Similarity')
    plt.show()


