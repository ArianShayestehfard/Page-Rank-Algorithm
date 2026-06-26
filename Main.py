import numpy

def pagerank(adjacency, d=0.85):
    n = len(adjacency)

    out_degree = numpy.sum(adjacency, axis=1)
    out_degree = numpy.maximum(out_degree, 1)

    P = adjacency / out_degree[:, numpy.newaxis]

    v = numpy.ones(n) / n

    for i in range(50):
        v = d * (P.T @ v) + (1 - d) / n

    return v

adjacency = numpy.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]    ])

centrality = pagerank(adjacency)
print(centrality)