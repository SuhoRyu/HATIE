import pickle

def load_queries(path):
    with open(path, "rb") as f:
        queries = pickle.load(f)

    queryIds = []
    queryIdxs = []
    for imgID in queries.keys():
        for i, query in enumerate(queries[imgID]):
            queryIds.append(query['id'])
            queryIdxs.append((imgID, i))

    return queries, queryIds, queryIdxs
