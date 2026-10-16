import json, pathlib
DATA = pathlib.Path(__file__).resolve().parents[2] / "rag_data"
def load(): 
    j = lambda n: json.loads((DATA / n).read_text())
    return j("corpus.json"), j("queries.json"), j("embeddings.json")
