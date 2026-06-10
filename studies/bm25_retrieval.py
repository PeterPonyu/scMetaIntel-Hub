# ruff: noqa  (standalone research/repro script)
"""BM25 vs token-overlap retrieval eval (scMetaIntel-Hub issue #17).

Run:  python studies/bm25_retrieval.py


Does scMeta's 'BM25' label match its scorer? Retrieval eval on real qrels.

Compares three scorers on BEIR/SciFact (real biomedical IR with human qrels):
  (1) token-overlap  -- a faithful reproduction of scMeta retrieve.py's sparse scorer
                        (sum of query-term frequencies in the doc; no IDF/length norm)
  (2) BM25Okapi      -- a real BM25 (rank_bm25)
  (3) dense          -- a sentence-transformer bi-encoder, for context
Metric: Recall@k, nDCG@10, MRR averaged over queries; paired bootstrap test
between token-overlap and BM25.
"""
import os, re, math
for k in ["ALL_PROXY","all_proxy","HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"]:
    os.environ.pop(k,None)
os.environ["HF_HOME"]="/tmp/sci-studies/hf"; os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
rng=np.random.default_rng(0)

corpus=load_dataset("BeIR/scifact","corpus",split="corpus")
queries=load_dataset("BeIR/scifact","queries",split="queries")
qrels=load_dataset("BeIR/scifact-qrels",split="test")
docids=[str(r["_id"]) for r in corpus]
doctext=[(r["title"]+". "+r["text"]).strip() for r in corpus]
qtext={str(r["_id"]):r["text"].strip() for r in queries}
rel={}
for r in qrels:
    if int(r["score"])>=1: rel.setdefault(str(r["query-id"]),set()).add(str(r["corpus-id"]))
test_q=[q for q in rel if q in qtext]
print(f"corpus={len(docids)} eval_queries={len(test_q)}")

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

# (2) BM25
bm25=BM25Okapi([tok(t) for t in doctext])
# (1) token-overlap, exactly as scmetaintel/retrieve.py: Counter(text.split()) overlap of query tokens
from collections import Counter
doc_counters=[Counter(t.lower().split()) for t in doctext]
def overlap_scores(q):
    qset=set(q.lower().split())
    return np.array([sum(c[t] for t in qset if t in c) for c in doc_counters], dtype=float)

# (3) dense
from sentence_transformers import SentenceTransformer
dm=SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
D=dm.encode(doctext, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

def dcg(rels_in_rank,k):
    return sum((rels_in_rank[i]/math.log2(i+2)) for i in range(min(k,len(rels_in_rank))))
def eval_ranking(order, relset):
    ranked=[1 if docids[i] in relset else 0 for i in order]
    rr=0.0
    for i,x in enumerate(ranked):
        if x: rr=1.0/(i+1); break
    idcg=dcg(sorted(ranked,reverse=True),10); ndcg=(dcg(ranked,10)/idcg) if idcg>0 else 0.0
    out={"mrr":rr,"ndcg10":ndcg}
    for k in (10,50,100):
        out[f"r@{k}"]= sum(ranked[:k])/len(relset) if relset else 0.0
    return out

methods={"token-overlap (scMeta sparse)":[], "BM25Okapi (real)":[], "dense bge-small":[]}
qvecs=dm.encode([qtext[q] for q in test_q], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
for qi,q in enumerate(test_q):
    relset=rel[q]
    o_overlap=np.argsort(-overlap_scores(qtext[q]))
    o_bm25=np.argsort(-bm25.get_scores(tok(qtext[q])))
    o_dense=np.argsort(-(D@qvecs[qi]))
    methods["token-overlap (scMeta sparse)"].append(eval_ranking(o_overlap,relset))
    methods["BM25Okapi (real)"].append(eval_ranking(o_bm25,relset))
    methods["dense bge-small"].append(eval_ranking(o_dense,relset))

def agg(rows,key): return float(np.mean([r[key] for r in rows]))
print(f"\n{'method':32s} {'nDCG@10':>8s} {'R@10':>7s} {'R@50':>7s} {'R@100':>7s} {'MRR':>6s}")
print("-"*78)
for m,rows in methods.items():
    print(f"{m:32s} {agg(rows,'ndcg10'):8.3f} {agg(rows,'r@10'):7.3f} {agg(rows,'r@50'):7.3f} {agg(rows,'r@100'):7.3f} {agg(rows,'mrr'):6.3f}")

# paired bootstrap: BM25 - token-overlap on nDCG@10
a=np.array([r["ndcg10"] for r in methods["BM25Okapi (real)"]])
b=np.array([r["ndcg10"] for r in methods["token-overlap (scMeta sparse)"]])
diffs=[]
for _ in range(2000):
    idx=rng.integers(0,len(a),len(a)); diffs.append((a[idx]-b[idx]).mean())
lo,hi=np.percentile(diffs,2.5),np.percentile(diffs,97.5)
print(f"\nBM25 - token-overlap  nDCG@10 delta = {a.mean()-b.mean():+.3f}  (95% CI [{lo:+.3f},{hi:+.3f}])  -> {'significant' if lo>0 else 'n.s.'}")
