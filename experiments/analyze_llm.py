import json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import mapping
_, campaigns, _, _ = mapping.load()
groups = mapping.same_defect_groups(campaigns)
res = pd.DataFrame([json.loads(l) for l in open(sys.argv[1], encoding="utf-8")])
res["model"] = res.get("model", pd.Series(dtype=str)).fillna("gemini-3.8-flash")
res["odino"] = res.id.str.split("_").str[0]
orig = res[res.trial == "znana"].set_index("odino").truth
res["orig"] = res.odino.map(orig)
g = lambda c: groups.get(c, -1)
res["llm_ok"] = np.where(res.trial == "znana", res.pred.map(g) == res.truth.map(g),
                         (res.pred == "NONE") | (res.pred.map(g) == res.orig.map(g)))
res["llm_strict"] = res.pred == res.truth
res["tfidf_ok"] = np.where(res.trial == "znana", res.tfidf_pred.map(g) == res.truth.map(g), np.nan)
k, n = res[res.trial == "znana"], res[res.trial == "nowa"]
print(f"przypadków: {len(res)} ({len(k)} znana / {len(n)} nowa)")
print(f"ZNANA: LLM wskazał właściwą wadę {k.llm_ok.mean():.0%} (ściśle {k.llm_strict.mean():.0%}) | TF-IDF {k.tfidf_ok.mean():.0%}")
print(f"  LLM powiedział NONE mimo że wada była na liście: {(k.pred == 'NONE').mean():.0%}")
print(f"NOWA:  LLM poprawnie (NONE albo ta sama wada) {n.llm_ok.mean():.0%} (ściśle NONE {(n.pred == 'NONE').mean():.0%})")
both = res.groupby("odino").llm_ok.all()
print(f"Obie odpowiedzi dla skargi poprawne: {both.mean():.0%}")
print(res.groupby(["model", "trial"]).llm_ok.agg(["mean", "size"]))
