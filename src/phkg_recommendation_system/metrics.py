import numpy as np
def recommendation_metrics(prediction_df,k):
    vals={m:[] for m in ["Precision","Recall","MAP","NDCG","HitRate","MRR"]}
    for _,r in prediction_df.iterrows():
        rel=set(r["target"]); ranked=list(r["ranked_protocols"])[:k]
        if not rel: continue
        hits=len(rel&set(ranked)); vals["Precision"].append(hits/k); vals["Recall"].append(hits/len(rel))
        run=0; ps=0
        for rank,p in enumerate(ranked,1):
            if p in rel: run+=1; ps+=run/rank
        d=min(len(rel),k); vals["MAP"].append(ps/d if d else 0)
        dcg=sum(1/np.log2(rank+1) for rank,p in enumerate(ranked,1) if p in rel)
        idcg=sum(1/np.log2(r+1) for r in range(1,d+1)); vals["NDCG"].append(dcg/idcg if idcg else 0)
        vals["HitRate"].append(float(hits>0))
        vals["MRR"].append(next((1/rank for rank,p in enumerate(ranked,1) if p in rel),0))
    return {"K":k,**{m:float(np.mean(v)) for m,v in vals.items()},"N":len(vals["Precision"])}
