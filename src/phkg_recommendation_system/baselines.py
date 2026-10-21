from collections import Counter
def popularity_ranking(train,candidates):
    counts=Counter()
    for target in train.target: counts.update(target)
    ranking=[p for p,_ in counts.most_common() if p in set(candidates)]
    return ranking+[p for p in candidates if p not in ranking]
def recent_history_ranking(hist1,hist2,hist3,popularity,candidates):
    ranking=[]; seen=set()
    for ep in [hist3,hist2,hist1]:
        for p in ep:
            p=int(p)
            if p in candidates and p not in seen: ranking.append(p);seen.add(p)
    for p in popularity:
        if p not in seen: ranking.append(int(p));seen.add(int(p))
    return ranking
