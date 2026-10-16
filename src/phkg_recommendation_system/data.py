import pandas as pd
from neo4j import GraphDatabase

PRESCRIPTION_QUERY = """
MATCH (u:Patient)-[r:HAS_PRESCRIPTION]->(p:Protocol)
WHERE r.startingDate IS NOT NULL
  AND r.startingDate <> '0000-00-00 00:00:00'
  AND (toString(r.startingDate) STARTS WITH '2024'
       OR toString(r.startingDate) STARTS WITH '2025')
RETURN u.id AS patient_id, p.protocol_id AS protocol_id,
       r.prescriptionId AS prescription_id, r.startingDate AS starting_date,
       r.endingDate AS ending_date, r.source AS source
ORDER BY patient_id, starting_date
"""

def load_prescriptions(uri,user,password):
    with GraphDatabase.driver(uri,auth=(user,password)) as driver:
        with driver.session() as s:
            records=[r.data() for r in s.run(PRESCRIPTION_QUERY)]
    df=pd.DataFrame(records)
    if df.empty: return df
    df["patient_id"]=pd.to_numeric(df["patient_id"],errors="coerce")
    df["protocol_id"]=pd.to_numeric(df["protocol_id"],errors="coerce")
    df["starting_date"]=pd.to_datetime(df["starting_date"],errors="coerce",utc=True)
    return df.dropna(subset=["patient_id","protocol_id","starting_date"]).copy()

def build_episodes(df,gap_seconds=60):
    x=df.sort_values(["patient_id","starting_date","prescription_id"]).copy()
    x["time_diff_seconds"]=x.groupby("patient_id")["starting_date"].diff().dt.total_seconds()
    x["new_episode"]=(x["time_diff_seconds"].isna()|(x["time_diff_seconds"]>gap_seconds)).astype(int)
    x["episode_number"]=x.groupby("patient_id")["new_episode"].cumsum()
    x["episode_id"]=x["patient_id"].astype(str)+"_"+x["episode_number"].astype(str)
    return (x.groupby(["patient_id","episode_id"],as_index=False)
        .agg(episode_date=("starting_date","min"),
             protocols=("protocol_id",lambda s:sorted(set(map(int,s)))),
             prescription_ids=("prescription_id",lambda s:sorted(set(map(str,s)))))
        .sort_values(["patient_id","episode_date"]).reset_index(drop=True))

def build_windows(episodes,min_episodes=4):
    counts=episodes.groupby("patient_id")["episode_id"].nunique()
    eligible=counts[counts>=min_episodes].index
    e=episodes[episodes.patient_id.isin(eligible)]
    rows=[]
    for pid,g in e.groupby("patient_id"):
        g=g.sort_values("episode_date").reset_index(drop=True)
        for i in range(3,len(g)):
            rows.append({"patient_id":pid,"hist1":g.loc[i-3,"protocols"],
              "hist2":g.loc[i-2,"protocols"],"hist3":g.loc[i-1,"protocols"],
              "hist1_date":g.loc[i-3,"episode_date"],"hist2_date":g.loc[i-2,"episode_date"],
              "hist3_date":g.loc[i-1,"episode_date"],"target":g.loc[i,"protocols"],
              "target_date":g.loc[i,"episode_date"],"target_episode_id":g.loc[i,"episode_id"]})
    return pd.DataFrame(rows)

def chronological_split(windows):
    x=windows.sort_values("target_date").reset_index(drop=True)
    trainval=x[x.target_date.dt.year==2024].copy()
    test=x[x.target_date.dt.year==2025].copy()
    cut=int(len(trainval)*.8)
    return trainval.iloc[:cut].copy(),trainval.iloc[cut:].copy(),test
