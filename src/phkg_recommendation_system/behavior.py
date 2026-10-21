import pandas as pd
def engaged_sessions(session_df): return session_df[session_df["score"].fillna(0)>0].copy()
def closed_sessions(session_df): return session_df[session_df["status"]=="CLOSED"].copy()
def build_behavioural_followup(targets,sessions,horizon_days):
    rows=[]
    for _,r in targets.iterrows():
        end=r["target_date"]+pd.Timedelta(days=horizon_days)
        if pd.notna(r.get("next_episode_date",pd.NaT)): end=min(end,r["next_episode_date"])
        s=sessions[(sessions.patient_id==r.patient_id)&(sessions.starting_date>=r.target_date)&(sessions.starting_date<end)]
        played=sorted(s.protocol_id.dropna().astype(int).unique().tolist())
        rows.append({"patient_id":r.patient_id,"target_episode_id":r["target_episode_id"],"target_date":r.target_date,
          "observation_end":end,"horizon_days":horizon_days,"n_session_rows":len(s),
          "n_unique_sessions":s.session_id.nunique(),"n_played_protocols":len(played),
          "played_protocols":played,"has_followup":len(s)>0})
    return pd.DataFrame(rows)
