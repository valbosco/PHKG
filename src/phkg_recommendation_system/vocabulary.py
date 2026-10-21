def build_protocol_vocabulary(windows,df,development_year=2024):
    candidates=sorted(df.loc[df.starting_date.dt.year==development_year,"protocol_id"].astype(int).unique())
    history=set()
    for col in ["hist1","hist2","hist3"]:
        for ps in windows[col]: history.update(map(int,ps))
    graph_protocols=sorted(set(candidates)|history)
    return candidates,graph_protocols,{p:i for i,p in enumerate(candidates)},{p:i for i,p in enumerate(graph_protocols)}
