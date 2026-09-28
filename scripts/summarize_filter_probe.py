"""Create descriptive summaries of the exploratory filter probe, without significance claims."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
r=pd.read_csv(args.run/'per_track.csv')
rows=[]
for genre in sorted(r.genre.unique()):
 for subgroup in ['all','initially_correct']:
  d=r[(r.genre==genre)&(r.level_control=='unmatched')]
  if subgroup=='initially_correct':d=d[d.original_correct]
  for (f,g),a in d.groupby(['filter','gain_db']):
   rows.append({'genre':genre,'subgroup':subgroup,'filter':f,'gain_db':g,'n':len(a),
    'mean_delta_hiphop_probability':a.hiphop_probability_change.mean(),
    'median_delta_hiphop_probability':a.hiphop_probability_change.median(),
    'hiphop_predictions':int((a.prediction=='hiphop').sum()),
    'changed_predictions':int((a.prediction!=a.original_prediction).sum())})
pd.DataFrame(rows).to_csv(args.run/'subgroup_summary.csv',index=False)
b=r[r['filter']=='original'];sham=r[(r.gain_db==0)&(r['filter']!='original')]
m=r[r.level_control=='rms_matched'].merge(r[r.level_control=='unmatched'],on=['filename','filter','gain_db'],suffixes=('_rms','_unmatched'),validate='one_to_one')
summary={'n_tracks':len(b),'n_condition_rows':len(r),'baseline_correct':int(b.original_correct.sum()),
 'max_sham_probability_change':float(sham.original_prediction_probability_change.abs().max()),
 'max_rms_vs_unmatched_hiphop_change_difference':float((m.hiphop_probability_change_rms-m.hiphop_probability_change_unmatched).abs().max()),
 'failed_tracks':len(json.loads((args.run/'failures.json').read_text())),
 'uncertainty':'Descriptive only: artist/recording groups unresolved; validation was reused for selection. No independent-track inferential CI claimed.',
 'hypothesis_status':'Exploratory, selected in response to review and existing attribution observations; not preregistered.'}
(args.run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
