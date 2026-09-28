"""Check released evidence arithmetic and record counts without model/audio inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
a=root/'results/revision_audit';f=root/'results/filter_probe'
summary=json.loads((a/'summary.json').read_text())
p=pd.read_csv(a/'legacy_predictions.csv')
assert len(p)==summary['n_predictions']==199
assert int(p.correct.sum())==summary['n_correct']==166
r=pd.read_csv(f/'per_track.csv')
assert len(r)==199*21
assert (r.groupby('filename').size()==21).all()
assert not r.duplicated(['filename','filter','gain_db','level_control']).any()
probcols=[c for c in r if c.startswith('p_')]
assert len(probcols)==10
assert np.isfinite(r[probcols].to_numpy()).all()
np.testing.assert_allclose(r[probcols].sum(axis=1),1,atol=2e-7)
b=r[r['filter']=='original'].set_index('filename')
assert (b.prediction.reindex(p.filename).to_numpy()==p.pred_genre.to_numpy()).all()
base_hip=r.filename.map(b.p_hiphop)
np.testing.assert_allclose(r.p_hiphop-base_hip,r.hiphop_probability_change,atol=1e-7)
assert json.loads((f/'failures.json').read_text())==[]
assert r[r.gain_db==0].original_prediction_probability_change.abs().max()<1e-6
c=pd.read_csv(a/'faithfulness_comparison.csv')
assert len(c)==50
assert (c.conf_drop_recomputed-c.conf_drop_saved).abs().max()<4e-7
assert len(pd.read_csv(a/'explanation_samples.csv'))==10
print('Evidence checks passed: 199 baseline matches, 4,179 intervention rows, 50 masking comparisons, finite probabilities and correct arithmetic.')
