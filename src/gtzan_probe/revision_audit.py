"""Audit existing artifacts without replacing them or inferring missing provenance.

Run: python -m gtzan_probe.revision_audit --output results/revision_audit
Only load pickle/checkpoint files from a trusted local source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from .config import DATA_DIR, MODELS_DIR, N_MELS, SR, GENRES
from .model import MusicCNN

# Historical slices, preserved exactly; labels deliberately avoid musical concepts.
AUDIT_BANDS = {'B1': (0, 10), 'B2': (10, 30), 'B3': (30, 80), 'B4': (80, 128)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def exact_top_mask(values, fraction):
    """Fixed cardinality, deterministic flat-index tie break (report ties separately)."""
    values = np.asarray(values)
    if not 0 <= fraction <= 1 or not np.isfinite(values).all():
        raise ValueError('Expected finite values and fraction in [0,1]')
    order = np.argsort(-values.ravel(), kind='stable')
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:int(fraction * values.size)]] = True
    return mask.reshape(values.shape)


def band_metadata():
    # Librosa filters use n_mels+2 equally spaced mel edge points, not n_mels endpoints.
    edges = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SR/2)
    return pd.DataFrame([{'band': b, 'start_bin_inclusive': lo, 'stop_bin_exclusive': hi,
        'first_center_hz': edges[lo+1], 'last_center_hz': edges[hi],
        'support_min_hz': edges[lo], 'support_max_hz': edges[hi+1]}
        for b, (lo, hi) in AUDIT_BANDS.items()])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; audit runs are immutable.')
    args.output.mkdir(parents=True)
    out = args.output
    torch.set_num_threads(2)
    checkpoint = MODELS_DIR / 'cnn_gtzan_best.pt'
    model = MusicCNN().cpu().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    with open(MODELS_DIR / 'label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    sources = [checkpoint, MODELS_DIR / 'label_encoder.pkl']
    artifacts = {}
    for name in ['shap_values.pkl', 'lime_results.pkl']:
        path = DATA_DIR / name
        sources.append(path)
        with open(path, 'rb') as f: artifacts[name] = pickle.load(f)
    shap = artifacts['shap_values.pkl']; lime = artifacts['lime_results.pkl']
    band_metadata().to_csv(out / 'band_definitions.csv', index=False)
    pred = pd.read_csv(DATA_DIR / 'test_predictions.csv')
    sources.append(DATA_DIR / 'test_predictions.csv')
    meta = pd.read_csv(DATA_DIR / 'gtzan_metadata.csv')
    sources.append(DATA_DIR / 'gtzan_metadata.csv')
    clean = meta[meta.fault == 'clean'].copy()
    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(clean,clean.genre))
    split = pd.concat([clean.iloc[train_idx].assign(split='train'),clean.iloc[test_idx].assign(split='validation_reused_as_test')])
    split[['filename','genre','split']].sort_values('filename').to_csv(out/'legacy_split.csv',index=False)
    pred[['filename','true_genre','pred_genre','correct']].to_csv(out/'legacy_predictions.csv',index=False)
    confusion = pd.crosstab(pred.true_genre,pred.pred_genre).reindex(index=GENRES,columns=GENRES,fill_value=0)
    confusion.to_csv(out/'legacy_confusion.csv')
    band_rows, agreement_rows, faith_rows, sample_rows, energy_rows = [], [], [], [], []
    for genre, res in shap.items():
        sv = np.asarray(res['shap_values']); spec = np.asarray(res['spectrogram'])
        lr = lime[genre]
        if res['filename'] != lr['filename'] or not np.allclose(spec,lr['spectrogram']):
            raise ValueError(f'SHAP/LIME input mismatch for {genre}')
        imp = np.zeros_like(spec)
        for seg, weight in lr['local_exp'].items(): imp[lr['segments']==seg] = weight
        target = int(le.transform([genre])[0])
        def predict(x):
            with torch.no_grad():
                return torch.softmax(model(torch.tensor(x,dtype=torch.float32)[None,None]),dim=1).numpy()[0]
        probs = predict(spec)
        sample_rows.append({'genre': genre, 'filename': res['filename'], 'target_class': genre,
            'current_prediction': le.inverse_transform([probs.argmax()])[0], 'current_target_probability':probs[target],
            'lime_segments':len(np.unique(lr['segments']))})
        for b,(lo,hi) in AUDIT_BANDS.items():
            band_rows.append({'genre':genre,'filename':res['filename'],'band':b,
                'shap_mean_abs':np.abs(sv[lo:hi]).mean(),'lime_mean_abs':np.abs(imp[lo:hi]).mean(),
                'input_mean_normalized_logmel':spec[lo:hi].mean()})
        s=np.abs(sv).ravel(); l=np.abs(imp).ravel()
        sm=exact_top_mask(s,.1); lm=exact_top_mask(l,.1)
        agreement_rows.append({'genre':genre,'pearson_signed':pearsonr(sv.ravel(),imp.ravel()).statistic,
            'spearman_signed':spearmanr(sv.ravel(),imp.ravel()).statistic,
            'pearson_abs':pearsonr(s,l).statistic,
            'spearman_abs':spearmanr(s,l).statistic,
            'exact_top10_iou':np.sum(sm&lm)/np.sum(sm|lm),
            'threshold_top10_iou_sensitivity':np.sum((s>=np.percentile(s,90))&(l>=np.percentile(l,90)))/np.sum((s>=np.percentile(s,90))|(l>=np.percentile(l,90))),
            'lime_threshold_fraction':np.mean(l>=np.percentile(l,90)),
            'energy_shap_spearman_pixel':spearmanr(s,spec.ravel()).statistic})
        low=spec<=np.percentile(spec,10); high=spec>np.percentile(spec,90)
        energy_rows.append({'genre':genre,'low_high_attribution_ratio':s.reshape(spec.shape)[low].mean()/(s.reshape(spec.shape)[high].mean()+1e-8)})
        # Reproduce historical masking order exactly; absolute-magnitude ranking may remove negative evidence.
        order=np.argsort(np.abs(sv).ravel())[::-1]
        for frac in [.05,.1,.2,.3,.5]:
            masked=spec.copy();masked.ravel()[order[:int(frac*sv.size)]]=0
            after=predict(masked)[target]
            faith_rows.append({'genre':genre,'filename':res['filename'],'k_fraction':frac,
                'orig_conf':probs[target],'masked_conf':after,'conf_drop':probs[target]-after})
    bands=pd.DataFrame(band_rows);bands.to_csv(out/'band_profiles.csv',index=False)
    pd.DataFrame(sample_rows).to_csv(out/'explanation_samples.csv',index=False)
    pd.DataFrame(agreement_rows).to_csv(out/'agreement.csv',index=False)
    pd.DataFrame(energy_rows).to_csv(out/'energy_ratios.csv',index=False)
    faith=pd.DataFrame(faith_rows);faith.to_csv(out/'faithfulness_checkpoint_recomputed.csv',index=False)
    historical=pd.read_csv(DATA_DIR/'faithfulness_results.csv');sources.append(DATA_DIR/'faithfulness_results.csv')
    compared=faith.merge(historical,on=['genre','k_fraction'],suffixes=('_recomputed','_saved'),validate='one_to_one')
    compared.to_csv(out/'faithfulness_comparison.csv',index=False)
    summary={'status':'audit of legacy artifacts; not a fresh training replication',
        'n_predictions':len(pred),'n_correct':int(pred.correct.sum()),'accuracy':float(pred.correct.mean()),
        'n_train':len(train_idx),'n_validation':len(test_idx),'n_flag_clean':len(clean),
        'metadata_fault_counts':meta.fault.value_counts().to_dict(),
        'n_explained':len(shap),'n_model_parameters':sum(x.numel() for x in model.parameters()),
        'evaluation_filenames_match_reconstructed_split':set(pred.filename)==set(clean.iloc[test_idx].filename),
        'max_saved_recomputed_conf_drop_difference':float(np.abs(compared.conf_drop_recomputed-compared.conf_drop_saved).max()),
        'max_saved_recomputed_original_conf_difference':float(np.abs(compared.orig_conf_recomputed-compared.orig_conf_saved).max()),
        'shap_background_candidate_overlap_with_validation':sorted(set(pd.concat([clean[clean.genre==g].head(5) for g in GENRES]).filename)&set(pred.filename)),
        'historical_seeds_training_and_lime':'not recorded',
        'historical_checkpoint_to_attribution_link':'not recorded; matching input does not establish matching model',
        'python':platform.python_version(),'packages':{n:version(n) for n in ['torch','numpy','pandas','librosa','scipy','shap','lime']},
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'working_tree_changes_present':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'input_checksums.json').write_text(json.dumps({str(x.relative_to(DATA_DIR.parent)):sha256(x) for x in sources},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
