"""Exploratory waveform intervention on a frozen historical model and split.

This is not a new independent test set or proof of genre understanding.
No files are overwritten. No audio is redistributed. All class probabilities are saved.
"""
from __future__ import annotations
import argparse
import json
import pickle
import platform
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from skimage.transform import resize
import torch
from .config import DATA_DIR, MODELS_DIR, SR, SEG_S, N_MELS, N_FFT, HOP, FIXED_W
from .dataset import resolve_audio_path
from .model import MusicCNN
from .revision_audit import sha256


def intervene(y, sos, gain_db):
    """Parallel filtered component gain. Zero dB is exactly the identity."""
    if gain_db == 0: return y.copy()
    return y + (10.0**(gain_db/20.0)-1.0)*sosfiltfilt(sos,y)


def spectrogram(y):
    s=librosa.feature.melspectrogram(y=y,sr=SR,n_mels=N_MELS,n_fft=N_FFT,hop_length=HOP)
    s=librosa.power_to_db(s,ref=np.max)
    s=(s-s.min())/(s.max()-s.min()+1e-8)
    if s.shape[1]!=FIXED_W:s=resize(s,(N_MELS,FIXED_W),anti_aliasing=True)
    return s.astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='0 = every historical evaluation track; any smaller run is a pilot')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    model=MusicCNN().cpu().eval();cp=MODELS_DIR/'cnn_gtzan_best.pt'
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    with open(MODELS_DIR/'label_encoder.pkl','rb') as f:le=pickle.load(f)
    tracks=pd.read_csv(DATA_DIR/'test_predictions.csv')
    if args.limit: tracks=tracks.head(args.limit)
    filters={'lowpass_250':butter(4,250,fs=SR,output='sos'),
             'bandpass_2000_6000':butter(4,[2000,6000],btype='bandpass',fs=SR,output='sos')}
    config={'status':'exploratory post-review probe, no new training', 'checkpoint_sha256':sha256(cp),
        'source_prediction_sha256':sha256(DATA_DIR/'test_predictions.csv'),
        'script_sha256':sha256(Path(__file__)), 'python':platform.python_version(),
        'sample_rate':SR,'crop_start_seconds':0,'crop_duration_seconds':SEG_S,
        'gains_db':[-12,-6,0,6,12], 'filter_response':'parallel mix y+(10^(dB/20)-1)*sosfiltfilt(sos,y); transition response is not a brick wall',
        'sos':{k:v.tolist() for k,v in filters.items()},'classes':list(le.classes_),
        'rms_matching':'positive uniform scaling after filtering; preprocessing largely removes uniform level changes',
        'clip_handling':'floating-point model input; no clipping or peak normalization',
        'split':'historical validation reused as evaluation; artist overlap unresolved', 'limit':args.limit}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[];failures=[];manifest=[]
    for number,(_,r) in enumerate(tracks.iterrows(),1):
        try:
            path=Path(resolve_audio_path(r.filepath,r.filename,r.true_genre))
            y,_=librosa.load(path,sr=SR,duration=SEG_S)
            if len(y)<int(SR*SEG_S):y=np.pad(y,(0,int(SR*SEG_S)-len(y)))
            rms=float(np.sqrt(np.mean(y.astype(float)**2)))
            xs=[spectrogram(y)];conditions=[('original',0,'none',rms,float(np.max(np.abs(y))))]
            for name,sos in filters.items():
                for gain in config['gains_db']:
                    z=intervene(y,sos,gain)
                    for match in ['unmatched','rms_matched']:
                        z_rms=float(np.sqrt(np.mean(z**2)))
                        zz=z*(rms/z_rms) if match=='rms_matched' and z_rms>0 else z
                        xs.append(spectrogram(zz))
                        conditions.append((name,gain,match,float(np.sqrt(np.mean(zz**2))),float(np.max(np.abs(zz)))))
            with torch.no_grad():
                logits=model(torch.from_numpy(np.stack(xs))[:,None]).numpy()
                probs=torch.softmax(torch.from_numpy(logits),dim=1).numpy()
            pred=int(probs[0].argmax());truth=int(le.transform([r.true_genre])[0])
            hip=int(le.transform(['hiphop'])[0])
            for i,(band,gain,match,new_rms,peak) in enumerate(conditions):
                row={'filename':r.filename,'genre':r.true_genre,'filter':band,'gain_db':gain,'level_control':match,
                     'original_prediction':le.classes_[pred], 'prediction':le.classes_[probs[i].argmax()],
                     'original_correct':pred==truth,'original_target_probability':float(probs[0,pred]),
                     'original_prediction_probability_change':float(probs[i,pred]-probs[0,pred]),
                     'hiphop_probability_change':float(probs[i,hip]-probs[0,hip]),
                     'rms':new_rms,'peak':peak}
                for j,g in enumerate(le.classes_):row[f'p_{g}']=float(probs[i,j]);row[f'logit_{g}']=float(logits[i,j])
                rows.append(row)
            manifest.append({'filename':r.filename,'audio_sha256':sha256(path)})
        except Exception as exc:failures.append({'filename':r.filename,'error':str(exc)})
        if number%20==0:print(f'{number}/{len(tracks)} tracks processed; {len(failures)} failures',flush=True)
    result=pd.DataFrame(rows);result.to_csv(args.output/'per_track.csv',index=False)
    pd.DataFrame(manifest).to_csv(args.output/'audio_manifest.csv',index=False)
    (args.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    result.groupby(['genre','filter','gain_db','level_control'],dropna=False).agg(
        n=('filename','size'),mean_hiphop_probability_change=('hiphop_probability_change','mean'),
        mean_original_prediction_probability_change=('original_prediction_probability_change','mean')
    ).reset_index().to_csv(args.output/'descriptive_summary.csv',index=False)
    if failures:raise RuntimeError(f'{len(failures)} audio failures; partial outputs saved, run not complete')
    print(f'Completed {len(manifest)} tracks; outputs at {args.output}',flush=True)

if __name__=='__main__':main()
