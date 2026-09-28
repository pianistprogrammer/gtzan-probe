"""Build corrected figures exclusively from versioned audit tables (no training)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,default=Path('results/revision_audit'));p.add_argument('--output',type=Path,default=Path('figures/revision'));args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
b=pd.read_csv(args.audit/'band_profiles.csv');genres=b.genre.unique()
fig,axs=plt.subplots(1,3,figsize=(10,4.6),layout='constrained')
for ax,col,title in zip(axs,['input_mean_normalized_logmel','shap_mean_abs','lime_mean_abs'],['Input baseline','Absolute SHAP','Absolute LIME']):
 a=b.pivot(index='genre',columns='band',values=col).reindex(genres).to_numpy()
 a=a/a.max(axis=1,keepdims=True)
 im=ax.imshow(a,vmin=0,vmax=1,cmap='viridis',aspect='auto')
 ax.set_xticks(range(4),['B1','B2','B3','B4']);ax.set_yticks(range(10),genres);ax.set_title(title)
 for i in range(10):
  for j in range(4):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',color='black' if a[i,j]>.6 else 'white',fontsize=8)
fig.colorbar(im,ax=axs,shrink=.8,label='Within-row value / maximum')
fig.savefig(args.output/'band_comparison.pdf');plt.close(fig)
f=pd.read_csv(args.audit/'faithfulness_checkpoint_recomputed.csv')
fig,axs=plt.subplots(2,5,figsize=(10,4.3),sharex=True,sharey=True,layout='constrained')
for ax,g in zip(axs.flat,genres):
 d=f[f.genre==g];ax.plot(d.k_fraction*100,d.conf_drop,'o-',color='#2462a9');ax.axhline(0,color='gray',lw=.5);ax.set_title(g);ax.set_ylim(-.05,1);ax.set_xticks([5,20,50])
fig.supxlabel('Pixels replaced by zero (%)');fig.supylabel('Original minus masked target probability')
fig.savefig(args.output/'faithfulness_recomputed.pdf');plt.close(fig)
a=pd.read_csv(args.audit/'agreement.csv')
fig,ax=plt.subplots(figsize=(9,3.2),layout='constrained');x=np.arange(len(a));w=.26
for i,(col,label) in enumerate([('pearson_signed','Signed Pearson r'),('spearman_signed','Signed Spearman rho'),('exact_top10_iou','Absolute top-10% IoU')]):ax.bar(x+(i-1)*w,a[col],w,label=label)
ax.set_xticks(x,a.genre,rotation=30,ha='right');ax.axhline(0,color='gray',lw=.6);ax.set_ylabel('Descriptive score');ax.legend(ncol=3,loc='upper left',bbox_to_anchor=(0,1.25),frameon=False)
fig.savefig(args.output/'agreement_recomputed.pdf');plt.close(fig)
print(args.output)
