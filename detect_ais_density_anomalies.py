#!/usr/bin/env python3
"""Spatial statistical baseline for clean AIS and generated synthetic routes.
Install: python -m pip install numpy pandas scipy matplotlib cartopy
Supports Danish AIS headers and earlier internal headers. Three chunked scans.
Fits Gaussian-smoothed grid probability mass on first 70% of reference time span;
calibrates threshold on next 15%; scores last 15% and separate synthetic routes.
Percentages split elapsed time, NOT rows. No labelled anomaly used for fitting.
All vessel types pooled; this baseline detects spatial rarity, not unusual dynamics.
"""
import argparse
import gzip
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def stream(path,chunksize):
    names=pd.read_csv(path,nrows=0).columns
    aliases={'# Timestamp':'timestamp','Timestamp':'timestamp','MMSI':'mmsi',
             'Latitude':'lat','Longitude':'lon','latitude':'lat','longitude':'lon'}
    selected={c:aliases.get(c,c) for c in names if aliases.get(c,c) in ['timestamp','mmsi','lat','lon','route_id']}
    if not {'timestamp','lat','lon'}<=set(selected.values()):raise ValueError(f'Missing timestamp/coordinates in {path}')
    if len(set(selected.values()))!=len(selected):raise ValueError('Ambiguous duplicated columns')
    for d in pd.read_csv(path,usecols=list(selected),dtype=str,chunksize=chunksize):
        d=d.rename(columns=selected)
        # Explicit source convention avoids interpreting January 2 as February 1.
        d['timestamp']=pd.to_datetime(d.timestamp,format='mixed',dayfirst=('# Timestamp' in selected),errors='coerce')
        for k in ['lat','lon']:d[k]=pd.to_numeric(d[k],errors='coerce')
        yield d


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--normal',required=True,type=Path)
    p.add_argument('--synthetic',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cell-metres',type=float,default=100)
    p.add_argument('--bandwidth-metres',type=float,default=300,help='Gaussian smoothing standard deviation')
    p.add_argument('--alpha',type=float,default=.01,help='Target upper-tail calibration fraction')
    p.add_argument('--train-fraction',type=float,default=.70)
    p.add_argument('--validation-fraction',type=float,default=.15)
    p.add_argument('--route-fraction',type=float,default=.20,help='Route alarm when >= this fraction of its points are flagged')
    p.add_argument('--roi',nargs=4,type=float,metavar=('WEST','EAST','SOUTH','NORTH'))
    p.add_argument('--chunksize',type=int,default=200000)
    p.add_argument('--no-background',action='store_true')
    a=p.parse_args()
    if not(0<a.alpha<1 and 0<a.train_fraction<1 and 0<a.validation_fraction<1 and a.train_fraction+a.validation_fraction<1 and 0<a.route_fraction<=1):p.error('Invalid fractions')
    if min(a.cell_metres,a.bandwidth_metres,a.chunksize)<=0:p.error('Cell size, bandwidth and chunksize must be positive')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be new or empty')
    projection=None
    if not a.no_background:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            projection=ccrs.PlateCarree()
        except ImportError:p.error('Install cartopy for the background, or pass --no-background')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    roi=a.roi
    if roi is None:
        file=a.normal.parent/'report.json'
        if file.exists():
            settings=json.loads(file.read_text())['parameters'];roi=[settings[k] for k in ['west','east','south','north']]
        else:roi=[12.35,12.85,55.50,55.90]
    w,e,s,n=map(float,roi)
    if not(-180<=w<e<=180 and -85<s<n<85):p.error('Invalid ROI')
    # Local equirectangular metre approximation is suitable for this small ROI.
    if e-w>3 or n-s>3:p.error('This local model supports ROIs no wider/taller than 3 degrees')
    mx=111320*np.cos(np.deg2rad((s+n)/2));my=111320
    nx=int(np.ceil((e-w)*mx/a.cell_metres));ny=int(np.ceil((n-s)*my/a.cell_metres))
    if nx*ny>4_000_000:p.error('Grid too large; increase cell size')
    def valid(d):return d.timestamp.notna()&d.lon.between(w,e)&d.lat.between(s,n)
    def indices(d):
        return (np.clip(((d.lat.to_numpy()-s)*my/a.cell_metres).astype(int),0,ny-1),
                np.clip(((d.lon.to_numpy()-w)*mx/a.cell_metres).astype(int),0,nx-1))
    first=last=None;rejected=0;total=0
    logging.info('Pass 1/3: determining reference time span')
    for d in stream(a.normal,a.chunksize):
        total+=len(d);ok=valid(d);rejected+=int((~ok).sum());t=d.loc[ok,'timestamp']
        if len(t):first=t.min() if first is None else min(first,t.min());last=t.max() if last is None else max(last,t.max())
    if first is None or first==last:p.error('Need reference data spanning multiple timestamps')
    end=last+pd.Timedelta(nanoseconds=1)
    split1=first+(end-first)*a.train_fraction;split2=first+(end-first)*(a.train_fraction+a.validation_fraction)
    grids={k:np.zeros((ny,nx),dtype=np.int64) for k in ['train','validation','test']}
    logging.info('Pass 2/3: accumulating chronological train/validation/test grids')
    for d in stream(a.normal,a.chunksize):
        d=d.loc[valid(d)]
        masks={'train':d.timestamp<split1,'validation':(d.timestamp>=split1)&(d.timestamp<split2),'test':d.timestamp>=split2}
        for k,m in masks.items():np.add.at(grids[k],indices(d.loc[m]),1)
    if any(g.sum()==0 for g in grids.values()):p.error('An empty time split was found; adjust fractions or supply more data')
    smooth=gaussian_filter(grids['train'].astype(float),sigma=a.bandwidth_metres/a.cell_metres,mode='constant',truncate=4)
    # Tiny floor avoids infinities; values beyond kernel support share the maximum score.
    prob=(smooth+1e-12)/(smooth.sum()+1e-12*smooth.size)
    scores=-np.log10(prob)
    present=grids['validation']>0;v=scores[present];weights=grids['validation'][present]
    order=np.argsort(v);v=v[order];weights=weights[order]
    rank=int(np.ceil((1-a.alpha)*int(weights.sum())))
    threshold=float(v[np.searchsorted(np.cumsum(weights),rank)])
    if threshold >= float(scores.max()):
        logging.warning('Threshold equals the maximum score: validation has too many unsupported locations; no points can exceed it. Inspect coverage and bandwidth. Threshold was not relaxed using synthetic labels.')
    flagged=scores>threshold # Strict inequality: ties do not force arbitrary alarms.
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/'density_model.npz',probability_mass=prob,anomaly_score=scores,
        train_counts=grids['train'],validation_counts=grids['validation'],test_counts=grids['test'],
        roi=np.array(roi),cell_metres=a.cell_metres,metres_per_lon_degree=mx,metres_per_lat_degree=my,threshold=threshold)
    logging.info('Pass 3/3: exporting held-out reference scores; threshold %.4f',threshold)
    with gzip.open(a.output/'scored_reference_test.csv.gz','wt',newline='') as f:
        header=True
        for d in stream(a.normal,a.chunksize):
            d=d.loc[valid(d)&(d.timestamp>=split2)].copy()
            if not len(d):continue
            iy,ix=indices(d);d['anomaly_score']=scores[iy,ix];d['is_anomaly']=d.anomaly_score>threshold
            d.to_csv(f,index=False,header=header);header=False
    # Synthetic files from our generator are small; bounded to a documented limit.
    batches=[];sn=0
    for d in stream(a.synthetic,a.chunksize):
        sn+=len(d)
        if sn>1_000_000:p.error('Synthetic input exceeds one million points; use a smaller experiment')
        if not valid(d).all():p.error('Synthetic data contain invalid times or points outside ROI; fix input rather than treating unsupported locations as anomalies')
        if 'route_id' not in d:p.error('Synthetic CSV must contain route_id')
        if d.route_id.isna().any():p.error('Missing synthetic route_id')
        batches.append(d)
    if not sn:p.error('Synthetic file is empty')
    syn=pd.concat(batches,ignore_index=True);iy,ix=indices(syn)
    syn['anomaly_score']=scores[iy,ix];syn['is_anomaly']=syn.anomaly_score>threshold
    syn.to_csv(a.output/'scored_synthetic.csv',index=False)
    routes=syn.groupby('route_id').agg(points=('is_anomaly','size'),flagged_points=('is_anomaly','sum'),flagged_fraction=('is_anomaly','mean'),mean_score=('anomaly_score','mean'),max_score=('anomaly_score','max'))
    routes['route_detected']=routes.flagged_fraction>=a.route_fraction
    routes.to_csv(a.output/'route_summary.csv')
    # Score distributions use exact cell counts, with a common set of bins.
    bins=np.linspace(0,float(scores.max())+1e-6,80)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    histogram=[]
    for label,values,weight,color in [('Validation reference',scores.ravel(),grids['validation'].ravel(),'#829ab1'),('Held-out reference',scores.ravel(),grids['test'].ravel(),'#2477a8'),('Synthetic',syn.anomaly_score.to_numpy(),np.ones(len(syn)),'#cc3838')]:
        h,_=np.histogram(values,bins=bins,weights=weight);pct=h/max(1,h.sum())*100
        ax.stairs(pct,bins,label=label,color=color,linewidth=2)
        histogram.extend({'dataset':label,'bin_start':lo,'bin_end':hi,'percent':float(z)} for lo,hi,z in zip(bins[:-1],bins[1:],pct))
    ax.axvline(threshold,color='#333333',linestyle='--',label=f'Threshold = {threshold:.2f}')
    ax.set(xlabel='Anomaly score = −log10(spatial cell probability)',ylabel='% of observations per bin',title='Spatial rarity: reference versus synthetic observations');ax.legend();ax.grid(alpha=.2)
    fig.savefig(a.output/'score_distributions.png',dpi=180);plt.close(fig)
    pd.DataFrame(histogram).to_csv(a.output/'score_histograms.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained',subplot_kw={'projection':projection} if projection else {})
    xe=w+np.arange(nx+1)*a.cell_metres/mx;ye=s+np.arange(ny+1)*a.cell_metres/my
    for ax in axes:
        if projection:
            ax.set_extent(roi,crs=projection);ax.set_facecolor('#edf3f6')
            ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='#e6e5df',edgecolor='#aaaaaa',linewidth=.4,zorder=2)
            gl=ax.gridlines(draw_labels=True,alpha=.2);gl.top_labels=False;gl.right_labels=False
        else:ax.set(xlim=(w,e),ylim=(s,n),xlabel='Longitude',ylabel='Latitude');ax.set_aspect(1/np.cos(np.deg2rad((s+n)/2)))
    kw={'transform':projection} if projection else {}
    mesh=axes[0].pcolormesh(xe,ye,prob,cmap='Blues',norm=LogNorm(vmin=max(float(prob.min()),float(prob.max())*1e-6),vmax=float(prob.max())),zorder=1,**kw)
    fig.colorbar(mesh,ax=axes[0],label='Smoothed probability per cell',shrink=.65)
    axes[1].pcolormesh(xe,ye,np.ma.masked_equal(grids['train'],0),cmap='Blues',norm=LogNorm(vmin=1,vmax=max(2,grids['train'].max())),zorder=3,alpha=.45,**kw)
    for _,z in syn.groupby('route_id'):
        z=z.sort_values('timestamp');axes[1].plot(z.lon,z.lat,color='#555555',linewidth=.8,zorder=4,**kw)
    for alarm,color,label in [(False,'#28a07c','Not flagged'),(True,'#d33030','Detected synthetic point')]:
        z=syn[syn.is_anomaly==alarm];axes[1].scatter(z.lon,z.lat,s=8,color=color,label=label,zorder=5,**kw)
    axes[0].set_title('Model fitted on training period only');axes[1].set_title('Synthetic anomaly detections');axes[1].legend()
    fig.savefig(a.output/'density_and_detections.png',dpi=180);plt.close(fig)
    metrics={k:{'observations':int(g.sum()),'flagged':int(g[flagged].sum()),'alert_rate':float(g[flagged].sum()/g.sum())} for k,g in grids.items()}
    report={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'roi':roi,
        'time_split':{'start':str(first),'train_end_exclusive':str(split1),'validation_end_exclusive':str(split2),'last_observation':str(last)},
        'reference_input_rows':total,'excluded_invalid_or_outside_roi':rejected,'threshold':threshold,'reference':metrics,
        'synthetic_point_detection_rate':float(syn.is_anomaly.mean()),'synthetic_routes_detected':int(routes.route_detected.sum()),'synthetic_routes_total':len(routes),
        'notes':['Clean reference is not verified normal: held-out alert rate is not a confirmed false-positive rate.',
        'Synthetic generator deliberately selects empty areas. Good performance here is expected and is not evidence of real-world anomaly detection quality.',
        'Gaussian-smoothed histogram approximates spatial KDE; score is cell rarity, not probability of wrongdoing.',
        'All vessel types pooled. Observation weighting reflects AIS reporting frequency and stopping durations.',
        'Kernel truncated at four bandwidths with a tiny probability floor. Tied scores use a strict threshold.',
        'No land mask in the model; coastlines are only plot context. Spatial smoothing may cross land.',
        'Route decision uses flagged-point fraction; not calibrated for route-level false alarms.',
        'Temporal split allows the same vessels in different periods; it does not test unseen-vessel generalization.',
        'Default splits are fractions of elapsed observed time. Synthetic timestamps are not used for fitting or splitting.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    logging.info('Held-out reference alert rate: %.2f%%; synthetic point detection: %.2f%%; routes: %s/%s',100*metrics['test']['alert_rate'],100*syn.is_anomaly.mean(),routes.route_detected.sum(),len(routes))
    logging.info('Outputs: %s',a.output)

if __name__=='__main__':main()
