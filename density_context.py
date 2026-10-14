#!/usr/bin/env python3
"""Fit historical grid context on training trajectories; attach it to later rows.
Uses unique track-cell visits, NOT AIS message counts or vessel-hours. Grid size
is in degrees, not equal-area. Compare models with/without context separately.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def cells(f,size):
    return (np.floor(f.lon/size).astype(int).astype(str)+':'+np.floor(f.lat/size).astype(int).astype(str))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fit','apply']);p.add_argument('input')
    p.add_argument('--model',default='density_model.json');p.add_argument('--output',default='tracks_with_density.parquet')
    p.add_argument('--train-end');p.add_argument('--cell-degrees',type=float,default=.02)
    a=p.parse_args();f=pd.read_parquet(a.input)
    if a.mode=='fit':
        if not a.train_end or a.cell_degrees<=0:p.error('fit needs --train-end and positive --cell-degrees')
        cut=pd.Timestamp(a.train_end,tz='UTC')
        ends=f.groupby('track_id').timestamp.max();ids=ends.index[ends<cut]
        f=f[f.track_id.isin(ids)].copy()
        if f.empty:raise ValueError('No training trajectories')
        f['cell']=cells(f,a.cell_degrees)
        counts=f[['track_id','cell']].drop_duplicates().groupby('cell').size()
        model=dict(train_end=str(cut),cell_degrees=a.cell_degrees,visits=counts.to_dict(),
                   total_visits=int(counts.sum()),training_tracks=f.track_id.nunique(),
                   description='Unique track-cell visits; no dwell-time estimate or class/season conditioning')
        Path(a.model).write_text(json.dumps(model,indent=2))
        print('Saved training-only density context:',a.model)
    else:
        model=json.loads(Path(a.model).read_text());f['density_cell']=cells(f,model['cell_degrees'])
        f['historical_track_cell_visits']=f.density_cell.map(model['visits']).fillna(0).astype(int)
        f['unseen_training_cell']=f.historical_track_cell_visits.eq(0)
        # One unseen-cell bucket: descriptive surprisal, not a calibrated anomaly probability.
        denominator=model['total_visits']+len(model['visits'])+1
        f['density_surprisal']=-np.log((f.historical_track_cell_visits+1)/denominator)
        f.to_parquet(a.output,index=False)
        print('Saved',a.output,'; unseen cells can indicate missing coverage, not abnormal behaviour.')
if __name__=='__main__':main()
