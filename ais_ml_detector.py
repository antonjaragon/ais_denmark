#!/usr/bin/env python3
"""AIS ML discovery + expert review. CPU, bounded batches, disk-backed SQLite.
INSTALL: python -m pip install numpy pandas scipy scikit-learn matplotlib cartopy shapely pyproj joblib threadpoolctl
FIT: python ais_ml_detector.py fit --input ./copenhagen_january/clean_ais_fixed.csv.gz --output ./ais_ml_run
REVIEW: edit ais_ml_run/results/review_queue.csv: feedback_label=normal/anomaly/uncertain
UPDATE: python ais_ml_detector.py feedback --run ./ais_ml_run --labels ./ais_ml_run/results/review_queue.csv

Fits ALL valid input observations using SGDOneClassSVM.partial_fit + an approximate
RBF kernel. Only kernel landmarks/PCA visualization use a bounded random sample.
Input: cleaned Danish AIS or internal analysis CSV/GZ, or directory of these files.
Do not point input at a directory mixing quality summaries and synthetic files.
Model discovery is retrospective, not a held-out accuracy experiment. Keep new
months/vessels for independent evaluation. Natural Earth may download on first use.
"""
import argparse
from collections import Counter
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import time
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TYPES=['cargo','tanker','fishing','passenger','tug','pleasure','other','unknown']
NUM=['x_km','y_km','sog','cog_sin','cog_cos','heading_sin','heading_cos',
     'heading_cog_difference','log_gap_seconds','implied_speed','speed_disagreement',
     'acceleration_knots_min','turn_degrees_min','hour_sin','hour_cos',
     'on_land','log_inland_metres','sog_missing','cog_missing','heading_missing','temporal_missing']
FEATURES=NUM+['type_'+g for g in TYPES]
META=['record_id','timestamp','mmsi','lat','lon','vessel_type','source_file','source_row','land_motion_rule']


def vessel_type(v):
    s=str(v).strip().lower()
    try:
        code=float(s)
        if code.is_integer():
            c=int(code)
            if 70<=c<=79:return 'cargo'
            if 80<=c<=89:return 'tanker'
            if c==30:return 'fishing'
            if 60<=c<=69:return 'passenger'
            if c in [31,32,52]:return 'tug'
            if c in [36,37]:return 'pleasure'
            return 'unknown' if c==0 else 'other'
    except ValueError:pass
    for pattern,g in [('cargo','cargo'),('tanker','tanker'),('fishing','fishing'),('passenger','passenger'),('tug|towing','tug'),('pleasure|sailing','pleasure')]:
        if re.search(pattern,s):return g
    return 'unknown' if s in ['','nan','unknown','undefined','not available','none'] else 'other'


def input_chunks(path,size):
    header=pd.read_csv(path,nrows=0).columns
    aliases={'latitude':'lat','longitude':'lon','ship type':'ship_type','type of mobile':'mobile'}
    mapping={}
    for col in header:
        key=col.strip().lstrip('#').strip().lower();key=aliases.get(key,key)
        if key in ['timestamp','mmsi','lat','lon','sog','cog','heading','ship_type','mobile']:
            if key in mapping.values():raise ValueError(f'{path}: ambiguous {key}')
            mapping[col]=key
    if not {'timestamp','mmsi','lat','lon'}<=set(mapping.values()):raise ValueError(f'{path}: missing essential AIS columns')
    offset=0
    for d in pd.read_csv(path,usecols=list(mapping),dtype=str,chunksize=size):
        d=d.rename(columns=mapping);d['source_row']=np.arange(offset+2,offset+len(d)+2);offset+=len(d)
        raw=d.timestamp.str.strip();slash=raw.str.contains('/',na=False)
        ts=pd.to_datetime(raw.where(~slash),format='mixed',errors='coerce')
        ts.loc[slash]=pd.to_datetime(raw.loc[slash],format='%d/%m/%Y %H:%M:%S',errors='coerce');d['timestamp']=ts
        for key in ['lat','lon','sog','cog','heading']:
            d[key]=pd.to_numeric(d[key],errors='coerce') if key in d else np.nan
        d['vessel_type']=d.ship_type.fillna('').map(vessel_type) if 'ship_type' in d else 'unknown'
        yield d


def read_features(db,size,order='record_id'):
    yield from pd.read_sql_query('SELECT * FROM features ORDER BY '+order,db,chunksize=size)


def matrix(d):return d[FEATURES].to_numpy(dtype=np.float64)
def scaled(x,scaler):return np.column_stack([np.clip(scaler.transform(x[:,:len(NUM)]),-8,8),x[:,len(NUM):]])


def make_features(db,a):
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import unary_union,transform
    from shapely import contains_xy
    from cartopy.io import shapereader
    bounds=db.execute('SELECT min(lon),max(lon),min(lat),max(lat) FROM observations').fetchone()
    w,e,s,n=bounds
    if not (-10<=w<=e<=35 and 35<=s<=n<=72):raise ValueError('Geography currently configured for European AIS using EPSG:3035')
    fw=Transformer.from_crs(4326,3035,always_xy=True);inverse=Transformer.from_crs(3035,4326,always_xy=True)
    h=a.land_grid_metres
    # Densify geographic bounding box for raster bounds, padding beyond data.
    lons=np.r_[np.linspace(w,e,100),np.full(100,e),np.linspace(e,w,100),np.full(100,w)]
    lats=np.r_[np.full(100,s),np.linspace(s,n,100),np.full(100,n),np.linspace(n,s,100)]
    xx,yy=fw.transform(lons,lats);xmin=min(xx)-3*h;ymin=min(yy)-3*h
    nx=max(8,int(np.ceil((max(xx)-xmin)/h))+4);ny=max(8,int(np.ceil((max(yy)-ymin)/h))+4)
    if nx*ny>6_000_000:raise ValueError('Land grid too large; increase --land-grid-metres')
    logging.info('Loading land polygons and building %s x %s land-context grid',nx,ny)
    landfile=shapereader.natural_earth(resolution='10m',category='physical',name='land')
    region=box(w-1,s-1,e+1,n+1)
    land=transform(fw.transform,unary_union([g.intersection(region) for g in shapereader.Reader(landfile).geometries() if g.intersects(region)]))
    landmask=np.zeros((ny,nx),dtype=bool)
    for row in range(0,ny,100):
        gx,gy=np.meshgrid(xmin+(np.arange(nx)+.5)*h,ymin+(np.arange(row,min(ny,row+100))+.5)*h)
        landmask[row:row+len(gy)]=contains_xy(land,gx,gy)
    # Zero padding deliberately underestimates inland distance at the study boundary.
    depth=distance_transform_edt(np.pad(landmask,1),sampling=h)[1:-1,1:-1]
    tail=None;made=0
    query='SELECT * FROM observations ORDER BY mmsi,timestamp,record_id'
    for raw in pd.read_sql_query(query,db,chunksize=a.batch_size):
        count=len(raw);d=pd.concat([tail,raw],ignore_index=True) if tail is not None else raw.copy()
        tail=raw.iloc[-1:].copy()
        # SQLite batches containing only NULL values can have object dtype.
        # Normalize after concatenating the preceding row, preserving missingness.
        for column in ('lat','lon','sog','cog','heading'):
            d[column]=pd.to_numeric(d[column],errors='coerce').astype('float64')
        ts=pd.to_datetime(d.timestamp);x,y=fw.transform(d.lon.to_numpy(),d.lat.to_numpy())
        same=d.mmsi.eq(d.mmsi.shift());dt=ts.diff().dt.total_seconds()
        good=same&dt.gt(0)&dt.le(a.gap_minutes*60)
        dx=pd.Series(x).diff();dy=pd.Series(y).diff()
        implied=(np.hypot(dx,dy)/dt*3600/1852).where(good)
        sog=d.sog.where(d.sog.ge(0)&d.sog.lt(102.3));cog=d.cog.where(d.cog.ge(0)&d.cog.lt(360));heading=d.heading.where(d.heading.ge(0)&d.heading.lt(360)&d.heading.mod(1).eq(0))
        angle=lambda v:np.deg2rad(v.to_numpy(dtype=np.float64,na_value=np.nan))
        diff=((heading-cog+180)%360-180).abs()
        turn=(((cog-cog.shift()+180)%360-180).abs()/(dt/60)).where(good&sog.ge(1)&sog.shift().ge(1))
        acc=((sog-sog.shift())/(dt/60)).where(good)
        on=contains_xy(land,x,y)
        ix=np.clip(((x-xmin)/h).astype(int),0,nx-1);iy=np.clip(((y-ymin)/h).astype(int),0,ny-1)
        inland=np.where(on,np.maximum(0,depth[iy,ix]-np.sqrt(2)*h),0)
        rule=on&(inland>=a.land_margin_metres)&((sog.fillna(0).to_numpy()>2)|(implied.fillna(0).to_numpy()>2))
        hour=ts.dt.hour+ts.dt.minute/60
        vals={'x_km':(x-np.mean(xx))/1000,'y_km':(y-np.mean(yy))/1000,'sog':sog,
            'cog_sin':np.sin(angle(cog)),'cog_cos':np.cos(angle(cog)),
            'heading_sin':np.sin(angle(heading)),'heading_cos':np.cos(angle(heading)),
            'heading_cog_difference':diff,'log_gap_seconds':np.log1p(dt.where(same&dt.gt(0))),
            'implied_speed':implied.clip(0,500),'speed_disagreement':(implied-sog).abs().clip(0,500),
            'acceleration_knots_min':acc.clip(-100,100),'turn_degrees_min':turn.clip(0,360),
            'hour_sin':np.sin(hour*2*np.pi/24),'hour_cos':np.cos(hour*2*np.pi/24),
            'on_land':on.astype(float),'log_inland_metres':np.log1p(inland),
            'sog_missing':sog.isna().astype(float),'cog_missing':cog.isna().astype(float),
            'heading_missing':heading.isna().astype(float),'temporal_missing':(~good).astype(float)}
        feature=pd.DataFrame(vals).replace([np.inf,-np.inf],np.nan).fillna(0)
        for g in TYPES:feature['type_'+g]=d.vessel_type.eq(g).astype(float).to_numpy()
        meta=d[['record_id','timestamp','mmsi','lat','lon','vessel_type','source_file','source_row']].reset_index(drop=True)
        meta['land_motion_rule']=rule.astype(int)
        output=pd.concat([meta,feature],axis=1).iloc[-count:]
        output.to_sql('features',db,if_exists='append',index=False,chunksize=2000)
        made+=count;logging.info('Engineered features: %s',f'{made:,}')
    db.execute('CREATE UNIQUE INDEX feature_id ON features(record_id)');db.commit()
    return {'bounds':list(bounds),'projection':'EPSG:3035','land_grid_metres':h}


def fit(a):
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new/empty output directory')
    files=[a.input] if a.input.is_file() else sorted(f for f in a.input.rglob('*') if f.is_file() and (f.name.endswith('.csv') or f.name.endswith('.csv.gz')))
    if not files:raise ValueError('No input CSV/GZ files')
    a.output.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(a.output/'features.sqlite')
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-65536')
    db.execute('CREATE TABLE observations(record_id INTEGER PRIMARY KEY, timestamp TEXT,mmsi TEXT,lat REAL,lon REAL,sog REAL,cog REAL,heading REAL,vessel_type TEXT,source_file TEXT,source_row INTEGER)')
    counts=Counter()
    for f in files:
        logging.info('Reading %s',f)
        for d in input_chunks(f,a.batch_size):
            counts['input_rows']+=len(d)
            valid=d.timestamp.notna()&d.lat.between(-90,90)&d.lon.between(-180,180)&d.mmsi.str.fullmatch(r'\d{9}',na=False)&d.mmsi.ne('000000000')
            if 'mobile' in d:valid &= d.mobile.str.strip().str.lower().isin(['class a','class b'])
            counts['excluded_invalid_or_nonvessel']+=int((~valid).sum());d=d.loc[valid].copy()
            d.timestamp=d.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%S');d['source_file']=str(f.resolve())
            d[['timestamp','mmsi','lat','lon','sog','cog','heading','vessel_type','source_file','source_row']].to_sql('observations',db,if_exists='append',index=False,chunksize=2000)
    total=db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
    if total<20:raise ValueError('Need at least 20 valid observations')
    db.execute('CREATE INDEX obs_time ON observations(mmsi,timestamp)');db.commit()
    geography=make_features(db,a)
    db.execute('DROP TABLE observations');db.commit() # SQLite reuses pages; no costly VACUUM.
    scaler=StandardScaler();sample=None;keys=None;rng=np.random.default_rng(a.seed)
    logging.info('Fitting numeric scaling on ALL %s valid records',f'{total:,}')
    for d in read_features(db,a.batch_size):
        x=matrix(d);scaler.partial_fit(x[:,:len(NUM)])
        new_keys=rng.random(len(d))
        sample=d if sample is None else pd.concat([sample,d],ignore_index=True)
        keys=new_keys if keys is None else np.r_[keys,new_keys]
        if len(sample)>a.visual_sample:
            keep=np.argpartition(keys,a.visual_sample-1)[:a.visual_sample];sample=sample.iloc[keep].reset_index(drop=True);keys=keys[keep]
    xs=scaled(matrix(sample),scaler)
    kernel=Nystroem(kernel='rbf',gamma=a.gamma,n_components=min(a.components,len(sample)),random_state=a.seed)
    kernel.fit(xs)
    model=SGDOneClassSVM(nu=a.nu,random_state=a.seed,shuffle=True,average=True,learning_rate='optimal',tol=None)
    visits=0
    for epoch in range(a.epochs):
        # Disk-backed randomized ordering, not a RAM shuffle of the entire month.
        order=f'((record_id * 1103515245 + {a.seed+epoch*12345}) % 2147483647)'
        for d in read_features(db,a.batch_size,order):
            z=kernel.transform(scaled(matrix(d),scaler));model.partial_fit(z);visits+=len(d)
        logging.info('Epoch %s/%s complete; all %s records used',epoch+1,a.epochs,f'{total:,}')
    assert visits==total*a.epochs
    pca=PCA(n_components=min(10,len(FEATURES),len(sample)),random_state=a.seed).fit(xs)
    package={'scaler':scaler,'kernel':kernel,'model':model,'pca':pca,'features':FEATURES}
    joblib.dump(package,a.output/'model.joblib');sample.to_csv(a.output/'visual_sample.csv.gz',index=False)
    pd.DataFrame(pca.components_.T,index=FEATURES,columns=[f'PC{i+1}' for i in range(len(pca.components_))]).to_csv(a.output/'pca_loadings.csv')
    logging.info('Scoring every observation')
    for d in read_features(db,a.batch_size):
        z=scaled(matrix(d),scaler)
        out=d[META].copy();out['ml_score']=-model.decision_function(kernel.transform(z))
        out.to_sql('scores',db,if_exists='append',index=False,chunksize=2000)
    db.execute('CREATE UNIQUE INDEX score_id ON scores(record_id)')
    db.execute('CREATE INDEX type_scores ON scores(vessel_type,ml_score)')
    db.execute('CREATE TABLE reviews(record_id INTEGER PRIMARY KEY,label TEXT,notes TEXT)');db.commit()
    # Exact in-sample quantiles per type make type imbalance visible, without
    # declaring every unknown-type vessel anomalous. This is a review budget.
    thresholds={}
    for g,number in db.execute('SELECT vessel_type,count(*) FROM scores GROUP BY vessel_type').fetchall():
        offset=max(0,int(math.ceil((1-a.alert_fraction)*number))-1)
        threshold=db.execute('SELECT ml_score FROM scores WHERE vessel_type=? ORDER BY ml_score LIMIT 1 OFFSET ?',(g,offset)).fetchone()[0]
        thresholds[g]=float(threshold)
    config={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'counts':dict(counts),
            'valid_training_rows':total,'training_row_visits':visits,'thresholds':thresholds,'geography':geography,
            'sample_rows':len(sample),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
            'notes':['Every valid record trains the one-class model in each epoch. Only kernel landmarks/PCA plots use the bounded random sample.',
            'This is full-data retrospective discovery. No accuracy or false-positive rate is established.',
            'Alert fraction is an in-sample per-type review budget; unknown is a category, not a rule violation.',
            'Natural Earth coastline uncertainty remains. Land rule is separate from ML and requires movement plus conservative inland margin.',
            'Temporal features reset on vessel change/nonpositive gaps/gaps above configured limit. Same-timestamp conflicts should be cleaned upstream.',
            'Numeric missing values use zero plus missing indicators; temporal derived gaps are reported separately. Extreme derived values are clipped.',
            'Record IDs are local to this run. Apply feedback only against this features.sqlite; do not reuse IDs in a newly fitted run.',
            'Input duplicates are retained; use the cleaned input. Rare types can have unreliable thresholds.',
            'Feature scaling and model can absorb frequent errors in unlabeled data; land rules and expert review complement statistical rarity.',
            'PCA plots show a random subset and are not used for anomaly scoring. Two-dimensional proximity is not proof of normality.']}
    (a.output/'config.json').write_text(json.dumps(config,indent=2));db.close()
    export(a.output,a.output/'results',a.batch_size)


def export(run,out,batch_size,expert=None):
    out.mkdir(parents=True,exist_ok=False);db=sqlite3.connect(run/'features.sqlite')
    config=json.loads((run/'config.json').read_text());thresholds=config['thresholds']
    db.execute('DROP TABLE IF EXISTS ranked')
    reviews=pd.read_sql_query('SELECT * FROM reviews',db);review_map=dict(zip(reviews.record_id,reviews.label))
    stats=[]
    import gzip
    with gzip.open(out/'scored_ais.csv.gz','wt',newline='') as handle:
        header=True
        for d in pd.read_sql_query('SELECT f.*,s.ml_score FROM features f JOIN scores s USING(record_id) ORDER BY f.record_id',db,chunksize=batch_size):
            view=d[META+['ml_score','sog','implied_speed','speed_disagreement','on_land','temporal_missing','heading_cog_difference']].copy()
            view['inland_distance_lower_bound_m']=np.expm1(d.log_inland_metres)
            view['ml_flag']=view.ml_score>view.vessel_type.map(thresholds)
            view['review_label']=view.record_id.map(review_map).fillna('')
            view['expert_score']=expert.predict_proba(matrix(d))[:,list(expert.classes_).index(1)] if expert is not None else np.nan
            view['candidate']=view.ml_flag|view.land_motion_rule.eq(1)
            # Expert model proposes candidates separately; its score is uncalibrated.
            if expert is not None:view['candidate']|=view.expert_score.ge(.8)
            view.loc[view.review_label=='normal','candidate']=False
            view.loc[view.review_label=='anomaly','candidate']=True
            view.to_csv(handle,index=False,header=header);header=False
            view.to_sql('ranked',db,if_exists='append',index=False,chunksize=2000)
    db.execute('CREATE INDEX rank_idx ON ranked(vessel_type,ml_score)');db.commit()
    # A mix of candidates and controls prevents feedback from containing anomalies only.
    selected=[]
    for g in TYPES:
        d=pd.read_sql_query("SELECT * FROM ranked WHERE vessel_type=? AND review_label='' ORDER BY candidate DESC,land_motion_rule DESC,expert_score DESC,ml_score DESC LIMIT 1000",db,params=(g,))
        if len(d):
            d['hour']=d.timestamp.str[:13]
            d=d.drop_duplicates(['mmsi','hour']).head(25).drop(columns='hour');d['selection_reason']='ranked_candidate';selected.append(d)
    controls=pd.read_sql_query("SELECT * FROM ranked WHERE review_label='' ORDER BY ((record_id*1103515245)%2147483647) LIMIT 80",db)
    controls['selection_reason']='comparison_sample';selected.append(controls)
    queue=pd.concat(selected,ignore_index=True).drop_duplicates('record_id')
    queue['feedback_label']='';queue['feedback_notes']='';queue.to_csv(out/'review_queue.csv',index=False)
    bytype=pd.read_sql_query('SELECT vessel_type,count(*) AS observations,sum(ml_flag) AS ml_flags,sum(land_motion_rule) AS land_motion_flags,sum(candidate) AS combined_candidates FROM ranked GROUP BY vessel_type',db)
    bytype.to_csv(out/'summary_by_type.csv',index=False)
    reviewed=pd.read_sql_query('SELECT * FROM reviews',db);reviewed.to_csv(out/'stored_feedback.csv',index=False)
    for g in bytype.vessel_type:
        if int(bytype.loc[bytype.vessel_type==g,'observations'].iloc[0])<100:logging.warning('Type %s has fewer than 100 observations; interpret threshold cautiously',g)
    package=joblib.load(run/'model.joblib');sample=pd.read_csv(run/'visual_sample.csv.gz')
    scores=pd.read_sql_query('SELECT record_id,ml_score FROM scores',db) if config['valid_training_rows']<=50000 else None
    if scores is None:
        # Bounded parameter batches for sample IDs.
        parts=[];ids=sample.record_id.astype(int).tolist()
        for i in range(0,len(ids),500):parts.append(pd.read_sql_query('SELECT record_id,ml_score FROM scores WHERE record_id IN ('+','.join('?' for _ in ids[i:i+500])+')',db,params=ids[i:i+500]))
        scores=pd.concat(parts,ignore_index=True)
    sample=sample.merge(scores,on='record_id',validate='one_to_one')
    pc=package['pca'].transform(scaled(matrix(sample),package['scaler']))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for g in TYPES:
        m=sample.vessel_type.eq(g).to_numpy()
        axes[0].scatter(pc[m,0],pc[m,1],s=5,alpha=.45,label=g)
    axes[0].legend(markerscale=2,fontsize=8);axes[0].set_title('PCA by reported vessel type')
    im=axes[1].scatter(pc[:,0],pc[:,1],s=5,c=sample.ml_score,cmap='magma',alpha=.6)
    fig.colorbar(im,ax=axes[1],label='Unsupervised anomaly score');axes[1].set_title('Same PCA, coloured by ML score')
    for ax in axes:ax.set(xlabel='PC1',ylabel='PC2')
    fig.savefig(out/'pca_features.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for g in TYPES:
        values=sample.loc[sample.vessel_type==g,'ml_score']
        if len(values)>1:axes[0].hist(values,bins=30,histtype='step',density=True,label=g)
    axes[0].set(title='Sampled score distributions',xlabel='ML score',ylabel='Density');axes[0].legend(fontsize=8)
    axes[1].bar(bytype.vessel_type,bytype.combined_candidates/ bytype.observations*100)
    axes[1].set(title='Candidates by vessel type (all records)',ylabel='% observations');axes[1].tick_params(axis='x',rotation=45)
    fig.savefig(out/'score_distributions.png',dpi=180);plt.close(fig)
    # Lightweight review map; sample background plus selected candidates.
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig,ax=plt.subplots(figsize=(9,8),subplot_kw={'projection':ccrs.PlateCarree()},layout='constrained')
    w,e,s,n=config['geography']['bounds'];dx=max(.01,(e-w)*.05);dy=max(.01,(n-s)*.05)
    ax.set_extent([w-dx,e+dx,s-dy,n+dy]);ax.set_facecolor('#edf3f6')
    ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='#e6e5df',edgecolor='#aaaaaa',linewidth=.4)
    ax.scatter(sample.lon,sample.lat,s=1,color='#3f7898',alpha=.12,transform=ccrs.PlateCarree())
    chosen=queue[queue.candidate.astype(bool)]
    ax.scatter(chosen.lon,chosen.lat,s=10,color='#d13c3c',transform=ccrs.PlateCarree(),label='Review candidates')
    ax.legend(loc='upper right');ax.set_title('Review candidates over sampled reference observations')
    gl=ax.gridlines(draw_labels=True,alpha=.2);gl.top_labels=False;gl.right_labels=False
    fig.savefig(out/'review_map.png',dpi=180);plt.close(fig)
    (out/'README.txt').write_text('Edit feedback_label in review_queue.csv: normal, anomaly, uncertain. Add optional feedback_notes.\nThen run: python ais_ml_detector.py feedback --run '+str(run)+' --labels '+str(out/'review_queue.csv')+'\nLabels persist in features.sqlite. Repeat the feedback command with subsequent queues.\nExpert scores are uncalibrated preferences, not probabilities of wrongdoing. Model and land-rule scores remain separate.\nReview short tracks around candidates, not isolated dots alone. The comparison sample includes unflagged observations.\n')
    db.close();logging.info('Saved review queue, complete scores and plots: %s',out)


def feedback(a):
    db=sqlite3.connect(a.run/'features.sqlite');labels=pd.read_csv(a.labels,dtype={'record_id':str}).fillna('')
    if not {'record_id','feedback_label'}<=set(labels):raise ValueError('Need record_id and feedback_label columns')
    labels.feedback_label=labels.feedback_label.str.strip().str.lower();labels=labels[labels.feedback_label!='']
    if labels.empty:raise ValueError('No labels entered')
    if not labels.feedback_label.isin(['normal','anomaly','uncertain']).all():raise ValueError('Labels must be normal/anomaly/uncertain')
    labels['record_id']=pd.to_numeric(labels.record_id,errors='raise').astype(int)
    if labels.record_id.duplicated().any():raise ValueError('Duplicate record IDs in feedback file')
    with db:
        for row in labels.to_dict('records'):
            if db.execute('SELECT 1 FROM features WHERE record_id=?',(row['record_id'],)).fetchone() is None:raise ValueError('Feedback ID does not exist in this run')
            db.execute('INSERT OR REPLACE INTO reviews VALUES(?,?,?)',(row['record_id'],row['feedback_label'],str(row.get('feedback_notes',''))))
    train=pd.read_sql_query("SELECT f.*,r.label FROM features f JOIN reviews r USING(record_id) WHERE r.label IN ('normal','anomaly')",db)
    counts=train.label.value_counts().to_dict();expert=None
    if min(counts.get('normal',0),counts.get('anomaly',0))>=5:
        expert=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=2,class_weight='balanced',n_jobs=a.threads,random_state=42)
        expert.fit(matrix(train),train.label.eq('anomaly').astype(int))
        joblib.dump(expert,a.run/'expert_model.joblib')
        logging.info('Expert model fitted on %s reviews; scores are not calibrated or independently evaluated',len(train))
    else:logging.info('Stored reviews and applied exact overrides. Need at least 5 normal and 5 anomaly reviews for expert generalization; 20+ of each is preferable.')
    db.close()
    out=a.run/('feedback_'+time.strftime('%Y%m%d_%H%M%S'))
    export(a.run,out,a.batch_size,expert)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit');f.add_argument('--input',required=True,type=Path);f.add_argument('--output',required=True,type=Path)
    f.add_argument('--epochs',type=int,default=3);f.add_argument('--components',type=int,default=64)
    f.add_argument('--gamma',type=float,default=.03);f.add_argument('--nu',type=float,default=.02)
    f.add_argument('--alert-fraction',type=float,default=.01);f.add_argument('--visual-sample',type=int,default=10000)
    f.add_argument('--gap-minutes',type=float,default=30);f.add_argument('--land-grid-metres',type=float,default=100)
    f.add_argument('--land-margin-metres',type=float,default=200);f.add_argument('--seed',type=int,default=42)
    r=sub.add_parser('feedback');r.add_argument('--run',required=True,type=Path);r.add_argument('--labels',required=True,type=Path)
    for parser in [f,r]:parser.add_argument('--batch-size',type=int,default=10000);parser.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.batch_size<1 or a.threads<1:p.error('Batch size/threads must be positive')
    if a.command=='fit':
        if min(a.epochs,a.components,a.gamma,a.visual_sample,a.gap_minutes,a.land_grid_metres,a.land_margin_metres)<=0 or a.visual_sample<20 or not 0<a.nu<1 or not 0<a.alert_fraction<1:p.error('Invalid training settings')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    with threadpool_limits(limits=a.threads):
        if a.command=='fit':fit(a)
        else:feedback(a)

if __name__=='__main__':main()
