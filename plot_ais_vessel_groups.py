#!/usr/bin/env python3
"""Compare cargo/tankers with fishing in clean_ais.csv[.gz].
Accepts Danish AIS headers (including # Timestamp) and internal analysis headers.
Requires pandas, numpy, matplotlib, cartopy for the default soft basemap.
Natural Earth land/coastline data may download on first use.
Use --no-background for offline plots without Cartopy.
Two streaming passes; memory scales with vessels and vessel-days, not rows.
Histograms are exact fixed-bin counts, not random samples. No raw data edits.
"""
import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

GROUPS=['cargo_tanker','fishing']
LABEL={'cargo_tanker':'Cargo / tankers','fishing':'Fishing'}
COLOR={'cargo_tanker':'#2176ae','fishing':'#df7c28'}

def classify(value):
    s=str(value).strip().lower()
    try:
        n=float(s)
        if n.is_integer():
            n=int(n)
            if 70<=n<=89: return 'cargo_tanker'
            if n==30: return 'fishing'
            if n==0: return 'unknown'
            return 'other'
    except ValueError: pass
    if re.search(r'\bfishing\b',s): return 'fishing'
    if re.search(r'\b(cargo|tanker|tankers)\b',s): return 'cargo_tanker'
    if s in ('','nan','none','unknown','undefined','not available','unspecified'): return 'unknown'
    return 'other'

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path('vessel_group_plots'))
    p.add_argument('--chunksize',type=int,default=200000)
    p.add_argument('--roi',nargs=4,type=float,metavar=('WEST','EAST','SOUTH','NORTH'),help='Otherwise use adjacent report.json or full input extent')
    p.add_argument('--grid-size',type=int,default=100,help='Spatial bins per axis')
    p.add_argument('--moving-knots',type=float,default=1,help='Minimum SOG for direction plots')
    p.add_argument('--coastlines',action='store_true',help='Compatibility flag; soft background is now the default')
    p.add_argument('--no-background',action='store_true',help='Disable basemap (no Cartopy required)')
    a=p.parse_args()
    if a.chunksize<1 or not 2<=a.grid_size<=1000 or a.moving_knots<0:p.error('Invalid chunk/grid/movement setting')
    if not a.input.is_file():p.error('Input file not found')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be new or empty')
    projection=None
    if not a.no_background:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.io import shapereader
            projection=ccrs.PlateCarree()
            # Fetch/cache required map layers before the expensive CSV passes.
            for layer in ['land','coastline']:
                shapereader.natural_earth(resolution='10m',category='physical',name=layer)
        except ImportError:
            p.error('Soft background requires: python -m pip install cartopy (or use --no-background)')
        except Exception as exc:
            p.error(f'Cannot load Natural Earth background: {exc}. Connect once to cache map data, or use --no-background.')
    a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    header=pd.read_csv(a.input,nrows=0).columns
    required={'mmsi','timestamp','lat','lon','ship_type','sog','cog','heading'}
    aliases={'latitude':'lat','longitude':'lon','ship type':'ship_type'}
    mapping={}
    for column in header:
        key=column.strip().lstrip('#').strip().lower()
        key=aliases.get(key,key)
        if key in required|{'jump_flag'}:
            if key in mapping.values():p.error(f'Ambiguous columns for {key}')
            mapping[column]=key
    if not required<=set(mapping.values()):p.error(f'Missing columns: {required-set(mapping.values())}')
    has_jump_flags='jump_flag' in mapping.values()
    def chunks():
        for frame in pd.read_csv(a.input,usecols=list(mapping),dtype=str,chunksize=a.chunksize):
            frame=frame.rename(columns=mapping)
            raw=frame.timestamp.str.strip()
            # Danish exports use day/month/year; original analysis files use ISO.
            danish=raw.str.match(r'^\d{1,2}/\d{1,2}/\d{4}',na=False)
            parsed=pd.to_datetime(raw.where(~danish),format='mixed',errors='coerce')
            parsed.loc[danish]=pd.to_datetime(raw.loc[danish],format='%d/%m/%Y %H:%M:%S',errors='coerce')
            frame['timestamp']=parsed
            yield frame
    # Establish a consistent group per vessel; any known cross-group disagreement
    # (including other types) is excluded, never silently assigned by majority.
    types=defaultdict(Counter); raw_types=Counter(); bounds=[np.inf,-np.inf,np.inf,-np.inf]
    first=last=None; total=0
    for d in chunks():
        d['class']=d.ship_type.fillna('').map(classify)
        for (m,g),n in d.groupby(['mmsi','class']).size().items(): types[m][g]+=int(n)
        raw_types.update(d.ship_type.fillna('(missing)').astype(str))
        x=pd.to_numeric(d.lon,errors='coerce'); y=pd.to_numeric(d.lat,errors='coerce')
        if x.notna().any():bounds[0]=min(bounds[0],x.min());bounds[1]=max(bounds[1],x.max())
        if y.notna().any():bounds[2]=min(bounds[2],y.min());bounds[3]=max(bounds[3],y.max())
        t=pd.to_datetime(d.timestamp,errors='coerce').dropna()
        if len(t):first=t.min() if first is None else min(first,t.min());last=t.max() if last is None else max(last,t.max())
        total+=len(d);logging.info('Pass 1: %s rows',f'{total:,}')
    assignments={}; catalog=[]
    for m,c in types.items():
        known=set(c)-{'unknown'}
        group=next(iter(known)) if len(known)==1 else ('conflict' if len(known)>1 else 'unknown')
        assignments[m]=group
        catalog.append({'mmsi':m,'assigned_group':group,**{k:c[k] for k in GROUPS+['other','unknown']}})
    pd.DataFrame(catalog,columns=['mmsi','assigned_group']+GROUPS+['other','unknown']).to_csv(a.output/'vessel_classification.csv',index=False)
    pd.DataFrame([{'raw_ship_type':s,'mapped_group':classify(s if s!='(missing)' else ''),'observations':n} for s,n in raw_types.items()],columns=['raw_ship_type','mapped_group','observations']).to_csv(a.output/'ship_type_audit.csv',index=False)
    roi=a.roi
    if roi is None and (a.input.parent/'report.json').exists():
        try:
            settings=json.loads((a.input.parent/'report.json').read_text())['parameters']
            roi=[float(settings[k]) for k in ['west','east','south','north']]
        except (KeyError,ValueError,TypeError):pass
    if roi is None:roi=bounds
    if not np.isfinite(roi).all():p.error('No valid positions; supply --roi for an empty dataset')
    w,e,s,n=roi
    if w==e:w-=.01;e+=.01
    if s==n:s-=.01;n+=.01
    if not(-180<=w<e<=180 and -90<=s<n<=90):p.error('Invalid ROI')
    xb=np.linspace(w,e,a.grid_size+1);yb=np.linspace(s,n,a.grid_size+1)
    bins={'sog':np.arange(0,103,.5),'cog':np.arange(0,361,10),'heading':np.arange(0,361,10)}
    hist={g:{k:np.zeros(len(b)-1,dtype=np.int64) for k,b in bins.items()} for g in GROUPS}
    maps={g:np.zeros((a.grid_size,a.grid_size),dtype=np.int64) for g in GROUPS}
    daily=defaultdict(Counter); day_vessels=defaultdict(set); hourly={g:np.zeros(24,dtype=np.int64) for g in GROUPS}
    vessels=defaultdict(lambda:Counter()); counts={g:Counter() for g in GROUPS}; skipped=Counter()
    for i,d in enumerate(chunks(),1):
        d['group']=d.mmsi.map(assignments)
        skipped.update(d.loc[~d.group.isin(GROUPS),'group'])
        d=d[d.group.isin(GROUPS)].copy()
        for k in ['lat','lon','sog','cog','heading']:d[k]=pd.to_numeric(d[k],errors='coerce')
        t=pd.to_datetime(d.timestamp,errors='coerce');valid=t.notna()&d.lon.between(w,e)&d.lat.between(s,n)
        skipped['invalid_time_or_outside_plot_roi']+=int((~valid).sum());d=d.loc[valid].copy();t=t.loc[valid]
        d['day']=t.dt.strftime('%Y-%m-%d');d['hour']=t.dt.hour
        for g in GROUPS:
            z=d[d.group==g].copy();c=counts[g];c['observations']+=len(z)
            jump=pd.to_numeric(z.get('jump_flag',pd.Series(0,index=z.index)),errors='coerce').fillna(0).ne(0)
            c['jump_flagged_observations']+=int(jump.sum())
            maps[g]+=np.histogram2d(z.lat,z.lon,bins=[yb,xb])[0].astype(np.int64)
            for k in bins:
                v=z[k];ok=v.ge(0)&v.lt(102.3 if k=='sog' else 360)
                if k=='heading':ok &= v.mod(1).eq(0)
                c[k+'_missing_or_invalid']+=int((~ok).sum())
                if k!='sog':ok &= z.sog.ge(a.moving_knots)&z.sog.lt(102.3)
                hist[g][k]+=np.histogram(v[ok],bins[k])[0]
                c[k+'_plotted']+=int(ok.sum())
            ok=z.sog.ge(0)&z.sog.lt(102.3)
            c['low_speed_observations']+=int((ok&z.sog.lt(a.moving_knots)).sum())
            for day,ct in z.groupby('day').size().items():daily[g][day]+=int(ct)
            for day,m in z[['day','mmsi']].drop_duplicates().itertuples(index=False,name=None):day_vessels[(g,day)].add(m)
            hourly[g]+=np.bincount(z.hour.to_numpy(dtype=int),minlength=24)
            for m,zz in z.groupby('mmsi'):
                q=vessels[(g,m)];q['observations']+=len(zz)
                vv=zz.sog[zz.sog.ge(0)&zz.sog.lt(102.3)]
                q['valid_sog']+=len(vv);q['sog_sum']+=float(vv.sum());q['low_speed']+=int(vv.lt(a.moving_knots).sum())
        logging.info('Pass 2: chunk %s',i)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'figure.facecolor':'white'})
    def save(fig,name):
        fig.savefig(a.output/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    for normalized,name in [(False,'01_spatial_counts'),(True,'02_spatial_percentage')]:
        data={g:maps[g]*100/max(1,maps[g].sum()) if normalized else maps[g] for g in GROUPS}
        positive=np.concatenate([v[v>0] for v in data.values()]); vmax=float(positive.max()) if len(positive) else 1
        vmin=float(positive.min()) if len(positive) else 1;vmax=max(vmax,vmin*1.01)
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained',subplot_kw={'projection':projection} if projection else {})
        for ax,g in zip(axes,GROUPS):
            kw={'transform':projection} if projection else {}
            mesh=ax.pcolormesh(xb,yb,np.ma.masked_equal(data[g],0),cmap='viridis',norm=LogNorm(vmin=vmin,vmax=vmax),zorder=3,**kw)
            if projection:
                ax.set_extent([w,e,s,n],crs=projection)
                ax.set_facecolor('#edf3f6')
                ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='#e6e5df',edgecolor='none',zorder=1)
                ax.coastlines(resolution='10m',color='#a4aaa7',linewidth=.55,zorder=2)
                # Basemap stays below every observation, including on-land outliers.
                ax.text(.99,.01,'Natural Earth',transform=ax.transAxes,ha='right',va='bottom',fontsize=7,color='#777f85',zorder=4)
                gl=ax.gridlines(draw_labels=True,color='#abb8c0',linewidth=.45,alpha=.4,zorder=2);gl.top_labels=False;gl.right_labels=False
            else:
                ax.set(xlim=(w,e),ylim=(s,n),xlabel='Longitude',ylabel='Latitude');ax.set_aspect(1/np.cos(np.deg2rad((s+n)/2)))
            ax.set_title(f'{LABEL[g]} | {counts[g]["observations"]:,} observations')
            if not counts[g]['observations']:ax.text(.5,.5,'No observations',transform=ax.transAxes,ha='center')
        fig.colorbar(mesh,ax=axes,label='% of group observations per cell (log scale)' if normalized else 'Observations per cell (log scale)',shrink=.75)
        fig.suptitle('Spatial activity — observation counts, not vessel density or residence time')
        save(fig,name)
    # Within-group normalization keeps unequal sample sizes comparable.
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for g in GROUPS:
        h=hist[g]['sog'];den=max(1,h.sum());axes[0].stairs(h/den*100,bins['sog'],color=COLOR[g],label=LABEL[g]);axes[1].plot(bins['sog'][1:],np.cumsum(h)/den*100,color=COLOR[g],label=LABEL[g])
    axes[0].set(xlabel='SOG (knots)',ylabel='% of valid observations per 0.5-knot bin',title='Speed distribution',xlim=(0,102.3))
    axes[1].set(xlabel='SOG (knots)',ylabel='Cumulative % (binned)',title='Speed cumulative distribution',ylim=(0,100))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'03_speed_distribution')
    fig,axes=plt.subplots(1,2,figsize=(11,5),subplot_kw={'projection':'polar'},layout='constrained')
    for ax,k in zip(axes,['cog','heading']):
        for g in GROUPS:
            h=hist[g][k];v=h/max(1,h.sum())*100;centres=np.deg2rad((bins[k][:-1]+bins[k][1:])/2)
            ax.plot(np.r_[centres,centres[0]],np.r_[v,v[0]],color=COLOR[g],label=LABEL[g])
        ax.set_theta_zero_location('N');ax.set_theta_direction(-1);ax.set_title(f'{k.upper()} — % per 10° bin\nSOG ≥ {a.moving_knots:g} knots');ax.legend(loc='lower center',bbox_to_anchor=(.5,-.25))
    save(fig,'04_direction_distributions')
    days=pd.date_range(first.normalize(),last.normalize(),freq='D').strftime('%Y-%m-%d') if first is not None else []
    table=[];fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
    for g in GROUPS:
        obs=[daily[g][day] for day in days];nv=[len(day_vessels[(g,day)]) for day in days]
        axes[0].plot(pd.to_datetime(days),obs,label=LABEL[g],color=COLOR[g]);axes[1].plot(pd.to_datetime(days),nv,label=LABEL[g],color=COLOR[g])
        axes[2].plot(np.arange(24),hourly[g]/max(1,hourly[g].sum())*100,label=LABEL[g],color=COLOR[g])
        table.extend({'group':g,'date':day,'observations':o,'unique_vessels':v} for day,o,v in zip(days,obs,nv))
    for ax,title,ylabel in zip(axes,['Daily observations','Daily unique vessels','Time-of-day activity (source timestamp convention)'],['Observations','Vessels','% of group observations']):ax.set(title=title,ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
    axes[2].set(xlabel='Hour',xticks=range(0,24,2));save(fig,'05_activity')
    pd.DataFrame(table,columns=['group','date','observations','unique_vessels']).to_csv(a.output/'daily_activity.csv',index=False)
    vt=pd.DataFrame([{'group':g,'mmsi':m,'observations':v['observations'],'mean_sog':v['sog_sum']/v['valid_sog'] if v['valid_sog'] else np.nan,'low_speed_percent':v['low_speed']*100/v['valid_sog'] if v['valid_sog'] else np.nan} for (g,m),v in vessels.items()],columns=['group','mmsi','observations','mean_sog','low_speed_percent'])
    vt.to_csv(a.output/'vessel_statistics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for g in GROUPS:
        z=vt[vt.group==g]
        for ax,k,b in [(axes[0],'mean_sog',np.arange(0,104,2)),(axes[1],'low_speed_percent',np.arange(0,105,5))]:
            h,_=np.histogram(z[k].dropna(),b);ax.stairs(h/max(1,h.sum())*100,b,color=COLOR[g],label=LABEL[g])
    axes[0].set(title='Per-vessel mean reported speed',xlabel='Mean SOG (knots)',ylabel='% of vessels per bin')
    axes[1].set(title=f'Per-vessel low-speed share (SOG < {a.moving_knots:g} knots)',xlabel='% of valid observations at low speed',ylabel='% of vessels per bin')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'06_vessel_level_distributions')
    hrows=[]
    for g in GROUPS:
        if not has_jump_flags: counts[g]['jump_flagged_observations']=None
        counts[g]['unique_vessels']=int((vt.group==g).sum())
        for k,b in bins.items():
            for lo,hi,ct in zip(b[:-1],b[1:],hist[g][k]):hrows.append({'group':g,'variable':k,'bin_start':lo,'bin_end':hi,'count':int(ct),'percent':float(ct/max(1,hist[g][k].sum())*100)})
    pd.DataFrame(hrows).to_csv(a.output/'histogram_data.csv',index=False)
    notes=['Cargo and tanker combined; fishing separate. Numeric codes: 70–89 and 30; Danish text labels also recognized.',
           'Group assigned per MMSI from consistent known types over input. Unknown reports inherit that group. Any known group conflict excludes the vessel.',
           'Type inheritance is retrospective for exploration only; do not use future metadata in prospective evaluation.',
           'No behavioural observations removed. Jump-flagged positions remain included. If jump_flag is absent (Danish export), jump counts are null; clean_quality.csv.gz is not joined.',
           'Observation histograms and maps are affected by reporting rate. Vessel-level plots give each vessel equal weight, but its mean uses message weighting.',
           'Low speed is not evidence of fishing or anchoring; vessel type is not activity.',
           'COG and heading are circular and shown only when reported SOG meets moving threshold.',
           'No time weighting or interpolation. Other and unknown types excluded. Soft Natural Earth land/water background by default; --no-background disables it.',
           'Maps share scales within each figure; normalized map is percentage per group, not unique-vessel density.',
           'Time plots use input timestamp convention with no timezone conversion.']
    (a.output/'summary.json').write_text(json.dumps({'input':str(a.input),'input_rows':total,'roi':[w,e,s,n],'counts':counts,'excluded_rows':dict(skipped),'notes':notes},indent=2))
    (a.output/'README.txt').write_text('\n'.join(notes))
    logging.info('Done. Plots and CSV summaries: %s',a.output)
    for g in GROUPS:
        if not counts[g]['observations']:logging.warning('No %s observations; inspect ship_type_audit.csv',LABEL[g])

if __name__=='__main__':main()
