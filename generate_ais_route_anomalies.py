#!/usr/bin/env python3
"""Generate synthetic curved vessel routes in water away from observed AIS positions.
Install: python -m pip install pandas numpy scipy matplotlib cartopy shapely pyproj
Example:
 python generate_ais_route_anomalies.py --input copenhagen_january/clean_ais.csv.gz --output synthetic_routes
Accepts Danish AIS and internal analysis headers; CSV or CSV.GZ.
No input edits. This is an exploratory spatial-anomaly generator, not a navigation tool.
Reference occupancy uses ALL valid input positions (including jump-flagged records).
No observations near a route does not prove no real vessel used it: gaps/coverage matter.
Does not model bathymetry, navigation rules, port access, or vessel dynamics beyond
constant segment speed on smooth projected curves. No detector trained.
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from pyproj import CRS, Transformer, Geod
from shapely.geometry import box, LineString
from shapely.ops import transform, unary_union
from shapely import contains_xy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_reference_chunks(path, chunksize):
    """Read Danish AIS or internal headers and parse timestamps unambiguously."""
    header = pd.read_csv(path, nrows=0).columns
    aliases = {'latitude': 'lat', 'longitude': 'lon'}
    mapping = {}
    for column in header:
        key = column.strip().lstrip('#').strip().lower()
        key = aliases.get(key, key)
        if key in {'lat', 'lon', 'timestamp'}:
            if key in mapping.values():
                raise ValueError(f'Ambiguous columns for {key}')
            mapping[column] = key
    missing = {'lat', 'lon', 'timestamp'} - set(mapping.values())
    if missing:
        raise ValueError(f'Missing coordinate/time columns: {missing}')
    for frame in pd.read_csv(path, usecols=list(mapping), dtype=str, chunksize=chunksize):
        frame = frame.rename(columns=mapping)
        raw = frame.timestamp.str.strip()
        danish = raw.str.match(r'^\d{1,2}/\d{1,2}/\d{4}', na=False)
        parsed = pd.to_datetime(raw.where(~danish), format='mixed', errors='coerce')
        parsed.loc[danish] = pd.to_datetime(raw.loc[danish], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        frame['timestamp'] = parsed
        yield frame


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--n-routes',type=int,default=3)
    p.add_argument('--min-distance-m',type=float,default=500,help='Minimum conservative clearance from reference observations')
    p.add_argument('--min-length-km',type=float,default=3)
    p.add_argument('--max-length-km',type=float,default=12)
    p.add_argument('--speed-knots',type=float,default=12)
    p.add_argument('--step-seconds',type=float,default=60)
    p.add_argument('--grid-metres',type=float,default=100)
    p.add_argument('--coast-buffer-m',type=float,default=200)
    p.add_argument('--roi',nargs=4,type=float,metavar=('WEST','EAST','SOUTH','NORTH'))
    p.add_argument('--shape',choices=['mixed','detour','s_bend','meander'],default='mixed',help='Mixed cycles through three distinct shapes')
    p.add_argument('--min-turn-radius-m',type=float,default=150,help='Minimum radius of dense curve; illustrative, not vessel-specific')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attempts',type=int,default=30000)
    p.add_argument('--chunksize',type=int,default=200000)
    a=p.parse_args()
    for key in ['n_routes','min_distance_m','min_length_km','max_length_km','speed_knots','step_seconds','grid_metres','attempts','chunksize','min_turn_radius_m']:
        if getattr(a,key)<=0:p.error(f'{key} must be positive')
    if a.coast_buffer_m<0 or a.min_length_km>a.max_length_km:p.error('Invalid coast buffer or length range')
    if not a.input.is_file():p.error('Input not found')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be new or empty')
    roi=a.roi
    if roi is None:
        report=a.input.parent/'report.json'
        if report.exists():
            settings=json.loads(report.read_text())['parameters']
            roi=[float(settings[k]) for k in ['west','east','south','north']]
        else:roi=[12.35,12.85,55.50,55.90]
    w,e,s,n=roi
    if not(-180<w<e<180 and -85<s<n<85):p.error('Invalid ROI')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    crs=CRS.from_proj4(f'+proj=aeqd +lat_0={(s+n)/2} +lon_0={(w+e)/2} +datum=WGS84 +units=m')
    fw=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    inv=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
    # Densify rectangular boundary before projection.
    lon=np.r_[np.linspace(w,e,100),np.full(100,e),np.linspace(e,w,100),np.full(100,w)]
    lat=np.r_[np.full(100,s),np.linspace(s,n,100),np.full(100,n),np.linspace(n,s,100)]
    xx,yy=fw.transform(lon,lat);h=a.grid_metres
    xe=np.arange(min(xx)-h,max(xx)+2*h,h);ye=np.arange(min(yy)-h,max(yy)+2*h,h)
    if (len(xe)-1)*(len(ye)-1)>4_000_000:p.error('Grid too large: increase --grid-metres')
    counts=np.zeros((len(ye)-1,len(xe)-1),dtype=np.int64)
    total=used=0;start=None
    for d in read_reference_chunks(a.input,a.chunksize):
        total+=len(d);la=pd.to_numeric(d.lat,errors='coerce');lo=pd.to_numeric(d.lon,errors='coerce')
        ok=la.between(s,n)&lo.between(w,e)
        x,y=fw.transform(lo[ok].to_numpy(),la[ok].to_numpy())
        counts+=np.histogram2d(y,x,bins=[ye,xe])[0].astype(np.int64);used+=int(ok.sum())
        ts=pd.to_datetime(d.loc[ok,'timestamp'],errors='coerce').dropna()
        if len(ts):start=ts.min() if start is None else min(start,ts.min())
        logging.info('Read %s rows; %s reference positions inside ROI',f'{total:,}',f'{used:,}')
    if not used or start is None:p.error('No valid reference positions/timestamps inside ROI')
    logging.info('Preparing land mask (Natural Earth download on first use)')
    landfile=shapereader.natural_earth(resolution='10m',category='physical',name='land')
    region=box(w-.1,s-.1,e+.1,n+.1)
    geoms=[g.intersection(region) for g in shapereader.Reader(landfile).geometries() if g.intersects(region)]
    land=transform(fw.transform,unary_union(geoms)).buffer(a.coast_buffer_m)
    X,Y=np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2)
    LON,LAT=inv.transform(X,Y)
    inside=(LON>w)&(LON<e)&(LAT>s)&(LAT<n)
    clear=distance_transform_edt(counts==0,sampling=h)
    # Conservative correction: positions may lie anywhere inside their occupied cell.
    # Two half diagonals bound query-to-cell and reference-to-cell distances.
    correction=np.sqrt(2)*h
    safe=inside&(~contains_xy(land,X,Y))&(clear>=a.min_distance_m+correction+h/4)
    cells=np.argwhere(safe)
    if len(cells)<2:p.error('No eligible water cells. Reduce clearance or choose another ROI; constraints were not relaxed.')
    rng=np.random.default_rng(a.seed);routes=[];lines=[];meta=[];geod=Geod(ellps='WGS84')
    def clearance(points):
        ix=np.floor((points[:,0]-xe[0])/h).astype(int);iy=np.floor((points[:,1]-ye[0])/h).astype(int)
        if np.any(ix<0)|np.any(ix>=counts.shape[1])|np.any(iy<0)|np.any(iy>=counts.shape[0]):return -np.inf
        return float(np.min(clear[iy,ix])-correction)
    for attempt in range(a.attempts):
        pair=cells[rng.integers(0,len(cells),size=2)]
        pts=np.array([[X[r,c],Y[r,c]] for r,c in pair])
        chord=float(np.linalg.norm(pts[1]-pts[0]))
        if chord<500 or chord>a.max_length_km*1000:continue
        shape=['detour','s_bend','meander'][len(routes)%3] if a.shape=='mixed' else a.shape
        direction=(pts[1]-pts[0])/chord
        normal=np.array([-direction[1],direction[0]])
        # Smooth analytic curves with randomized amplitude and orientation.
        t=np.linspace(0,1,4001)
        amplitude=chord*rng.uniform(.18,.35)*rng.choice([-1,1])
        if shape=='detour': offset=amplitude*np.sin(np.pi*t)**2
        elif shape=='s_bend': offset=amplitude*np.sin(2*np.pi*t)*np.sin(np.pi*t)
        else: offset=amplitude*.65*np.sin(4*np.pi*t)*np.sin(np.pi*t)
        curve=pts[0]+t[:,None]*(pts[1]-pts[0])+offset[:,None]*normal
        cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        length=float(cumulative[-1])
        if not a.min_length_km*1000<=length<=a.max_length_km*1000:continue
        # Reject excessively tight turns before exporting any candidate.
        v=np.gradient(curve,t,axis=0); acc=np.gradient(v,t,axis=0)
        curvature=np.abs(v[:,0]*acc[:,1]-v[:,1]*acc[:,0])/np.linalg.norm(v,axis=1)**3
        radius=1/max(float(curvature.max()),1e-12)
        if radius<a.min_turn_radius_m:continue
        def at_distance(d):
            return np.column_stack([np.interp(d,cumulative,curve[:,k]) for k in range(2)])
        dense=at_distance(np.linspace(0,length,max(2,int(np.ceil(length/(h/4)))+1)))
        line=LineString(dense)
        if line.intersects(land):continue
        if any(line.distance(old)<a.min_distance_m for old in lines):continue
        dl,da=inv.transform(dense[:,0],dense[:,1])
        if not ((dl>w)&(dl<e)&(da>s)&(da<n)).all():continue
        lower=clearance(dense)-h/4
        if lower<a.min_distance_m:continue
        speed=a.speed_knots*1852/3600
        # Cap point spacing so plotted/exported chords preserve the checked curve.
        spacing=min(speed*a.step_seconds,h/4)
        dist=np.unique(np.r_[np.arange(0,length,spacing),length])
        xy=at_distance(dist)
        exported_line=LineString(xy)
        if exported_line.intersects(land):continue
        export_lower=clearance(xy)-spacing
        if export_lower<a.min_distance_m:continue
        lower=min(lower,export_lower)
        lons,lats=inv.transform(xy[:,0],xy[:,1])
        bearing,_,metres=geod.inv(lons[:-1],lats[:-1],lons[1:],lats[1:])
        seconds=np.r_[0,np.cumsum(metres/speed)]
        route_id=f'SYNTHETIC_ROUTE_{len(routes)+1:03d}'
        df=pd.DataFrame({'route_id':route_id,'route_shape':shape,'timestamp':(start+pd.to_timedelta(seconds,unit='s')).astype(str),
                         'latitude':lats,'longitude':lons,'sog':a.speed_knots,
                         'cog':np.r_[bearing%360,bearing[-1]%360],
                         'is_synthetic':True,'anomaly_type':'spatial_route_without_nearby_reference_observations'})
        routes.append(df);lines.append(exported_line)
        meta.append({'route_id':route_id,'route_shape':shape,'minimum_curve_radius_m':radius,'length_km':float(metres.sum()/1000),'points':len(df),
                     'conservative_reference_clearance_m':lower,'land_buffer_intersection':False})
        logging.info('Generated %s: %.2f km, clearance >= %.0f m',route_id,metres.sum()/1000,lower)
        if len(routes)==a.n_routes:break
    if len(routes)<a.n_routes:
        p.error(f'Found {len(routes)}/{a.n_routes} eligible routes after {a.attempts} attempts. Try fewer routes, a shorter minimum length, or smaller clearance. No constraints silently relaxed.')
    a.output.mkdir(parents=True,exist_ok=True)
    pd.concat(routes,ignore_index=True).to_csv(a.output/'synthetic_routes.csv',index=False)
    features=[{'type':'Feature','properties':m,'geometry':{'type':'LineString','coordinates':df[['longitude','latitude']].to_numpy().tolist()}} for df,m in zip(routes,meta)]
    (a.output/'synthetic_routes.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    projection=ccrs.PlateCarree();fig,axes=plt.subplots(1,3,figsize=(18,7),subplot_kw={'projection':projection},layout='constrained')
    edges_x,edges_y=np.meshgrid(xe,ye);edge_lon,edge_lat=inv.transform(edges_x,edges_y)
    mesh=None
    for i,ax in enumerate(axes):
        ax.set_extent(roi,crs=projection);ax.set_facecolor('#edf3f6')
        ax.add_feature(cfeature.LAND.with_scale('10m'),facecolor='#e6e5df',edgecolor='#a4aaa7',linewidth=.4,zorder=1)
        gl=ax.gridlines(draw_labels=True,alpha=.25,linewidth=.4);gl.top_labels=False;gl.right_labels=False
        if i<2:
            mesh=ax.pcolormesh(edge_lon,edge_lat,np.ma.masked_equal(counts,0),norm=LogNorm(vmin=1,vmax=max(2,int(counts.max()))),cmap='Blues',transform=projection,zorder=2)
        if i>0:
            for j,df in enumerate(routes):
                ax.plot(df.longitude,df.latitude,color=['#c62828','#ef6c00','#9c2763'][j%3],linewidth=2.2,transform=projection,zorder=4,label=f'{j+1}: {meta[j]["route_shape"].replace("_"," ")}')
                ax.scatter(df.longitude.iloc[0],df.latitude.iloc[0],s=30,c='#d23838',marker='o',transform=projection,zorder=5)
                ax.scatter(df.longitude.iloc[-1],df.latitude.iloc[-1],s=35,c='#d23838',marker='X',transform=projection,zorder=5)
            ax.legend(loc='upper left')
        ax.set_title(['Reference traffic','Reference + synthetic routes','Synthetic routes only'][i])
        ax.text(.99,.01,'Natural Earth',transform=ax.transAxes,ha='right',fontsize=7,color='#777777')
    fig.colorbar(mesh,ax=axes[:2],label='Reference AIS observations per cell (log)',shrink=.6)
    fig.suptitle(f'Synthetic spatial anomalies | ≥ {a.min_distance_m:g} m from reference positions\nCircles: start • crosses: end | reference data are not guaranteed anomaly-free',fontsize=13)
    fig.savefig(a.output/'normal_vs_anomalies.png',dpi=180,bbox_inches='tight');plt.close(fig)
    notes=['Unused means no nearby observations in the provided ROI and time period, not globally unused routes.',
           'Reference positions are not certified normal; flagged positions are included conservatively.',
           'Clearance uses a metric grid with conservative cell and sampling corrections; full line checked against buffered land.',
           'No reference-track interpolation: a vessel could cross empty space between recorded positions.',
           'Smooth detours, S-bends and meanders have constant segment speed and a configurable minimum curve radius. These shapes are illustrative, not validated vessel dynamics.',
           'Export spacing is capped at one quarter grid cell to preserve curve geometry; step-seconds is a maximum, with shorter intervals on output. Synthetic identities are not real MMSIs.',
           'Natural Earth is a general map, not a nautical chart; depth/restrictions/coverage are not checked.',
           'Keep synthetic data separate from training data. This generator is exploratory, not an independent test benchmark.']
    (a.output/'generation_report.json').write_text(json.dumps({'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'roi':roi,'reference_rows':used,'routes':meta,'notes':notes},indent=2))
    logging.info('Saved %s routes and comparison plot in %s',len(routes),a.output)

if __name__=='__main__':main()
