#!/usr/bin/env python3
"""Local trajectory reviewer for ais_ml_detector.py runs.
Install: python -m pip install 'streamlit>=1.54,<2' 'plotly>=5.24,<7' pandas numpy
Run: streamlit run review_ais_tracks.py -- --run ais_ml_run_fixed
Stop with Ctrl+C; saved reviews persist in RUN/trajectory_reviews.sqlite.
Uses indexed features joined to scores; never loads the full month.
Adds a lookup index to the detector database; no records or models are changed.
"""
import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CATEGORIES = ['Not specified', 'Position / data quality', 'Reported movement over land',
              'Unusual route / deviation', 'Unusual speed / acceleration',
              'Unusual turn / heading', 'Vessel-type inconsistency', 'Other']


def connect_source(run):
    return sqlite3.connect((run / 'features.sqlite').as_uri() + '?mode=ro', uri=True)


def read_track(run, mmsi, start, end):
    with connect_source(run) as db:
        d = pd.read_sql_query('''SELECT f.record_id,f.timestamp,f.mmsi,f.lat,f.lon,
            f.sog,f.sog_missing,f.cog_sin,f.cog_cos,f.cog_missing,
            f.heading_sin,f.heading_cos,f.heading_missing,f.vessel_type,f.implied_speed,f.temporal_missing,
            f.land_motion_rule,s.ml_score FROM features f
            JOIN scores s ON s.record_id=f.record_id
            WHERE f.mmsi=? AND f.timestamp>=? AND f.timestamp<=?
            ORDER BY f.timestamp,f.record_id LIMIT 30001''', db,
            params=(mmsi, start.isoformat(), end.isoformat()))
    d['timestamp'] = pd.to_datetime(d.timestamp)
    for c in ['cog','heading']:
        d[c] = (np.rad2deg(np.arctan2(d[c+'_sin'],d[c+'_cos'])) % 360).round(6) % 360
        d[c] = d[c].where(d[c+'_missing'].eq(0))
    d['sog'] = d.sog.where(d.sog_missing.eq(0))
    for c, maximum in [('sog', 102.3), ('cog', 360), ('heading', 360)]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
        d[c] = d[c].where(d[c].ge(0) & d[c].lt(maximum))
    d['implied_speed'] = d.implied_speed.where(d.temporal_missing.eq(0))
    return d


def review_db(run):
    db = sqlite3.connect(run / 'trajectory_reviews.sqlite')
    db.executescript('''CREATE TABLE IF NOT EXISTS events(
        event_id INTEGER PRIMARY KEY, created TEXT, mmsi TEXT, start TEXT, end TEXT,
        label TEXT, category TEXT, confidence TEXT, notes TEXT, scope TEXT, points INTEGER);
        CREATE TABLE IF NOT EXISTS event_points(event_id INTEGER, record_id INTEGER,
        PRIMARY KEY(event_id,record_id));''')
    return db


def save_review(db, selected, label, category, confidence, notes, scope):
    with db:
        cursor = db.execute('INSERT INTO events VALUES(NULL,?,?,?,?,?,?,?,?,?,?)',
            (datetime.now(timezone.utc).isoformat(), str(selected.mmsi.iloc[0]),
             selected.timestamp.min().isoformat(), selected.timestamp.max().isoformat(),
             label, category, confidence, notes, scope, len(selected)))
        db.executemany('INSERT INTO event_points VALUES(?,?)',
                      [(cursor.lastrowid, int(i)) for i in selected.record_id])
    return cursor.lastrowid


def exports(db):
    events = pd.read_sql_query('SELECT * FROM events ORDER BY event_id DESC', db)
    # Latest EXPLICIT point review wins. Segment judgments never become point labels.
    points = pd.read_sql_query('''SELECT p.record_id,e.label AS feedback_label,
        e.notes AS feedback_notes,e.category,e.confidence,e.event_id
        FROM event_points p JOIN events e USING(event_id)
        WHERE e.scope='every selected point' ORDER BY e.event_id''', db)
    return events, points.drop_duplicates('record_id', keep='last')


def map_figure(d, selected, show_scores, gap_minutes, style):
    fig = go.Figure()
    # Do not draw a straight connection across absent data or duplicate times.
    breaks = d.timestamp.diff().dt.total_seconds()
    breaks = breaks.le(0) | breaks.gt(gap_minutes * 60)
    lon, lat = [], []
    for row, split in zip(d.itertuples(), breaks):
        if split:
            lon.append(None); lat.append(None)
        lon.append(row.lon); lat.append(row.lat)
    fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode='lines',
        line=dict(color='#8190a0', width=2), name='Context track', hoverinfo='skip'))
    custom = np.column_stack([d.timestamp.astype(str), d.record_id.astype(str),
                               d.sog.round(2).astype(str), d.vessel_type.astype(str)])
    fig.add_trace(go.Scattermap(lon=d.lon, lat=d.lat, mode='markers',
        marker=dict(size=6, color='#6b829b'), name='Observations', customdata=custom,
        hovertemplate='%{customdata[0]}<br>ID %{customdata[1]}<br>SOG %{customdata[2]} kn<br>%{customdata[3]}<extra></extra>'))
    if show_scores:
        flags = d[d.flag]
        fig.add_trace(go.Scattermap(lon=flags.lon, lat=flags.lat, mode='markers',
            marker=dict(size=10, color='#de6060'), name='Detector candidates'))
    fig.add_trace(go.Scattermap(lon=selected.lon, lat=selected.lat, mode='markers',
        marker=dict(size=8, color='#e7a522'), name='Selected for review'))
    for row, name, color in [(d.iloc[0], 'Start', '#29906d'), (d.iloc[-1], 'End', '#61459c')]:
        fig.add_trace(go.Scattermap(lon=[row.lon], lat=[row.lat], mode='markers',
            marker=dict(size=14,color=color),name=name,text=[str(row.timestamp)],hoverinfo='text+name'))
    span = max(float(d.lon.max()-d.lon.min()), float(d.lat.max()-d.lat.min())*1.6, .005)
    zoom = float(np.clip(np.log2(360/span)-1.5, 2, 15))
    fig.update_layout(map=dict(style=style,center=dict(lon=float(d.lon.mean()),lat=float(d.lat.mean())),zoom=zoom),
        height=570,margin=dict(l=0,r=0,t=0,b=0),legend=dict(orientation='h'),
        uirevision=f'{d.mmsi.iloc[0]}-{d.timestamp.iloc[0]}-{d.timestamp.iloc[-1]}')
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    st.set_page_config(page_title='AIS trajectory review', layout='wide')
    st.title('AIS trajectory review')
    st.caption('Review movement in context. Unusual does not automatically mean incorrect or suspicious.')
    if not (run/'features.sqlite').is_file() or not (run/'config.json').is_file():
        st.error('Select a completed ais_ml_detector.py fit output folder.'); st.stop()
    config = json.loads((run/'config.json').read_text())
    with st.spinner('Preparing track lookup index (first opening may take a few minutes)…'):
        with sqlite3.connect(run/'features.sqlite',timeout=60) as source:
            source.execute('CREATE INDEX IF NOT EXISTS review_track_lookup ON features(mmsi,timestamp,record_id)')
    queues = sorted(run.glob('feedback_*/review_queue.csv'),reverse=True)
    if (run/'results/review_queue.csv').exists(): queues.append(run/'results/review_queue.csv')
    if not queues:
        st.error('No review queue found. Finish detector scoring first.'); st.stop()
    queue_path = st.sidebar.selectbox('Review queue',queues,format_func=lambda p:str(p.relative_to(run)))
    queue = pd.read_csv(queue_path, dtype={'mmsi':str})
    group = st.sidebar.selectbox('Reported vessel type',['All']+sorted(queue.vessel_type.dropna().unique().tolist()))
    if group != 'All': queue = queue[queue.vessel_type.eq(group)]
    mode = st.sidebar.radio('Find a track',['Review queue','Enter MMSI'])
    if mode == 'Review queue':
        if queue.empty: st.info('No queue entries for this type.'); st.stop()
        queue = queue.reset_index(drop=True)
        ix = st.sidebar.selectbox('Track to inspect',range(len(queue)),
            format_func=lambda i:f'{i+1}. {queue.iloc[i].mmsi} | {queue.iloc[i].timestamp} | {queue.iloc[i].get("selection_reason", "review")}')
        row = queue.iloc[ix]; mmsi = row.mmsi; anchor = pd.Timestamp(row.timestamp)
    else:
        mmsi = st.sidebar.text_input('MMSI (9 digits)')
        anchor_text = st.sidebar.text_input('Center time (UTC)','2025-01-01 12:00:00')
        if len(mmsi)!=9 or not mmsi.isdigit(): st.info('Enter a nine-digit MMSI.'); st.stop()
        try: anchor = pd.Timestamp(anchor_text)
        except ValueError: st.error('Use YYYY-MM-DD HH:MM:SS.'); st.stop()
        if anchor.tzinfo: anchor=anchor.tz_convert('UTC').tz_localize(None)
    hours = st.sidebar.slider('Context hours before and after',1,24,3)
    start, end = anchor-pd.Timedelta(hours=hours), anchor+pd.Timedelta(hours=hours)
    d = read_track(run,mmsi,start,end)
    if d.empty: st.info('No observations in this window.'); st.stop()
    if len(d)>30000: st.warning('More than 30,000 observations. Reduce the context window; no partial track is shown.'); st.stop()
    threshold = d.vessel_type.map(config['thresholds'])
    d['flag'] = d.ml_score.gt(threshold) | d.land_motion_rule.eq(1)
    st.subheader(f'MMSI {mmsi} · '+', '.join(d.vessel_type.unique()))
    st.write(f'{len(d):,} observations · {d.timestamp.min()} → {d.timestamp.max()} UTC')
    key = f'{run}:{mmsi}:{anchor}:{hours}'
    if len(d)>1:
        lo,hi = st.slider('Select the interval to label (observation numbers)',0,len(d)-1,(0,len(d)-1),key='range:'+key)
    else: lo=hi=0
    selected = d.iloc[lo:hi+1]
    st.caption(f'Selected: {selected.timestamp.min()} → {selected.timestamp.max()} UTC · {len(selected):,} observations. Gold points are the selection, not anomaly labels.')
    show_scores = st.sidebar.checkbox('Reveal detector flags and scores',False)
    style = st.sidebar.selectbox('Background',['carto-positron','open-street-map','white-bg'])
    st.plotly_chart(map_figure(d,selected,show_scores,config.get('parameters',{}).get('gap_minutes',30),style),width='stretch')
    st.caption('Pan and zoom; hover for time and speed. Green = start; purple = end. Background tiles require internet. Connections break across long gaps; lines are not proof of the path between reports.')
    speeds = go.Figure()
    for column,label in [('sog','Reported SOG'),('implied_speed','Position-derived speed')]:
        speeds.add_trace(go.Scatter(x=d.timestamp,y=d[column],name=label,mode='lines',connectgaps=False))
    speeds.add_vrect(x0=selected.timestamp.min(),x1=selected.timestamp.max(),fillcolor='#e7a522',opacity=.15,line_width=0)
    speeds.update_layout(height=230,margin=dict(l=0,r=0,t=20,b=0),yaxis_title='knots',xaxis_title='UTC')
    st.plotly_chart(speeds,width='stretch')
    if show_scores:
        st.line_chart(d.set_index('timestamp')[['ml_score']])
        st.caption('These are the original unsupervised scores and land rule, not feedback-model predictions.')
    with st.expander('Inspect AIS records'):
        columns=['record_id','timestamp','lat','lon','sog','cog','heading','vessel_type']
        if show_scores: columns+=['ml_score','land_motion_rule']
        st.dataframe(selected[columns],hide_index=True,width='stretch')
    with st.expander('How to judge a segment',expanded=True):
        st.write('Choose uncertain whenever the map and AIS evidence do not resolve the question. A strange route may be legitimate fishing, pilotage, anchoring or port activity. Unknown vessel type alone is not an anomaly. A land crossing can reflect erroneous positions, sparse reports or coastline/map errors; inspect the actual points and speeds.')
        st.write('Default: judge whether the selected segment contains an anomaly. This does not label every point. Narrow the interval around the change if possible. Use the point-label option only when your judgment applies to every selected report.')
    db=review_db(run)
    with st.form('review:'+key):
        label=st.radio('Segment judgment',['uncertain','normal','anomaly'],horizontal=True)
        category=st.selectbox('Anomaly category (optional)',CATEGORIES)
        confidence=st.selectbox('Confidence',['low','medium','high'])
        notes=st.text_area('Evidence / explanation','')
        explicit=st.checkbox('This judgment applies to EVERY selected point; include these points in detector feedback',False)
        st.caption('Leave unchecked for a segment-level judgment. The detector currently accepts point labels only. Category and confidence are saved for your research; they are not learned by the existing detector.')
        saved=st.form_submit_button('Save review',type='primary')
    if saved:
        save_review(db,selected,label,category,confidence,notes,'every selected point' if explicit else 'segment contains / overall')
        st.success('Review saved. You can stop and reopen the app without losing it.')
    events,points=exports(db)
    st.subheader('Saved reviews')
    st.caption(f'{len(events)} segment reviews · {len(points)} explicitly labeled points available for the existing detector')
    st.dataframe(events,hide_index=True,width='stretch')
    if len(events):
        if st.button('Undo most recent saved review'):
            last=int(events.event_id.max())
            with db:
                db.execute('DELETE FROM event_points WHERE event_id=?',(last,))
                db.execute('DELETE FROM events WHERE event_id=?',(last,))
            db.close(); st.rerun()
    st.download_button('Download segment reviews',events.to_csv(index=False).encode(),'trajectory_reviews.csv','text/csv')
    st.download_button('Download explicit point feedback',points.to_csv(index=False).encode(),'point_feedback.csv','text/csv',disabled=points.empty)
    st.caption('Feedback IDs belong to this detector run only. Downloading does not train a model. Undo here does not retract feedback already imported into the detector. Re-review affected points before importing a corrected file.')
    db.close()


if __name__=='__main__':
    main()
