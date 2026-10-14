#!/usr/bin/env python3
"""Memory-bounded first-pass Danish AIS analysis (CSV / CSV.GZ).
Install: python -m pip install pandas numpy matplotlib
Run: python analyze_copenhagen_ais.py --input ./january --month 2025-01 --output ./copenhagen_january

Raw files are never modified. Output must be new/empty. No network required.
Stages: chunked filtering -> ROI-only SQLite staging -> ordered cleaning/export.
SQLite uses disk for sorting; allow disk for the ROI database plus outputs.
Default export uses the 26 Danish AIS headers; absent source fields are blank.
Quality metadata is in clean_quality.csv.gz, joined on timestamp and MMSI.
Use --output-format analysis for compatibility with existing plotting scripts.
Only position/SOG/COG/heading checks are applied; other source fields are preserved.
Final clean CSV is ordered by MMSI/time, NOT globally by time. No interpolation.
Missing SOG/COG/heading does not invalidate a position. Jumps remain flagged.
Rejections are mutually exclusive, in pipeline order. Field repairs overlap.
Density counts AIS observations, not vessels or time spent (reporting-rate bias).
Timestamps use the source convention; timezone conversion is not performed.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import itertools
import json
import logging
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

COLS = {'timestamp': 'timestamp', 'type of mobile': 'mobile', 'mmsi': 'mmsi',
        'latitude': 'lat', 'longitude': 'lon', 'sog': 'sog', 'cog': 'cog',
        'heading': 'heading', 'ship type': 'ship_type',
        'navigational status': 'nav_status', 'name': 'name'}
FIELDS = ['timestamp', 'mmsi', 'lat', 'lon', 'sog', 'cog', 'heading',
          'mobile', 'ship_type', 'nav_status', 'name', 'quality_flags', 'source_file', 'source_row']
EXTRA = ['segment_id', 'gap_seconds', 'implied_speed_knots', 'jump_flag']
# Danish AIS export order. Retain source fields rather than inventing values.
AIS_COLUMNS = {
    '# Timestamp': 'timestamp', 'Type of mobile': 'mobile', 'MMSI': 'mmsi',
    'Latitude': 'lat', 'Longitude': 'lon', 'Navigational status': 'nav_status',
    'ROT': 'rot', 'SOG': 'sog', 'COG': 'cog', 'Heading': 'heading',
    'IMO': 'imo', 'Callsign': 'callsign', 'Name': 'name', 'Ship type': 'ship_type',
    'Cargo type': 'cargo_type', 'Width': 'width', 'Length': 'length',
    'Type of position fixing device': 'position_fixing_device', 'Draught': 'draught',
    'Destination': 'destination', 'ETA': 'eta', 'Data source type': 'data_source_type',
    'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd',
}
for label, key in AIS_COLUMNS.items():
    COLS[label.lstrip('#').strip().lower()] = key
    if key not in FIELDS:
        FIELDS.append(key)
QC_FIELDS = ['quality_flags', 'source_file', 'source_row'] + EXTRA


def norm(s):
    return s.strip().lstrip('#').strip().lower().replace('_', ' ')

def distance_nm(a, b):
    lat1, lat2 = math.radians(a['lat']), math.radians(b['lat'])
    dl = math.radians(b['lon'] - a['lon'])
    dp = lat2 - lat1
    h = math.sin(dp/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dl/2)**2
    return 3440.065 * 2 * math.asin(min(1, math.sqrt(max(0, h))))

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--output-format', choices=['ais','analysis'], default='ais',
                   help='ais: Danish headers/timestamps plus separate quality sidecar; analysis: internal headers for existing plotting scripts')
    p.add_argument('--month', required=True, help='YYYY-MM; e.g. 2025-01')
    p.add_argument('--south', type=float, default=55.50)
    p.add_argument('--north', type=float, default=55.90)
    p.add_argument('--west', type=float, default=12.35)
    p.add_argument('--east', type=float, default=12.85)
    p.add_argument('--chunksize', type=int, default=200_000)
    p.add_argument('--gap-minutes', type=float, default=30)
    p.add_argument('--jump-knots', type=float, default=80, help='Review flag only, not deletion')
    p.add_argument('--jump-min-metres', type=float, default=100, help='Ignore small positional jitter for jump flags')
    p.add_argument('--keep-staging', action='store_true')
    p.add_argument('--timestamp-format', default='%d/%m/%Y %H:%M:%S')
    a = p.parse_args()
    try:
        start = pd.Timestamp(datetime.strptime(a.month, '%Y-%m'))
    except ValueError:
        p.error('--month must be YYYY-MM')
    end = start + pd.offsets.MonthBegin(1)
    if not (-90 <= a.south < a.north <= 90 and -180 <= a.west < a.east <= 180):
        p.error('Invalid ROI bounds')
    if a.chunksize < 1 or a.gap_minutes <= 0 or a.jump_knots <= 0 or a.jump_min_metres < 0:
        p.error('Chunk size/gap/jump speed must be positive; jump distance must be nonnegative')
    if not a.input.is_dir():
        p.error('--input must be an existing directory')
    files = sorted(x for x in a.input.rglob('*') if x.is_file() and
                   (x.name.lower().endswith('.csv') or x.name.lower().endswith('.csv.gz')))
    if not files:
        p.error('No CSV or CSV.GZ input files found')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Output directory must be empty: prevents accidental overwrite or input reuse')
    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(a.output/'run.log')])
    counts, repairs = Counter(), Counter()
    database = a.output/'staging.sqlite'
    db = sqlite3.connect(database)
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('CREATE TABLE observations (' + ','.join(f'{k} '+('REAL' if k in ['lat','lon','sog','cog','heading'] else 'INTEGER' if k=='source_row' else 'TEXT') for k in FIELDS) + ')')
    audit = gzip.open(a.output/'removed_rows.csv.gz', 'wt', newline='')
    aw = csv.DictWriter(audit, fieldnames=FIELDS+['reason'], extrasaction='ignore')
    aw.writeheader()
    def reject(frame, mask, reason):
        mask = mask.fillna(True)
        bad = frame.loc[mask]
        counts[reason] += len(bad)
        for row in bad.to_dict('records'):
            aw.writerow({**row, 'reason': reason})
        return frame.loc[~mask].copy()
    started = time.monotonic()
    file_reports = []
    for idx, f in enumerate(files, 1):
        t = time.monotonic()
        raw_n, roi_n = 0, 0
        header = pd.read_csv(f, nrows=0)
        mapping = {c: COLS[norm(c)] for c in header.columns if norm(c) in COLS}
        required = {'timestamp','mmsi','lat','lon','mobile'}
        if not required.issubset(mapping.values()):
            raise ValueError(f'{f}: missing required columns {required-set(mapping.values())}')
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f'{f}: ambiguous duplicate column mappings')
        logging.info('[%s/%s] Reading %s', idx, len(files), f.name)
        for chunk in pd.read_csv(f, usecols=list(mapping), dtype=str, chunksize=a.chunksize,
                                 keep_default_na=False, on_bad_lines='error'):
            d = chunk.rename(columns=mapping)
            n = len(d)
            d['source_file'] = str(f.relative_to(a.input))
            d['source_row'] = np.arange(raw_n+2, raw_n+n+2)
            raw_n += n
            counts['input_rows'] += n
            for k in FIELDS:
                if k not in d:
                    d[k] = ''
            # Audit invalid core fields before filtering by geography/month.
            ts = pd.to_datetime(d.timestamp, format=a.timestamp_format, errors='coerce')
            d = reject(d, ts.isna(), 'invalid_timestamp')
            ts = ts.loc[d.index]
            inside_month = (ts >= start) & (ts < end)
            counts['outside_month'] += int((~inside_month).sum())
            d = d.loc[inside_month].copy()
            d['timestamp'] = ts.loc[d.index].dt.strftime('%Y-%m-%dT%H:%M:%S')
            vessel = d.mobile.str.strip().str.lower().isin(['class a', 'class b'])
            counts['non_vessel_records'] += int((~vessel).sum())
            d = d.loc[vessel].copy()
            valid_id = d.mmsi.str.fullmatch(r'\d{9}') & d.mmsi.ne('000000000')
            d = reject(d, ~valid_id, 'invalid_mmsi')
            for k in ['lat','lon']:
                d[k] = pd.to_numeric(d[k], errors='coerce')
            valid_pos = d.lat.between(-90,90) & d.lon.between(-180,180)
            d = reject(d, ~valid_pos, 'invalid_position')
            roi = d.lat.between(a.south,a.north) & d.lon.between(a.west,a.east)
            counts['outside_roi'] += int((~roi).sum())
            d = d.loc[roi].copy()
            for k, upper in [('sog',102.3),('cog',360),('heading',360)]:
                values = pd.to_numeric(d[k], errors='coerce')
                valid = values.ge(0) & values.lt(upper)
                if k == 'heading':
                    valid &= values.mod(1).eq(0)
                missing = d[k].str.strip().str.lower().isin(['','nan','na','unknown'])
                fixed = ~valid & ~missing
                repairs[k+'_invalid_or_unavailable'] += int(fixed.sum())
                repairs[k+'_missing'] += int(missing.sum())
                d.loc[~valid, 'quality_flags'] += k+'_missing_or_invalid;'
                d[k] = values.where(valid)
            d = d[FIELDS]
            if len(d):
                d.to_sql('observations', db, if_exists='append', index=False, chunksize=5000)
            roi_n += len(d)
            logging.info('  rows read %s; ROI staged %s', f'{raw_n:,}', f'{roi_n:,}')
        file_reports.append({'file': str(f.relative_to(a.input)), 'input_rows':raw_n,
                             'staged_roi_rows':roi_n, 'seconds':round(time.monotonic()-t,2)})
    counts['staged_roi_rows'] = sum(x['staged_roi_rows'] for x in file_reports)
    logging.info('Indexing %s ROI observations on disk', counts['staged_roi_rows'])
    db.execute('CREATE INDEX time_order ON observations(mmsi,timestamp)')
    db.commit()
    db.row_factory = sqlite3.Row
    cursor = db.execute('SELECT * FROM observations ORDER BY mmsi,timestamp,source_file,source_row')
    daily, grid, speed_hist = Counter(), Counter(), np.zeros(103,dtype=np.int64)
    clean = gzip.open(a.output/'clean_ais.csv.gz','wt',newline='')
    cw = csv.DictWriter(clean, fieldnames=list(AIS_COLUMNS) if a.output_format == 'ais' else FIELDS+EXTRA)
    cw.writeheader()
    quality = None
    if a.output_format == 'ais':
        quality = gzip.open(a.output/'clean_quality.csv.gz','wt',newline='')
        qw = csv.DictWriter(quality, fieldnames=['# Timestamp','MMSI']+QC_FIELDS)
        qw.writeheader()
    vf = open(a.output/'vessel_summary.csv','w',newline='')
    vw = csv.DictWriter(vf, fieldnames=['mmsi','observations','start','end','segments','jump_flags']); vw.writeheader()
    vessel_stats = None
    previous = None
    for (mmsi, timestamp), group in itertools.groupby(cursor, key=lambda r:(r['mmsi'],r['timestamp'])):
        # Streaming group: bounded even if a timestamp is repeated millions of times.
        candidate, total, conflict, variants = None, 0, False, False
        # Two passes are only needed for conflicting timestamps to retain their audit rows.
        for raw in group:
            r = dict(raw); total += 1
            if candidate is None:
                candidate = r
            else:
                conflict |= (r['lat'],r['lon']) != (candidate['lat'],candidate['lon'])
                variants |= any(r[k] != candidate[k] for k in ['sog','cog','heading','mobile','ship_type','nav_status','name'])
                score = lambda x: sum(x[k] is not None for k in ['sog','cog','heading'])
                if score(r) > score(candidate):
                    candidate = r
        if conflict:
            counts['conflicting_position_rows'] += total
            for raw in db.execute('SELECT * FROM observations WHERE mmsi=? AND timestamp=?',(mmsi,timestamp)):
                aw.writerow({**dict(raw),'reason':'conflicting_same_time_position'})
            # Do not bridge a known conflicting time.
            if previous is not None and previous['mmsi'] == mmsi:
                previous = None
            continue
        r = candidate
        counts['duplicate_or_redundant_rows'] += total-1
        if total > 1:
            skipped = False
            for raw in db.execute('SELECT * FROM observations WHERE mmsi=? AND timestamp=?',(mmsi,timestamp)):
                rr = dict(raw)
                if not skipped and rr['source_file']==r['source_file'] and rr['source_row']==r['source_row']:
                    skipped = True
                    continue
                aw.writerow({**rr,'reason':'redundant_same_time_position'})
        if variants:
            r['quality_flags'] += 'same_time_metadata_disagreement;'
        if vessel_stats is None or vessel_stats['mmsi'] != mmsi:
            if vessel_stats:
                vw.writerow(vessel_stats)
            counts['vessels'] += 1
            vessel_stats = dict(mmsi=mmsi,observations=0,start=timestamp,end=timestamp,segments=0,jump_flags=0)
            previous = None
        dt, speed, jump = None, None, False
        new_segment = previous is None
        if previous is not None:
            dt = (datetime.fromisoformat(timestamp)-datetime.fromisoformat(previous['timestamp'])).total_seconds()
            dist = distance_nm(previous,r)
            speed = dist*3600/dt
            jump = dt <= a.gap_minutes*60 and speed > a.jump_knots and dist*1852 >= a.jump_min_metres
            new_segment = dt > a.gap_minutes*60 or jump
        if new_segment:
            vessel_stats['segments'] += 1
        if jump:
            counts['jump_flagged_rows'] += 1
            vessel_stats['jump_flags'] += 1
            r['quality_flags'] += 'implausible_displacement_review;'
        r.update(segment_id=f'{mmsi}_{vessel_stats["segments"]}',gap_seconds=dt,
                 implied_speed_knots=round(speed,3) if speed is not None else None,jump_flag=int(jump))
        if a.output_format == 'ais':
            exported = {label: r[key] for label,key in AIS_COLUMNS.items()}
            exported['# Timestamp'] = datetime.fromisoformat(r['timestamp']).strftime('%d/%m/%Y %H:%M:%S')
            exported['Heading'] = int(r['heading']) if r['heading'] is not None else ''
            cw.writerow(exported)
            qw.writerow({'# Timestamp':exported['# Timestamp'], 'MMSI':r['mmsi'],
                         **{key:r[key] for key in QC_FIELDS}})
        else:
            cw.writerow({**r, 'heading': int(r['heading']) if r['heading'] is not None else ''})
        previous = r
        vessel_stats['observations'] += 1; vessel_stats['end'] = timestamp
        counts['clean_rows'] += 1
        daily[timestamp[:10]] += 1
        grid[(int((r['lon']-a.west)/.005),int((r['lat']-a.south)/.005))] += 1
        if r['sog'] is not None:
            speed_hist[min(102,int(r['sog']))] += 1
        if counts['clean_rows'] % 500000 == 0:
            logging.info('Exported %s clean rows', f'{counts["clean_rows"]:,}')
    if vessel_stats:
        vw.writerow(vessel_stats)
    clean.close(); vf.close(); audit.close(); db.close()
    if quality is not None: quality.close()
    pd.DataFrame(file_reports).to_csv(a.output/'file_summary.csv',index=False)
    days = pd.date_range(start,end-pd.Timedelta(days=1),freq='D').strftime('%Y-%m-%d')
    pd.DataFrame({'date':days,'observations':[daily[d] for d in days]}).to_csv(a.output/'daily_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    if grid:
        nx=int(math.ceil((a.east-a.west)/.005))+1; ny=int(math.ceil((a.north-a.south)/.005))+1
        arr=np.zeros((ny,nx))
        for (x,y),n in grid.items(): arr[y,x]=n
        from matplotlib.colors import LogNorm
        mesh=axes[0].pcolormesh(a.west+np.arange(nx+1)*.005,a.south+np.arange(ny+1)*.005,
                                np.ma.masked_equal(arr,0),norm=LogNorm(vmin=1,vmax=max(2,arr.max())))
        fig.colorbar(mesh,ax=axes[0],label='AIS observations')
    axes[0].set(xlim=(a.west,a.east),ylim=(a.south,a.north),xlabel='Longitude',ylabel='Latitude',title='Copenhagen ROI: observation density')
    axes[0].set_aspect(1/math.cos(math.radians((a.south+a.north)/2)))
    axes[0].plot(12.5683,55.6761,'r+',markersize=8)
    axes[1].bar(np.arange(len(days))+1,[daily[d] for d in days]); axes[1].set(title='Daily retained observations',xlabel='Day of month',ylabel='Count')
    axes[2].bar(np.arange(103),speed_hist); axes[2].set(title='Reported SOG (valid values)',xlabel='Speed (knots)',ylabel='Count',yscale='symlog')
    fig.suptitle(f'AIS {a.month} — quality-controlled positions; jumps retained for review')
    fig.tight_layout(); fig.savefig(a.output/'overview.png',dpi=180); plt.close(fig)
    accounted = sum(counts[k] for k in ['invalid_timestamp','outside_month','non_vessel_records','invalid_mmsi','invalid_position','outside_roi','conflicting_position_rows','duplicate_or_redundant_rows','clean_rows'])
    assert accounted == counts['input_rows'], (accounted,counts)
    report = {'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'counts':dict(counts),'field_repairs_roi_only':dict(repairs),'files':file_reports,
              'days_without_retained_roi_observations':[d for d in days if not daily[d]],
              'seconds':round(time.monotonic()-started,2),
              'notes':['ROI selected before temporal analysis; exits/reentries may therefore be connected within gap threshold.',
                       'No coastline or land mask applied. Red cross is approximate Copenhagen city centre.',
                       'Density is observation count, affected by reporting frequency.',
                       'Jump flags are uncertain: retained, with segment break before flagged point.',
                       'No interpolation, coordinate rounding, or behavioural anomaly removal.',
                       'MMSI checks syntax only; this does not authenticate vessel identity.',
                       'source_row counts CSV data records plus header; assumes standard single-line records.',
                       'Days without ROI observations do not necessarily imply missing source files.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    if not a.keep_staging: database.unlink()
    logging.info('Done: %s clean rows, %s vessels. Outputs: %s',counts['clean_rows'],counts['vessels'],a.output)

def repair_heading():
    """Repair an already generated CSV without repeating the month analysis."""
    from decimal import Decimal, InvalidOperation
    import os
    import tempfile
    parser = argparse.ArgumentParser(description='Rewrite Heading as integer text, preserving all other fields. Streams CSV/GZ; never overwrites an existing file.')
    parser.add_argument('--repair-heading', required=True, type=Path)
    parser.add_argument('--output-file', required=True, type=Path)
    args = parser.parse_args()
    src, dst = args.repair_heading, args.output_file
    if not src.is_file(): parser.error('Source file does not exist')
    if dst.exists(): parser.error('Output file already exists; choose a new filename')
    if not (str(dst).endswith('.csv') or str(dst).endswith('.csv.gz')):
        parser.error('Output filename must end with .csv or .csv.gz')
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.heading_repair_', dir=dst.parent)
    os.close(fd)
    rows = changed = 0
    try:
        source_open = gzip.open if str(src).endswith('.gz') else open
        target_open = gzip.open if str(dst).endswith('.gz') else open
        with source_open(src, 'rt', newline='', encoding='utf-8-sig') as source, target_open(temporary, 'wt', newline='', encoding='utf-8') as target:
            reader = csv.reader(source)
            header = next(reader)
            matches = [i for i, name in enumerate(header) if name.strip().lower() == 'heading']
            if len(matches) != 1: raise ValueError('Expected exactly one Heading column')
            index = matches[0]
            writer = csv.writer(target); writer.writerow(header)
            for row in reader:
                rows += 1
                if len(row) != len(header): raise ValueError(f'Record {rows}: wrong column count')
                old = row[index]; value = old.strip()
                if value.lower() in ('', 'nan', 'none', 'null', 'na'):
                    new = ''
                else:
                    try: number = Decimal(value)
                    except InvalidOperation: raise ValueError(f'Record {rows}: invalid Heading {value!r}')
                    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number <= 359:
                        raise ValueError(f'Record {rows}: Heading {value!r} is not a valid whole degree; no output committed')
                    new = str(int(number))
                row[index] = new; changed += (new != old)
                writer.writerow(row)
                if rows % 1_000_000 == 0: print(f'Processed {rows:,} records; changed {changed:,}', flush=True)
        # Exclusive creation prevents overwriting a file created while processing.
        os.link(temporary, dst)
        print(f'Saved {dst}: {rows:,} records, {changed:,} headings reformatted.')
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    import sys
    if '--repair-heading' in sys.argv:
        repair_heading()
    else:
        main()
