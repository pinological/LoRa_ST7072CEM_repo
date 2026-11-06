import os, glob
from pathlib import Path
import numpy as np
import pandas as pd

CASE = {'A': 1, 'B': 2, 'C': 3, 'D': 3}


def find_data_dir():
    here = Path(__file__).resolve().parent
    for d in [os.environ.get('LORA_DATA'), here / 'LoRa_data', here.parent / 'LoRa_data', Path.cwd() / 'LoRa_data']:
        if d and (Path(d) / 'raw').is_dir():
            return Path(d)
    raise FileNotFoundError('LoRa_data folder not found. Put it next to app.py or one level up, or set LORA_DATA.')


def hav(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def load_data(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    logs = []
    for f in sorted(glob.glob(str(data_dir / 'raw' / 'site_*_roamer_log.csv'))):
        d = pd.read_csv(f)
        d['site'] = Path(f).name.split('_')[1]
        logs.append(d)
    log = pd.concat(logs, ignore_index=True)

    sw = log[log.status.isin(['OK', 'TIMEOUT'])].copy()
    sw['stop_key'] = sw.site + sw.stop_id.astype(int).astype(str)
    n = sw.groupby('stop_key').size()
    p = sw[sw.stop_key.isin(n[n == 90].index)].copy()
    p['case'] = p.site.map(CASE)
    p['ok'] = (p.status == 'OK').astype(int)
    p['pl_db'] = np.where(p.ok == 1, p.pwr - p.rssi, np.nan)

    feat = pd.read_csv(data_dir / 'stop_features.csv')
    p = p.merge(feat.drop(columns='site'), on='stop_key', how='left')

    s = p.groupby('stop_key').agg(site=('site', 'first'), case=('case', 'first'), n_ok=('ok', 'sum'),
                                  lat=('lat', 'median'), lon=('lon', 'median')).reset_index()
    s['delivery_rate'] = s.n_ok / 90
    s = s.merge(p[p.pwr == 14].groupby('stop_key').pl_db.median().rename('pl_median_14dbm'), on='stop_key', how='left')
    s = s.merge(feat.drop(columns='site'), on='stop_key')

    st = pd.read_csv(data_dir / 'stations.csv').set_index('site')
    fails = log[log.status == 'NOACK'][['site', 'stop_id', 'lat', 'lon']].sort_values(['site', 'stop_id'])
    fails['dist_m'] = hav(st.loc[fails.site, 'station_lat'].values, st.loc[fails.site, 'station_lon'].values, fails.lat, fails.lon)
    fails['nearest_sweep_m'] = [hav(r.lat, r.lon, s[s.site == r.site].lat.values, s[s.site == r.site].lon.values).min()
                                for r in fails.itertuples()]
    keep = []
    for r in fails[fails.nearest_sweep_m > 30].itertuples():
        if not any(k.site == r.site and hav(r.lat, r.lon, k.lat, k.lon) < 30 for k in keep):
            keep.append(r)
    lost = pd.DataFrame(keep)[['site', 'dist_m']]
    lost['case'] = lost.site.map(CASE)

    p = p.drop(columns=['lat', 'lon', 'phone_lat', 'phone_lon', 'gps_alt_m', 'phone_alt_m'], errors='ignore')
    s = s.drop(columns=['lat', 'lon'])
    return p, s, lost
