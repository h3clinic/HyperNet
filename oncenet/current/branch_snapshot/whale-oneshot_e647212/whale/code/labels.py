"""DCLDE 2013 annotation log -> per-file call lists (seconds from file start)."""
import os, re, datetime as dt
import pandas as pd

LOG = 'NEFSC_SBNMS_200903_NOPP6_CH10_allbaleen_detection_log_updated-2025-07-02.csv'


def file_start(name):
    m = re.search(r'_(\d{8})_(\d{6})_', name)
    return dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')


def load(root):
    d = pd.read_csv(os.path.join(root, 'labels', LOG))
    # Times carry -05:00 and file names are EST local, so drop the offset.
    d['start'] = pd.to_datetime(d.Start_DateTime_ISO8601.str[:19])
    d['end'] = pd.to_datetime(d.End_DateTime_ISO8601.str[:19])
    d['low'] = d['Low.Freq..Hz.']; d['high'] = d['High.Freq..Hz.']
    d['species'] = d.Species; d['confidence'] = d.Detection_Confidence
    return d[['start', 'end', 'low', 'high', 'species', 'confidence']]
