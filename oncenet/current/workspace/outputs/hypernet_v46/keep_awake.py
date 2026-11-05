"""Run-scoped sleep inhibition; no persistent system setting is changed."""
import ctypes,time,json
from pathlib import Path
R=Path(__file__).resolve().parent;api=ctypes.windll.kernel32.SetThreadExecutionState
assert api(0x80000001);start=time.time()
try:
 (R/'awake_status.json').write_text(json.dumps(dict(active=True,scope='v46 only; screen may sleep')))
 while time.time()-start<10800 and not (R/'report_summary.json').exists():time.sleep(10)
finally:
 api(0x80000000)
 (R/'awake_status.json').write_text(json.dumps(dict(active=False,seconds=time.time()-start)))
