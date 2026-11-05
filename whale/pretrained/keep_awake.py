"""Tell Windows the system is in use so it never idle-sleeps while long jobs run.
Changes no settings; the request ends when this process exits.
Runs until the file STOP_KEEP_AWAKE exists next to it (or 72 h)."""
import ctypes, os, time
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
stop = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STOP_KEEP_AWAKE')
t0 = time.time()
while not os.path.exists(stop) and time.time() - t0 < 72 * 3600:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
    time.sleep(30)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
