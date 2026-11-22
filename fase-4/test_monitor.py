import subprocess,sys
from pathlib import Path

def test_empty(tmp_path):
    p=tmp_path/'empty.csv'
    out=subprocess.check_output([sys.executable,str(Path(__file__).with_name('monitor.py')),str(p)],text=True)
    assert 'Total de solicitudes: 0' in out and 'sin solicitudes' in out
