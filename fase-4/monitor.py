"""Indicadores de operación; uso: python fase-4/monitor.py [ruta_csv]."""
import csv,sys,os
from pathlib import Path
p=Path(sys.argv[1] if len(sys.argv)>1 else os.getenv('MONITOR_LOG',str(Path(__file__).with_name('solicitudes.csv'))))
rows=list(csv.DictReader(p.open(encoding='utf8'))) if p.exists() and p.stat().st_size else []
failed=sum(400<=int(r['estado_http'])<=599 for r in rows)
print(f'Total de solicitudes: {len(rows)}')
print(f'Porcentaje de solicitudes fallidas: {failed/len(rows)*100:.1f} %' if rows else 'Porcentaje de solicitudes fallidas: sin solicitudes')
