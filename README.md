# House Price Prediction — Proyecto Integrador 2026-II

Universidad de Antioquia · Modelos y Simulación de Sistemas I · Carlos Andrés Teleche

Conserva las fases 1 y 2; añade **Fase 3 (API REST)** y **Fase 4 (monitoreo)**. Instrucciones de instalación, entrenamiento, predicción, Docker y pruebas: [fase-3/README.md](fase-3/README.md). Reporte, métricas y política de reentrenamiento: [fase-4/README.md](fase-4/README.md).

## Ejecución

```powershell
docker compose up --build -d
python fase-3/cliente.py
python fase-4/monitor.py fase-4/registros/solicitudes.csv
```

Verificar `http://localhost:8000/health` y `http://localhost:8000/docs`. Las dos versiones de ejemplo están en `artifacts/`; no se sobrescriben al entrenar. Los datos y el registro se montan fuera del contenedor.

**Entrega Git**: trabajar en una rama de la fase, abrir Pull Request a `develop`, revisarlo, integrar en `main` y marcar el commit o etiqueta de entrega, conforme al PDF de especificaciones.
