# Fase 4 — Monitoreo y política de reentrenamiento

Cada solicitud POST `/predict`, incluso las rechazadas, genera exactamente un evento con `fecha_hora`, `estado_http`, `version`, `prediccion`. El archivo persistente se guarda en `fase-4/registros/solicitudes.csv` mediante el volumen `/logs` de Docker. Para cada solicitud, la versión y predicción se anotan solo si el modelo respondió correctamente.

## Reporte (ensayo reproducible)

Se enviaron **dos solicitudes válidas y una inválida** (sin `bedrooms`) con el cliente de pruebas. El registro de muestra está en `fase-4/muestra_solicitudes.csv` y contiene tres filas: dos HTTP 200 con `v1` y una HTTP 422. Indicadores: **total 3; fallidas 1/3 = 33,3 %**. El fallo era esperado, porque el esquema debe rechazar una variable obligatoria ausente. Estos son datos de una demostración; el registro de producción local crece con nuevas llamadas.

Calcular indicadores del registro real:

```powershell
python fase-4/monitor.py fase-4/registros/solicitudes.csv
```

Con registro vacío: total 0, porcentaje `sin solicitudes`. HTTP 4xx y 5xx se consideran fallos.

## Versiones y reentrenamiento

Se conservan `v1` (Gradient Boosting) y `v2` (Random Forest) en `artifacts/`. La API usa `v1` por defecto; para volver a `v1`, definir `MODEL_VERSION=v1` y recrear el contenedor. Para consultar la otra sin cambiar la configuración, enviar `"version":"v2"` en el JSON.

**Política:** Cuando se reúnan al menos 100 viviendas nuevas con precio real conocido, la persona responsable del proyecto ejecutará POST `/train` para generar una versión candidata sin sobrescribir las anteriores. Se comparará el MAE de la candidata con el de la versión activa sobre el mismo conjunto reservado de datos reales; se activará únicamente si mejora y se verifica que no haya degradación grave en RMSE. Los fallos HTTP por campos ausentes se corrigen en el cliente, no mediante reentrenamiento.

El porcentaje de fallos HTTP mide operación, **no** exactitud del precio: para medir calidad predictiva se necesitan los precios reales.
