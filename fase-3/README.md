# Fase 3 — API REST de precios de viviendas

## Inicio rápido (desde la raíz del repositorio)

Requisito: Docker Desktop encendido. Ejecutar:

```powershell
docker compose up --build -d
```

Abrir http://localhost:8000/health y http://localhost:8000/docs.

**Predicción**: `python fase-3/cliente.py` (requiere pandas en Python local). Alternativamente, usar Swagger en `/docs`, ruta POST `/predict`, con los campos de `fase-3/ejemplo_prediccion.json`. Respuesta: `{"prediccion": <número>, "version": "v1"}`.

**Entrenamiento**: en `/docs` usar POST `/train` con `{}`. Devuelve HTTP 201 y el identificador de una versión nueva. Opcionalmente enviar `{"version":"v3"}`; si ya existe, HTTP 409. Los artefactos `modelo_<version>.joblib` y `.json` permanecen en `artifacts/` incluso al borrar el contenedor.

**Cambio de versión**: definir `MODEL_VERSION=v2` en archivo `.env` junto a `compose.yaml` y ejecutar `docker compose up -d --force-recreate`; para volver, poner `MODEL_VERSION=v1`. También se puede indicar `"version":"v2"` en la solicitud a `/predict` sin cambiar la configuración general. Versiones incluidas: `v1` (Gradient Boosting) y `v2` (Random Forest). La versión no disponible devuelve HTTP 503.

**Pruebas automatizadas**: `docker compose run --rm api python -m pytest -q fase-3/test_api.py fase-4/test_monitor.py`.

**Persistencia**: CSV de datos montado en `/data` (solo lectura), modelos en `/models` y eventos en `/logs`; todos persisten en el equipo anfitrión. `docker compose down` no borra estos archivos.

**Esquema de entrada**: `date` (texto con fecha válida); `city` y `statezip` (texto); `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated` (número o null). La variable objetivo `price` **no** se envía. Campos extra se ignoran. Categorías desconocidas se procesan con `OneHotEncoder(handle_unknown="ignore")`; nulos numéricos usan la imputación del pipeline guardado. Campo ausente o tipo incorrecto: HTTP 422.

**Ejemplo realista**: ver `ejemplo_prediccion.json`. Los datos provienen de una fila del CSV de Fase 1, eliminando `price`, `price_per_sqft` y `street`.

**Evaluación**: separación aleatoria reproducible 80 % entrenamiento y 20 % test (semilla 42). Dentro del 80 %, validación del 20 % para elegir modelo con MAE. Se vuelve a entrenar con el 80 % completo. Metadatos JSON contienen métricas finales y baseline de mediana sobre el mismo test. No se usa test para elegir el algoritmo. Se excluye `price_per_sqft` para evitar fuga de información.

**Alcance**: el código es una demostración local. No se deben cargar modelos `.joblib` de fuentes desconocidas.
