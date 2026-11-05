<!--
SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
SPDX-License-Identifier: LicenseRef-Proprietary
-->

# Parking Intelligence

Sistema de control de acceso y ocupación para dos portones, con captura RTSP,
detección y clasificación de vehículos, sesiones auditables, búsqueda de
incidentes y telemetría del modelo.

## Levantarlo

```bash
./scripts/dev.sh up
```

API en `http://localhost:8000`, panel en `http://localhost:5173`. El comando
aplica las migraciones y verifica PostgreSQL y Redis. Ver `ARCHITECTURE.md` para
las capas y `RUNBOOK.md` para la reproducción de datasets y modelos.

## Propiedad y licencia

© 2026 Nina Marco <marconina999@gmail.com>. **Software propietario.**

El texto legal completo está en inglés en [`LICENSE`](LICENSE); esa versión es
la que prevalece. [`NOTICE`](NOTICE) documenta los componentes de terceros.

No se concede uso comercial, copia, redistribución ni obras derivadas sin
autorización escrita. La evaluación académica de la tesis tiene un permiso
limitado y no comercial. El código, los pesos y los datasets no se publican con
la tesis.

Para licencias comerciales o consultas: marconina999@gmail.com.

## Terceros

Ultralytics YOLO es **AGPL-3.0**. Para un producto comercial se requiere la
licencia enterprise de Ultralytics o cumplir la AGPL. El resto de las
dependencias y sus licencias están en `NOTICE`.
