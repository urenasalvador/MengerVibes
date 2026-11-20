# MengerVibes

<img src="assets/mengervibes.png" alt="Icono de MengerVibes" width="128">

Explorador de fractales 3D en tiempo real para Windows, desarrollado en Rust con wgpu. Puedes volar por el interior de una esponja de Menger, combinar fórmulas fractales y exportar tus renders en PNG.

Es un proyecto experimental desarrollado mediante *vibe coding*, con la ayuda de OpenAI.

## Qué puedes hacer

- Explorar seis fórmulas: Menger, Mandelbulb, Mandelbox, KIFS, Amazing Surface y Pseudo-Kleinian.
- Ajustar los parámetros de cada fórmula y combinar varias mediante mutaciones.
- Cambiar el aspecto con matcaps, oclusión ambiental, niebla y profundidad de campo.
- Guardar configuraciones en una galería de presets con miniaturas.
- Renderizar a una resolución independiente del visor y guardar el resultado en `renders/`.

## Compilar y ejecutar en Windows

Necesitas Rust estable con Cargo, las herramientas de compilación de C++ de Visual Studio y el SDK de Windows. Desde la raíz del repositorio, ejecuta:

```powershell
cargo run --release --locked
```

El ejecutable queda en `target/release/mengervibes.exe`.

## Controles básicos

| Control | Acción |
| --- | --- |
| W / S | Avanzar o retroceder |
| A / D | Desplazarse lateralmente |
| Q / E | Bajar o subir |
| Botón derecho del ratón + movimiento | Orientar la cámara |
| R | Restablecer la cámara |
| 1–6 | Cambiar de fórmula |

## Licencia

GNU General Public License v3.0 (GPL-3.0-only). El texto completo está en [COPYING](COPYING).
