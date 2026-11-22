# Resolución adaptativa del visor

## Diagnóstico

El visor ya dibujaba una pasada espacialmente completa a resolución reducida para varias modalidades. Durante el movimiento no hay reproyección temporal. En la ruta reducida anterior el shader redondea el nivel de detalle adaptativo y reduce el presupuesto de marcha de 512 a 256 pasos. Los rayos que agotan ese presupuesto pueden devolver fondo o una aproximación de intersección. Además, el compositor anterior podía pasar a mostrar la acumulación por bloques antes de completar la primera muestra.

## Implementación

- **Sampling → Adaptive resolution** permite usar la nueva ruta o comparar con el visor anterior.
- Cada sesión de navegación empieza a un cuarto de resolución por eje. El compositor bilineal existente amplía la imagen completa. El muestreo subpíxel se escala al tamaño de esa pasada. Las normales conservan la referencia de la resolución completa.
- El visor usa la misma regla de detalle, precisión y presupuesto de marcha principal que el refinamiento. Las opciones existentes de muestras y rebotes de navegación siguen gobernando la iluminación durante el movimiento. El modo adaptativo refina a resolución completa al detener la cámara, incluido Ambient + AO.
- El mando **Pass budget** fija el tiempo objetivo. Un promedio por rayo estima el coste de cuadruplicar los píxeles. Para subir un nivel hacen falta dos estimaciones consecutivas dentro del presupuesto. Al superar el presupuesto holgadamente se baja un nivel y se esperan ocho pasadas antes de volver a evaluar una subida. Si la validez estimada baja del 90 %, se conserva la resolución mientras se siga en presupuesto.
- La ruta adaptativa mantiene una sola pasada de visor en vuelo. Las solicitudes acumuladas se vuelven a evaluar con el estado más reciente de cámara y escena cuando esa pasada termina. Mientras se dibuja la primera muestra en bloques, el compositor conserva la última imagen completa. Cada cambio de cámara o escena reinicia el refinamiento.
- Los temporizadores GPU se activan si el adaptador los admite. En otros adaptadores el planificador usa el tiempo hasta completar la cola. La estimación de rayos válidos toma 1.024 posiciones estratificadas de la imagen ya trazada. Cuenta intersecciones y salidas resueltas. Excluye aproximaciones de intersección y marchas primarias agotadas. Las consultas de iluminación secundaria y las evaluaciones individuales de SDF quedan fuera de esta métrica.

El presupuesto es un objetivo. La pasada de un cuarto de resolución puede superar el presupuesto si cada rayo es muy costoso. Una pasada ya enviada termina normalmente. Los estados del planificador se mantienen durante la sesión y son independientes de los presets de fractal.

## Validación reproducible

```powershell
cargo test --offline -- --include-ignored --test-threads=1 --nocapture
cargo test --offline benchmark_adaptive_viewport -- --ignored --nocapture
$env:MENGERVIBES_BENCHMARK_1080P = '1'
cargo test --offline benchmark_adaptive_viewport -- --ignored --nocapture
```

El benchmark usa Vulkan y temporizadores GPU. Compara una muestra por píxel en tres orientaciones después de calentar el shader. El tiempo de pasada mide el trazado en GPU. La latencia hasta mostrar el primer fotograma completo en una ventana incluye composición, presentación y sistema operativo, por lo que requiere una prueba interactiva aparte.

En una RTX 3090 con Vulkan se completaron las 23 pruebas, incluida la presentación en una ventana Windows oculta. Con Mandelbox + Menger a 300 × 180, el error RGB medio tras reducir ambas imágenes a 10 × 6 bajó de 10,35 a 6,08 sobre 255 respecto a la primera muestra completa. La primera muestra calculada por bloques coincidió píxel a píxel con el cálculo de una pasada completa. El error reducido describe la composición general, y deja fuera la resolución de detalle fino.

A 640 × 360, la mediana de la primera pasada de Menger bajó de 0,80 a 0,26 ms. En el interior con path tracing y dos mutaciones, las dos rutas empezaron a la misma resolución reducida, y la marcha de mayor precisión subió el coste de 1,02 a 1,73 ms. La pasada de métricas consumió entre 0,004 y 0,027 ms de GPU en las escenas de esta prueba. Los valores de validez alcanzaron el 100 % en las 1.024 posiciones comprobadas, así que estas escenas pequeñas no reproducen por sí solas todos los rayos agotados de la navegación real.

### Trazado GPU con objetivo de 1920 × 1080

| Escena | Anterior | ¼ por eje | ½ por eje | Completa |
| --- | ---: | ---: | ---: | ---: |
| Menger exterior | 4,47 ms | 0,83 ms | 1,45 ms | 4,72 ms |
| Mandelbulb con tres mutaciones | 2,47 ms | 2,77 ms | 5,23 ms | 15,95 ms |
| Interior con path tracing y dos mutaciones | 3,38 ms | 5,78 ms | 16,72 ms | 71,07 ms |

En el último caso, la pasada a mitad de resolución alcanzó 31,00 millones de rayos primarios válidos estimados por segundo. El modo anterior ya empleaba un cuarto por eje en las escenas mutadas. Sus tiempos reflejan el coste de la marcha abreviada que estaba activa durante la navegación.

## Comparación local de muestreo subpíxel

```powershell
cargo test --offline compare_local_sampling -- --ignored --nocapture
```

La prueba usa el trazador del proyecto a 256 × 144 y guarda los paneles en `target/sampling`. Compara 1, 4, 16 y 64 muestras por píxel. Cada método recibe el mismo número de rayos. Cambia el desplazamiento del rayo primario. Las otras dimensiones del muestreo conservan su secuencia. La referencia toma 256 muestras independientes por píxel.

Se comparan la secuencia actual, Sobol de dos dimensiones con desplazamiento digital por píxel y una máscara escalar de ruido azul espacio-temporal generada localmente por cada eje. La máscara de prueba mide 16 × 16 × 32 y se repite. La prueba de giro utiliza cuatro poses renderizadas por separado en la mutación Menger + Mandelbox. El cambio del error filtrado entre poses es un indicador de parpadeo. La prueba no mide la latencia de un giro interactivo.

En Menger, Mandelbulb y Menger + Mandelbox, R2 obtuvo el menor error con una muestra por píxel. Con 16 muestras y cámara quieta, Sobol redujo el error RMS un 9–11 % respecto a R2. En el giro probado, el cambio medio del error filtrado fue 0,00885 con R2, 0,01115 con Sobol y 0,01155 con la máscara local.

**Sampling → Pixel sampling** permite comparar **R2 (original)**, **Sobol** y **STBN (blue noise)** en el visor y en el PNG. R2 arranca seleccionado. Cambiar el método reinicia la acumulación de muestras del visor. La selección pertenece a la sesión y las tres opciones cambian el desplazamiento subpíxel del rayo primario. El artefacto local de STBN se integra en la compilación y se puede regenerar mediante `cargo test --offline generate_stbn_asset -- --ignored`.

## Guiado de rayos en el visor

**Sampling → Intelligent ray sampling** activa el guiado durante el refinamiento con cámara quieta. Las dos primeras muestras cubren todos los píxeles. Una pasada de GPU puntúa las discontinuidades de profundidad y normales, las marchas sin resolver y las diferencias de color entre ambas muestras. Una pareja puede trasladar un rayo de un píxel estable a otro con más incertidumbre. Las pasadas guiadas alternan con pasadas uniformes, y cada píxel calcula su media de color y de guías OIDN según su número real de muestras.

El presupuesto total de rayos primarios coincide con el muestreo uniforme y el contador de spp corresponde al promedio de la imagen. Cada rayo conserva su trazado completo. El mapa pertenece a la vista actual y se reconstruye al cambiar la cámara o la escena. El movimiento conserva la ruta adaptativa y el PNG conserva su muestreo propio. El selector arranca desactivado porque la mejora medida depende de la escena. Al activarlo, el visor continúa con muestreo uniforme mientras prepara los shaders guiados en segundo plano.

La prueba de ventana verifica que cada pareja conserva su presupuesto, también con dimensiones impares. Compara cuatro pasadas uniformes o guiadas con una referencia uniforme de 32 en Menger, Mandelbulb y Mandelbox + Menger, comprueba que el resultado guiado mejora al llegar a 16 pasadas y prueba OIDN con el guiado activado.

La prueba de giros usa cinco poses en una ventana de 300 × 180. Compara cada captura con una referencia de la misma pose y mide el error RGB medio después de reducir a 10 × 6. El umbral de 12 sobre 255 sirve como referencia de composición gruesa. El tiempo de captura va desde el envío al renderizador hasta disponer de los píxeles en CPU e incluye la lectura de la imagen. No mide la latencia del ratón a la pantalla.

### Validación de las fases 4 a 6

En la RTX 3090 con Vulkan, las tres escenas de la prueba subpíxel redujeron su error al acumular más muestras. Frente a la referencia de 256 muestras, R2 tuvo el menor error a 1 spp y Sobol lo tuvo a 16 y 64 spp en las tres escenas. La máscara STBN quedó cerca de Sobol a 16 spp, con diferencias que dependen de la escena.

El error RGB RMS del guiado a 4 spp, frente a una referencia uniforme de 32 spp, cambió de 2,439 a 2,432 en Menger, de 4,177 a 4,169 en Mandelbulb y de 6,664 a 6,672 en Mandelbox + Menger. A 16 spp guiadas el error bajó a 0,694, 1,168 y 2,275 respectivamente. La GPU tardó entre 0,006 y 0,008 ms en calcular el reparto de rayos para el caso a 149 × 89. Esa cifra corresponde a la pasada de asignación y excluye la copia de la primera muestra y el trazado de los rayos.

Las cinco series de giro quedaron bajo el umbral de error grueso en las cinco poses, con ambas rutas. La resolución adaptativa llegó a completa en los exteriores sencillos y se quedó a mitad por eje en las dos escenas mutadas. En Mandelbulb con tres mutaciones, el error grueso medio bajó de 1,55 a 1,20 sobre 255.

A 1920 × 1080, el interior con trazado de caminos tardó 5,84 ms a un cuarto por eje, 16,85 ms a mitad y 70,87 ms a resolución completa. A mitad se estimaron 30,77 millones de rayos primarios válidos por segundo. En estas escenas, el muestreo de validez devolvió el 100 % de las posiciones comprobadas. Esos datos no validan el comportamiento de todos los rayos agotados ni todos los presets.

## Muestreo MIS del HDRI

Al cargar un HDRI Radiance, el visor construye una tabla de alias de hasta 512 × 256 celdas, ponderada por luminancia y ángulo sólido. El 2 % de la distribución es uniforme sobre la esfera. El shader evalúa la radiancia en el panorama original y usa la rotación tanto para generar direcciones como para consultar su densidad. Los mapas negros y los de baja variación conservan el muestreo del material.

El refinamiento del visor y la exportación PNG combinan un rayo hacia el HDRI con el rebote del material, con la heurística de potencia y densidades por estereorradián. Ambos prueban la visibilidad mediante la marcha de rayos existente. El rayo hacia el panorama añade trabajo por rebote. Durante la navegación se conserva la ruta interactiva.

```powershell
cargo test --offline compare_hdri_importance_sampling -- --ignored --nocapture
```

La prueba compara el error de la imagen PNG frente a una referencia de 512 muestras y registra la mediana de diez renders intercalados por estrategia. Usa un panorama sintético con una fuente brillante, tres rebotes y la misma intensidad, material y rotación en ambos casos.

### HDRI en Ambient + AO

Ambient + AO usa el HDRI como iluminación difusa direccional. Para panoramas con zonas brillantes, cada muestra elige entre el hemisferio coseno y la distribución de luminancia del HDRI, con la densidad de la mezcla como peso. La misma dirección se usa en la consulta de AO. El radio y la intensidad de AO solo atenúan la contribución ambiental de esa muestra. La luz direccional conserva sus propias sombras. Esta ruta funciona en el visor y en la exportación PNG.
