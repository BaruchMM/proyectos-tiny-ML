# Checklist del proyecto TinyML: sen(x) en Arduino Nano 33 BLE Sense

Actualizado el 2026-10-30 tras reestructurar `Proyecto_1.py` y correrlo completo en local.

Como leerlo:

- `[x]` esta hecho y el archivo existe en `resultados/` o `arduino/`.
- `[ ]` falta. Si dice PARCIAL, existe algo pero no cumple el objetivo completo.
- (IDE) = Arduino IDE, no necesita tarjeta. (TARJETA) = sesion con una Nano 33 BLE; el Nano con CH340 prestado (ATmega328P, 2 KB de RAM) no sirve. (ANALISIS) = redactar, los datos ya existen. (DATOS TARJETA) = trabajo sin tarjeta que espera sus mediciones.
- Los apartados del reporte van del 1 al 11. Los apartados 1 a 8 coinciden con las Actividades 1 a 8.

`objetivos proyecto.md` no incluye el enunciado de las Actividades 1 a 6: empieza en la Parte A de la Actividad 7. Esas actividades salen aqui del resumen de `Claude.md` y de las secciones 7.1 a 7.6 del enunciado, asi que conviene cotejarlas con el original.

## Estado general

| Apartado | Datos de Python | Arduino | Analisis escrito |
|---|---|---|---|
| 1 Modelo | listos | no aplica | falta |
| 2 Precision | listos | no aplica | falta |
| 3 Memoria | listos | compilado; falta tarjeta | falta |
| 4 Rendimiento | no aplica | falta tarjeta | falta |
| 5 Operadores | listos | compilado; falta tarjeta | falta |
| 6 Generalizacion | listos | no aplica | falta |
| 7 Estrategias de cuantizacion | listos | compilado; falta tarjeta | falta |
| 8 Calibracion | listos | no aplica | falta |
| 9 Cuestionario | no aplica | no aplica | 0 de 10 |
| 10 Conclusiones | no aplica | no aplica | falta |
| 11 Archivos | falta notebook final | faltan evidencias | no aplica |

## Como se regenera todo

- `Proyecto_1.py` es la fuente de trabajo. `Proyecto_1.ipynb` es el cuaderno de ayuda original y no se modifica.
- Orden completo: correr `Proyecto_1.py`, luego `compilar_sketches.py` y otra vez `Proyecto_1.py`. La primera pasada regenera `resultados/` y `arduino/` (unos 4 minutos); la compilacion anota Flash y RAM; la ultima actualiza `5_tab_resolvers.csv` y `7_tab_despliegue.csv`.
- `3_tab_arduino.csv` y `4_tab_tiempos.csv` no las sobrescribe `Proyecto_1.py`. Las columnas de la tarjeta se llenan con las mediciones.
- Los numeros del reporte salen de `resultados/`, no de la salida guardada del notebook. Los tamanos del notebook vienen de un kernel con celdas reejecutadas.

## Apartado 1. Modelo

- [x] Versiones, configuracion de entrenamiento y conjuntos: `1_tab_entorno.csv`, `1_tab_entrenamiento.csv`, `1_tab_conjuntos.csv`.
- [x] Tabla por capa: 321 parametros (32 + 272 + 17), 1284 bytes FP32: `1_tab_parametros.csv`.
- [x] Figuras: `1_fig_datos.png`, `1_fig_curvas_aprendizaje.png`.

## Apartado 2. Precision

- [x] RMSE y MAE: float32 0.01390 / 0.01178; INT8 0.07501 / 0.05982: `2_tab_errores.csv`.
- [x] Figuras: `2_fig_predicciones.png`, `2_fig_error_absoluto.png`.
- [ ] (ANALISIS) Efecto de la cuantizacion sobre la precision.

## Apartado 3. Memoria

- [x] Bytes FP32 teoricos contra tamano real del `.tflite`: `3_tab_tflite_vs_fp32.csv`.
- [x] Sketches con `kTensorArenaSize` de 24 KiB: `arduino/`.
- [x] Entorno de compilacion: placa Arduino Nano 33 BLE, paquete `mbed_nano` 4.6.0 y `Chirale_TensorFLowLite` 2.0.0.
- [x] Las 10 carpetas compilan con `compilar_sketches.py`: Flash y RAM estatica en `3_tab_arduino.csv`, salida del compilador en `3_memoria/evidencias/3_compilacion.txt`. `arduino-cli` reprodujo exacto la medicion del IDE de M5 minops. La red no cambia: estos numeros valen para la familia Nano 33 BLE mientras no cambien los sketches ni las versiones.
- [ ] (TARJETA) Anotar `arena_used_bytes`. La llamada compila con Chirale 2.0.0; falta leer el valor en la tarjeta.
- [ ] (ANALISIS) Explicar por que difieren los bytes teoricos y el archivo.
- [ ] Discutir `.tflite`, Flash, RAM estatica y arena. La arena es un arreglo global: ya esta contada en la RAM estatica.

## Apartado 4. Rendimiento

- [x] Codigo de cronometraje: `micros()` alrededor de `Invoke()`, 100 inferencias tras calentamiento, sin `Serial.print` en la region medida. Los sketches ya salen con `MEDIR_TIEMPO` en 1 y cada linea `[tiempo]` trae tiempos, RMSE, MAE y arena usada. Compila, pero no se ha probado en una tarjeta.
- [x] Instrucciones para la sesion con la tarjeta: `arduino/LEEME.txt`.
- [ ] (TARJETA) Subir las 10 carpetas y copiar una linea de cada una a `mediciones_tarjeta.txt`.
- [ ] (DATOS TARJETA) Pasar esas lineas a `4_tab_tiempos.csv` y a `3_tab_arduino.csv`. El RMSE en placa debe parecerse a "RMSE malla placa (Python)" de `7_tab_despliegue.csv`.
- [ ] (DATOS TARJETA) Analizar que factores influyen en el tiempo. No asumir que INT8 sea mas rapido: el nRF52840 tiene FPU, hay que medirlo.

## Apartado 5. Operadores

- [x] Operadores por modelo: `5_tab_operadores.csv`.
- [x] Codigo con `MicroMutableOpResolver<N>`: N=2 en M1, M2 y M5; N=3 en M3; N=4 en M4.
- [x] Flash y RAM entre resolvers: `5_tab_resolvers.csv`.
- [ ] (TARJETA) Comparar tiempo de inferencia y correccion de resultados.
- [ ] (DATOS TARJETA) Discutir resultados.

## Apartado 6. Generalizacion

- [x] Modelo B entrenado con `EPOCHS_B = 400`: EarlyStopping lo detuvo en la epoca 263, ya no llega al tope.
- [x] Error dentro y fuera del intervalo, float32 e INT8: `6_tab_dentro_fuera.csv`.
- [x] Figura: `6_fig_modelos_a_b.png`.
- [ ] (ANALISIS) Separar extrapolacion de la red (visible en float32) de saturacion de la cuantizacion INT8.

## Apartado 7. Estrategias de cuantizacion

### Parte A. Generacion de modelos

- [x] M1 a M5, cada uno en su `.tflite`: `7_cuantizacion/modelos/`.
- [x] `supported_ops` en M4 y M5; tipos INT8 explicitos de entrada y salida en M5.

### Parte B. Analisis de la representacion

- [x] Salida del `Analyzer` por modelo: `7_cuantizacion/evidencias/7_analyzer_M1.txt` a `M5`.
- [x] Tamano, tensores, operadores y tipos: `7_tab_resumen.csv`.
- [x] Tabla por tensor con `scale`, `zero_point` y rango representable: `7_tab_tensores.csv`.
- [x] Figuras: `7_fig_bytes_por_tipo.png`, `7_fig_rangos_int8.png`.
- [x] Rango dinamico en red 1-64-64-1: `7_tab_red_ancha_tamanos.csv`, `7_tab_red_ancha_constantes.csv`.
- [ ] (ANALISIS) Comparar las representaciones FP32, Float16 e INT8.
- [ ] (ANALISIS) Determinar que se cuantizo realmente y que siguio en punto flotante.
- [ ] Capturas de Netron. No requieren Arduino: se abren los `.tflite` en Netron. El enunciado pide Analyzer o Netron, asi que son evidencia extra; Claude.md las lista como pendientes.

### Parte C. Tamano y precision

- [x] Tabla unica con tamano, reduccion, tipo de E/S, RMSE y MAE: `7_tab_comparacion.csv`.
- [x] Figuras: `7_fig_tamanos.png`, `7_fig_errores.png`, `7_fig_predicciones.png`.
- [ ] (ANALISIS) Decidir el mas pequeno y el mas preciso. M1 y M2 ganan ambos: ninguna estrategia redujo el archivo (M3 -7.4%, M4 -18.9%, M5 -8.3%).

### Parte D. Despliegue y compatibilidad

- [ ] (TARJETA) Identificar que modelos corren en TFLM. Compilar no basta: `AllocateTensors` e `Invoke` ocurren en ejecucion. Prediccion sin medir: M3 fallaria en `AllocateTensors`, porque `DequantizePrepare` (`dequantize_common.cpp`, lineas 44 a 46 de Chirale 2.0.0) solo acepta entrada int8, int16 o uint8, y M3 usa float16.
- [x] Flash y RAM de los cinco modelos: `7_tab_despliegue.csv`.
- [ ] (TARJETA) Tiempo promedio de los modelos compatibles.
- [ ] (ANALISIS) Distinguir precision de almacenamiento de precision de ejecucion (M3 guarda float16 y calcula en float32).
- [ ] (DATOS TARJETA) Determinar el mejor compromiso entre tamano, precision y costo.
- [ ] (ANALISIS) Explicar por que misma arquitectura y mismos parametros dan FlatBuffers distintos.
- [ ] (ANALISIS) Explicar por que `Optimize.DEFAULT` no garantiza cuantizar todo a INT8.

## Apartado 8. Calibracion

### Parte A. Generacion

- [x] C1 a C4 con `N_REP = 256` y las mismas opciones: `8_calibracion/modelos/`.

### Parte B. Analisis del FlatBuffer

- [x] `scale`, `zero_point` y tamanos: `8_tab_parametros.csv`, `8_tab_activaciones.csv`. Los cuatro archivos pesan 3696 bytes.
- [ ] (ANALISIS) Comparar rangos representables y relacionarlos con el intervalo de calibracion.

### Parte C. Evaluacion experimental

- [x] RMSE y MAE sobre [0, 2pi], con referencia float32: `8_tab_errores.csv`.
- [x] Figuras: `8_fig_predicciones.png`, `8_fig_error_absoluto.png`.
- [x] RMSE por cuadrante y ubicacion del error maximo: `8_tab_regiones.csv`.
- [ ] (ANALISIS) Relacionar las regiones de mayor error con los intervalos de calibracion.

### Parte D. Analisis y discusion

- [ ] (ANALISIS) Explicar como influye el conjunto representativo y por que uno poco representativo degrada la precision con los mismos pesos.
- [ ] (ANALISIS) Separar errores de entrenamiento de errores de cuantizacion (columna "RMSE cuant." de `8_tab_errores.csv`).
- [ ] (ANALISIS) Analisis final: por que la cuantizacion requiere evaluacion experimental.

## Apartado 9. Cuestionario

- [ ] 1. Entrenar en computadora e inferir en microcontrolador.
- [ ] 2. Pesos, sesgos y activaciones; que pasa sin no linealidades.
- [ ] 3. FlatBuffers y contenido del `.tflite`.
- [ ] 4. Rango dinamico, Float16 e INT8.
- [ ] 5. `scale`, `zero_point` y valores fuera de rango.
- [ ] 6. Tensor arena y por que el `.tflite` no determina la RAM.
- [ ] 7. Flash contra RAM.
- [ ] 8. AllOpsResolver contra MicroMutableOpResolver.
- [ ] 9. Por que medir muchas veces y que afecta el tiempo.
- [ ] 10. Intervalo de entrenamiento, conjunto representativo y como separar sus errores.

## Apartado 10. Conclusiones

- [ ] Ventajas, limitaciones y consideraciones para desplegar en microcontroladores.
- [ ] (DATOS TARJETA) Compromisos entre tamano, precision, memoria y tiempo.

## Apartado 11. Archivos y entrega

Se suben dos archivos: el PDF y el zip. El zip se arma con `ARMAR_ZIP = True` en `Proyecto_1.py` y no incluye `venv/`.

- [x] Archivos `.tflite`: `resultados/*/modelos/`.
- [x] Codigo de Arduino y `model.h`: `arduino/`.
- [x] Graficas y tablas como archivos: `resultados/`.
- [ ] Generar `Proyecto_1_final.ipynb` desde `Proyecto_1.py` con sus salidas. El enunciado pide un notebook y el original no se toca. El zip avisa si falta.
- [x] Evidencias de compilacion: `3_memoria/evidencias/3_compilacion.txt`.
- [ ] (TARJETA) Evidencias de ejecucion en `4_rendimiento/evidencias/`: `mediciones_tarjeta.txt` y capturas.
- [ ] Reporte en PDF.
- [ ] Armar el zip al final.

## Puntos que cambian lo escrito en Claude.md

- "M1 y M2 son identicos": tienen el mismo tamano y sus 12 tensores coinciden en nombre, tipo y valores, pero no son identicos byte a byte. Difieren 17 bytes entre las posiciones 344 y 389, fuera de los tensores. Hipotesis no verificada: es el metadato de conversion, porque el valor 0x03EA (1002) solo aparece en M2.
- Red 1-64-64-1: se confirma que rango dinamico solo cuantiza el tensor de 4096 pesos. Los bytes cambian: ahora 19544 a 8120 (-58.5%), contra 18708 a 7280 (-61%) en Claude.md. El codigo original de ese experimento no se conservo, asi que no se sabe que produjo la diferencia. El umbral exacto sigue sin medirse.
- Tamanos entre corridas: 3436, 3436, 3700, 4080, 3728 en Colab contra 3412, 3412, 3664, 4056, 3696 en una sesion nueva. La hipotesis de los nombres queda mejor sustentada, pero no demostrada: toda la diferencia cae en la parte no-datos y los nombres de tensores son 20 a 28 caracteres mas largos en Colab, aunque las cuentas no cierran exacto (M5 difiere 32 bytes con 20 caracteres), lo que podria ser relleno de alineacion.
- Dos puntos "sin verificar" ya tienen datos: C3 tiene la misma entrada que C1 (coincide con extender el rango hasta 0), y la escala 1/128 de las salidas de tanh se mantuvo en C1 a C4. Son cuatro casos, no una regla general.
