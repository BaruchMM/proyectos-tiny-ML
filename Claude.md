# Practica TinyML 2026: sen(x) en Arduino Nano 33 BLE Sense

Contexto para un agente de Claude Code. Idioma de trabajo: espanol.

## Quien es el usuario y como trabajar con el

- Maestria en IIMAS UNAM (PCIC). Sabe mucho de ML y de procesamiento de senales. NO sabe de circuitos ni microcontroladores: explicar Flash, RAM, Arduino, etc. desde cero y con analogias de ML.
- Le preocupa la precision tecnica: corrige imprecisiones. No afirmar como hecho lo que no se verifico; marcar suposiciones como suposiciones.
- Preferencias de estilo: prosa natural en las respuestas, sin emojis, sin LaTeX en el chat (no renderiza), sin listas excesivas. En archivos y salidas de consola usar solo ASCII (sin acentos ni simbolos especiales).
- Scripts de Python: sin argumentos de linea de comandos; configurar con constantes al inicio del archivo.
- No tiene la tarjeta Arduino: la tiene un companero. Compilar en Arduino IDE no necesita tarjeta (da Flash y RAM estatica); subir el programa y medir tiempos si.
- La fuente de trabajo es Proyecto_1.py. Proyecto_1.ipynb es el cuaderno de ayuda original y no se modifica.

## La tarea (resumen)

Aproximar sen(x) con una red 1-16-16-1 (tanh), convertirla a TensorFlow Lite con varias cuantizaciones, analizar el FlatBuffer, exportar model.h, correrla en Arduino con TensorFlow Lite Micro (biblioteca Chirale_TensorFlowLite 2.0.0, Arduino IDE 2.x, paquete Arduino Mbed OS Nano Boards) y medir memoria y tiempo.

Actividades:
1. Parametros y tamano: tabla por capa, bytes FP32 (parametros x 4) contra tamano real del .tflite, y por que difieren.
2. RMSE y MAE de float32 contra INT8, tabla y grafica.
3. Flash, RAM estatica y kTensorArenaSize en Arduino IDE (la arena es un arreglo global: ya esta contada en la RAM estatica, no sumarla de nuevo).
4. Tiempo de inferencia: micros() alrededor de interpreter->Invoke(), >= 100 inferencias tras calentamiento, sin Serial.print dentro de la region cronometrada; minimo, maximo, promedio, desviacion estandar.
5. AllOpsResolver contra MicroMutableOpResolver<N>: comparar Flash, RAM, tiempo y correccion.
6. Generalizacion: entrenar en [0, 2pi] y en [-pi, pi], evaluar en [-2pi, 2pi]; separar extrapolacion de la red (visible en float32) de saturacion de la cuantizacion INT8.
7. Cinco modelos M1 a M5 (FP32, rango dinamico, Float16, entera con E/S float32, entera completa INT8): tamano, reduccion %, RMSE, MAE, inspeccion del FlatBuffer (tipos, tensores, operadores, QUANTIZE/DEQUANTIZE, scale y zero_point), compatibilidad con TFLM.
8. Conjunto representativo: cuatro modelos INT8 calibrados en [0, 2pi], [0, pi], [pi, 2pi], [0, pi/2] con el mismo numero de muestras; comparar scale, zero_point, errores y tamano.

Entregables: reporte tecnico en PDF (secciones 7.1 a 7.11: modelo, precision, memoria, rendimiento, operadores, generalizacion, comparacion de cuantizaciones, calibracion, cuestionario de 10 preguntas, conclusiones, archivos y evidencias) y un zip con notebook, .tflite, .ino, model.h, graficas, tablas y evidencias de compilacion y ejecucion. Se suben dos archivos: el PDF y el zip.

## Entorno

- Colab: Python 3.13.x, TensorFlow 2.21.0, NumPy 2.1.3. Registrar versiones en el reporte.
- Para correr local: crear un entorno virtual con una version de Python que soporte esa version de TensorFlow, e instalar tensorflow, numpy, matplotlib. Verificar compatibilidad antes de asumir.
- tf.lite.Interpreter esta deprecado (avisa que se elimina); funciona. Alternativa: ai_edge_litert.interpreter.Interpreter.

## Estructura del proyecto

- Reporte: apartados 1 a 11 (equivalen a las secciones 7.1 a 7.11 del enunciado). Los apartados 1 a 8 coinciden con las Actividades 1 a 8.
- Proyecto_1.py: bloques 0.x con lo compartido y un bloque #%% por apartado.
- Salidas: resultados/<n>_<tema>/<n>_<fig|tab>_<contenido>, con modelos/ y evidencias/ dentro de cada apartado. Los sketches van en arduino/.
- 3_tab_arduino.csv y 4_tab_tiempos.csv no las sobrescribe Proyecto_1.py. compilar_sketches.py anota Flash y RAM y guarda 3_compilacion.txt; las columnas de la tarjeta salen de mediciones_tarjeta.txt.
- Orden: Proyecto_1.py, compilar_sketches.py y otra vez Proyecto_1.py para actualizar 5_tab_resolvers.csv y 7_tab_despliegue.csv.
- arduino/LEEME.txt son las instrucciones para quien tenga la Nano 33 BLE. El Nano prestado con CH340 (ATmega328P) no sirve.
- Los numeros del reporte salen de resultados/, no de la salida guardada del notebook.
- Al final se genera Proyecto_1_final.ipynb desde el script, porque el enunciado pide un notebook.

## Estado del trabajo

El estado vigente esta en checklist_proyecto.md. Lo que sigue en esta seccion describe la corrida de Colab.

Hecho en el notebook (corrido en Colab):
- Datos: 1600 de entrenamiento con ruido 0.015, 320 de validacion, 256 de prueba en [0, 2pi]; semilla 2026. Entrenamiento con Adam lr 0.005, batch 64, EarlyStopping paciencia 20 (paro en 118 epocas en la primera corrida).
- Conversion de M1 a M5 con un ciclo make_converter(estrategia); representative_data() con 256 puntos en [0, 2pi].
- Inspeccion con tf.lite.experimental.Analyzer y con la funcion collect_tensors (tabla por tensor, resumen, graficas de bytes por tipo y de rangos int8).
- Actividad 1: 321 parametros (32 + 272 + 17), 1284 bytes FP32.
- model.h por modelo y 10 carpetas de sketch Sketch_<M>_<allops|minops> con cronometraje.
- Experimento de red ancha (1-64-64-1) para probar el comportamiento de rango dinamico.
- Celdas de Actividad 8 (C1 a C4) y Actividad 6 (modelo B en [-pi, pi]) entregadas; verificar que corrieron sin error.

Pendiente:
- Compilar las 10 carpetas en Arduino IDE y anotar Flash y RAM estatica (Actividades 3 y 5). Algunos modelos pueden fallar en AllocateTensors o por operadores no soportados (sobre todo M2 y M3, no verificado): documentarlo como resultado valido.
- Sesion con la tarjeta (companero) para tiempos: sketch con MEDIR_TIEMPO en 1; capturas de compilacion y de monitor serie.
- Capturas de Netron de los cinco modelos.
- Verificar el modelo B de la Actividad 6: llego a 200 epocas sin parar; considerar epochs=400 para comparar justo con el modelo A.
- Reporte PDF, cuestionario y conclusiones; empaquetar el zip.

## Resultados obtenidos (corrida de Colab)

Tamanos del .tflite en bytes (ultima corrida): M1 3436, M2 3436, M3 3700, M4 4080, M5 3728. En una corrida anterior eran 3424, 3424, 3680, 4080, 3720; la diferencia viene probablemente de los nombres de los tensores (las capas pasaron a llamarse dense_9, dense_10, dense_11), no esta verificado.

Errores contra sen(x) en [0, 2pi] (corrida anterior; recalcular tras reentrenar): Keras float32 RMSE 0.01390, MAE 0.01178; M1 y M2 iguales; M3 RMSE 0.01420, MAE 0.01218; M4 y M5 RMSE 0.07501, MAE 0.05982, error maximo 0.243.

Parametros de cuantizacion de M5 (calibrado en [0, 2pi]): entrada scale 0.0246399, zero_point -128, rango [0, 6.2832]; salida scale 0.00791485, zero_point -2; activaciones antes de la tanh: scale 0.041064 (zp -4) y 0.018104 (zp -11); salidas de las tanh: scale 0.0078125 (1/128), zp 0. Pesos int8 con zero_point 0 y escala por canal; sesgos int32 con escala = escala_entrada x escala_peso.

## Hallazgos para el reporte

Verificados con la salida del notebook:
- M1 y M2 son identicos: Optimize.DEFAULT sin cuantizar nada en esta red.
- En una red 1-64-64-1, rango dinamico solo cuantizo a int8 el tensor de 4096 pesos (de 18708 a 7280 bytes, -61%); los de 64 elementos y todos los sesgos siguieron en float32. El umbral esta entre 257 y 4096 elementos; el valor exacto no esta medido. No se encontro una opcion publica del conversor para cambiarlo.
- M3: pesos y sesgos en float16, seis nodos DEQUANTIZE que reconstruyen float32; computo en float32; el archivo es mas grande que M1.
- M4: QUANTIZE al inicio y DEQUANTIZE al final, interior int8/int32, E/S float32. M4 y M5 tienen los mismos scale y zero_point internos y por eso el mismo error.
- M5: sin nodos extra, E/S int8; el codigo del microcontrolador debe cuantizar la entrada (q = round(x/scale) + zero_point, saturada a [-128, 127]) y descuantizar la salida.
- Los modelos cuantizados pesan mas que M1 aunque sus constantes bajan de 1284 a unos 420 bytes: el archivo es sobre todo estructura (en el Analyzer de M1, 59% del archivo es "Non-data buffer"). La explicacion probable es que los vectores de scale/zero_point por canal y los nodos extra pesan mas que el ahorro; no esta medido tensor por tensor.
- El error de INT8 (RMSE 0.075) es mucho mayor que el redondeo de entrada y salida; viene de las activaciones internas. Contribuyentes sugeridos, no medidos: resolucion 1/128 en la salida de cada tanh y pesos de la ultima capa de hasta 1.84 que amplifican el error.

Sin verificar (marcar como suposicion si se usan):
- Que el conversor extienda el rango de calibracion para incluir el 0 (explicaria que C3 de la Actividad 8 tenga la misma entrada que C1).
- Que la escala 1/128 de las salidas de tanh no dependa de la calibracion.
- Que TFLM no soporte M2 y M3 en Arduino.
- Que interpreter->arena_used_bytes() exista en Chirale 2.0.0 (hay un #define USE_ARENA_USED para apagarlo).
- El nRF52840 del Nano 33 BLE es Cortex-M4F con FPU de precision simple: no asumir que INT8 sea mas rapido; medirlo.

## Detalles de codigo que ya costaron errores

- tf.lite.Interpreter agrega un nodo DELEGATE (XNNPACK) a _get_ops_details(); no esta en el archivo. Filtrarlo siempre: [op['op_name'] for op in ops if op['op_name'] != 'DELEGATE']. Sin eso, el MicroMutableOpResolver<N> sale con N de mas y un TODO.
- collect_tensors(blob) devuelve (filas, operadores). Cada fila tiene: idx, rol (entrada, salida, activacion, constante), dtype, shape, n_el, bytes, n_scales, scale_max, scale_min, zp, lo, hi, es_bias. La clave 'scale' NO existe; se agrego un envoltorio que la copia de scale_max. En M3 los tensores producidos por DEQUANTIZE (pesos reconstruidos en float32) salen etiquetados como "activacion" porque la heuristica clasifica todo lo que sale de un operador asi.
- predict_tflite(blob, x, verbose=True, name='') maneja E/S float32 e int8 segun el modelo.
- _get_ops_details es API privada; si cambia, fallan collect_tensors y get_op_names.

## Sketches de Arduino

Plantilla SKETCH con marcadores @@NAME@@, @@RESOLVER_KIND@@, @@RESOLVER_INCLUDE@@, @@RESOLVER_DECL@@, @@ARENA@@ (24 KiB), @@NRUNS@@ (100). Se reemplazan desde Python; la plantilla pegada tal cual no compila. Cada carpeta Sketch_<M>_<allops|minops> contiene el .ino con el mismo nombre que la carpeta y su model.h. set_input() y get_output() ramifican por tipo de tensor, asi que sirven para M1 a M5. MEDIR_TIEMPO 0 = Serial Plotter con real y pred; 1 = cronometrar (min, max, promedio, desviacion) con micros() y Serial.print solo despues, mas RMSE, MAE y arena usada en una linea [tiempo]. Se genera en 1 desde la constante MEDIR_TIEMPO_SKETCH. Los errores se repiten como lineas [error] para no perderlos si el monitor abre tarde.

Operadores esperados en el MicroMutableOpResolver: M1 y M2: FullyConnected y Tanh (N=2); M3: agrega Dequantize (N=3); M4: agrega Quantize y Dequantize (N=4); M5: N=2.

## Como ayudar a continuar

1. Verificar que las celdas de Actividades 8 y 6 corren y recolectar sus tablas.
2. Armar la tabla de memoria con los numeros que reporte Arduino IDE (plantilla CSV de columnas: modelo, bytes_tflite, flash y RAM allops/minops, tiempos, compila_y_funciona).
3. Redactar el reporte siguiendo 7.1 a 7.11 y responder el cuestionario con los resultados reales, citando explicitamente que es medido y que es hipotesis.
4. No inventar numeros de la tarjeta: si faltan, dejar el campo marcado como pendiente.