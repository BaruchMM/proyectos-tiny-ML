Parte A. Generación de modelos
Genere al menos las siguientes versiones:

Modelo	Estrategia de cuantización	Configuración
M1	Sin cuantización	Conversión directa a TensorFlow Lite, FP32
M2	Cuantización de rango dinámico	tf.lite.Optimize.DEFAULT sin conjunto representativo
M3	Cuantización Float16	tf.lite.Optimize.DEFAULT y supported_types=[tf.float16]
M4	Cuantización entera con interfaz Float32	Conjunto representativo, entrada y salida Float32
M5	Cuantización entera completa INT8	Conjunto representativo, operadores y entrada/salida INT8
Para M4 y M5, configure los operadores enteros compatibles mediante supported_ops. En M5, establezca explícitamente los tipos INT8 de los tensores de entrada y salida.

Conserve cada modelo generado en un archivo .tflite independiente.

Parte B. Análisis de la representación del modelo
Inspeccione cada archivo .tflite utilizando el analizador de TensorFlow Lite (tf.lite.experimental.Analyzer) o la herramienta Netron.

Identifique y registre:

Tamaño total del archivo FlatBuffer en bytes.

Número de tensores y operadores contenidos en el modelo.

Tipo de datos utilizado para almacenar los pesos de cada capa.

Tipo de datos de las entradas, salidas y tensores intermedios.

Operaciones adicionales introducidas durante la conversión, como QUANTIZE y DEQUANTIZE.

Parámetros de cuantización (scale y zero_point) y su relación con los rangos numéricos representables.

Compare las diferencias entre las representaciones FP32, Float16 e INT8.

Determine si todos los pesos y tensores fueron realmente cuantizados o si algunas operaciones conservaron representaciones de punto flotante.

Preste especial atención a la cuantización de rango dinámico, ya que el conversor puede conservar determinados tensores en punto flotante dependiendo de la arquitectura y los operadores del modelo.

Parte C. Evaluación del tamaño y la precisión
Evalúe las cinco versiones utilizando el mismo conjunto de prueba.

Elabore una tabla que contenga:

Modelo	Tamaño (bytes)	Reducción (%)	Tipo de entrada/salida	RMSE	MAE
M1	 	0	 	 	 
M2	 	 	 	 	 
M3	 	 	 	 	 
M4	 	 	 	 	 
M5	 	 	 	 	 
Calcule la reducción porcentual del tamaño de cada modelo respecto al modelo sin cuantización:

[R=\frac{M_{\mathrm{FP32}}-M_{\mathrm{cuantizado}}}{M_{\mathrm{FP32}}}\times100]

Genere una gráfica de barras que compare los tamaños de los archivos y otra que presente los errores obtenidos.

Determine cuál de las estrategias permite obtener el archivo más pequeño y cuál conserva mejor la precisión del modelo original.

Parte D. Despliegue y compatibilidad
Identifique cuáles de los modelos generados pueden ejecutarse utilizando los operadores disponibles en la biblioteca TensorFlow Lite Micro instalada.

Para los modelos compatibles, realice mediciones de memoria Flash, RAM y tiempo promedio de inferencia en Arduino.

No suponga que todas las estrategias son compatibles con el microcontrolador únicamente porque los modelos pueden ejecutarse mediante el intérprete TensorFlow Lite en Python.

En particular, distinga entre la precisión utilizada para almacenar los pesos y la precisión empleada durante la ejecución de las operaciones.

Compare los resultados y determine qué estrategia ofrece el mejor compromiso entre tamaño, precisión y costo computacional.

Análisis: Explique por qué dos modelos con la misma arquitectura y los mismos parámetros entrenados pueden producir archivos FlatBuffer con tamaños y estructuras diferentes. Analice también por qué utilizar tf.lite.Optimize.DEFAULT no garantiza, por sí solo, que todas las operaciones, entradas y salidas del modelo estén cuantizadas a INT8.

Actividad 8. Influencia del conjunto representativo en la cuantización INT8
La cuantización entera completa utiliza un conjunto representativo de datos para estimar los rangos de activación de la red neuronal.

En esta actividad se analizará cómo la selección de dichos datos modifica los parámetros de cuantización y, en consecuencia, el comportamiento del modelo.

Utilice la red neuronal entrenada originalmente en el intervalo:

[x\in[0,2\pi]]

Genere cuatro modelos INT8 a partir de los mismos pesos entrenados, cambiando únicamente el conjunto representativo utilizado durante la conversión.

Parte A. Generación de modelos con diferentes intervalos de calibración
Modelo	Intervalo de calibración
C1	([0,2\pi])
C2	([0,\pi])
C3	([\pi,2\pi])
C4	([0,\pi/2])
Mantenga constante el número de muestras representativas para aislar el efecto del intervalo de calibración.

Conserve las mismas opciones de cuantización entera completa y los tipos INT8 de entrada y salida en las cuatro conversiones.

Parte B. Análisis del FlatBuffer
Inspeccione los modelos generados e identifique los parámetros scale y zero_point asociados con sus tensores de entrada, salida y activaciones intermedias.

Compare los valores obtenidos y determine si se modifican los rangos numéricos representables.

Analice cómo se relacionan estos parámetros con el intervalo de calibración utilizado.

Verifique también si existen diferencias en el tamaño de los archivos FlatBuffer.

Parte C. Evaluación experimental
Evalúe los cuatro modelos utilizando un conjunto de prueba común que cubra todo el intervalo:

[x\in[0,2\pi]]

Calcule RMSE y MAE para cada uno.

Elabore una tabla comparativa con los resultados.

Genere una gráfica que presente la función seno real y las predicciones de los cuatro modelos.

Identifique las regiones en las que se presentan los mayores errores y analice su relación con los intervalos utilizados durante la calibración.

Compare también los resultados con los obtenidos por el modelo original de punto flotante.

Parte D. Análisis y discusión
Explique cómo influye el conjunto representativo en la cuantización de las activaciones y por qué un conjunto de calibración poco representativo puede degradar la precisión del modelo, aun cuando los pesos entrenados permanezcan constantes.

Determine si existen diferencias entre los errores originados durante el entrenamiento y aquellos introducidos exclusivamente por el proceso de cuantización.

Análisis final: Discuta por qué la cuantización debe considerarse una etapa de optimización que requiere evaluación experimental y no simplemente una transformación automática para reducir el tamaño de un modelo.

6. Cuestionario
Responda las siguientes diez preguntas utilizando los conceptos revisados en clase, la documentación técnica y los resultados obtenidos durante la práctica.

1. ¿Por qué resulta conveniente entrenar una red neuronal en una computadora y ejecutar únicamente las inferencias en un microcontrolador? ¿Qué limitaciones de memoria, procesamiento y energía justifican esta separación?

2. ¿Cuál es la función de los pesos, los sesgos y las funciones de activación de una red neuronal? ¿Qué ocurriría si se eliminaran las funciones de activación no lineales de las capas ocultas?

3. ¿Qué es FlatBuffers y por qué TensorFlow Lite utiliza este formato para almacenar modelos de aprendizaje automático? ¿Qué información contiene el archivo .tflite además de los pesos de la red neuronal?

4. ¿Cuáles son las diferencias entre cuantización de rango dinámico, cuantización Float16 y cuantización entera INT8? ¿Por qué estas estrategias pueden producir modelos con diferentes tamaños, precisión y compatibilidad de ejecución?

5. ¿Qué representan los parámetros scale y zero_point en un tensor cuantizado? ¿Cómo se utilizan para convertir valores entre las representaciones de punto flotante e INT8? ¿Qué ocurre cuando un valor de entrada excede el rango representable?

6. ¿Qué función cumple la tensor arena en TensorFlow Lite Micro? ¿Qué sucede si la memoria reservada es insuficiente? ¿Por qué el tamaño del archivo .tflite no permite determinar directamente la cantidad de RAM necesaria para ejecutar inferencias?

7. ¿Qué diferencias existen entre la memoria Flash y la memoria RAM de un microcontrolador? ¿Qué elementos de una aplicación TinyML se almacenan en cada una y cómo influyen en el tamaño del firmware?

8. ¿Cuál es la diferencia entre AllOpsResolver y MicroMutableOpResolver<N>? ¿Por qué registrar únicamente los operadores utilizados puede ayudar a optimizar una aplicación TinyML?

9. ¿Por qué es necesario realizar múltiples mediciones para estimar el tiempo de inferencia? ¿Qué factores del hardware, del software y de la representación numérica del modelo pueden producir diferencias entre los tiempos de ejecución?

10. ¿Cómo influyen el intervalo de entrenamiento y el conjunto representativo utilizado durante la cuantización en la precisión de un modelo? ¿Cómo pueden distinguirse los errores de generalización de los errores producidos por una calibración inadecuada?

7. Entregables
El estudiante deberá elaborar un reporte técnico que contenga los siguientes apartados.

7.1. Descripción del modelo
Arquitectura de la red neuronal.

Número total de parámetros.

Configuración del entrenamiento.

Descripción de los conjuntos de entrenamiento, validación y prueba.

7.2. Evaluación de precisión
Valores RMSE y MAE del modelo original.

Valores RMSE y MAE del modelo INT8.

Gráficas comparativas de las predicciones.

Análisis del efecto de la cuantización sobre la precisión.

7.3. Análisis de memoria
Tamaño de los modelos .tflite.

Memoria Flash utilizada por el firmware.

Memoria RAM estática reportada por el compilador.

Tamaño reservado para la tensor arena.

Discusión de las diferencias entre las cifras anteriores.

7.4. Evaluación del rendimiento
Resultados de al menos 100 mediciones de inferencia.

Tiempo mínimo, máximo y promedio.

Desviación estándar.

Análisis de los factores que influyen en el tiempo de ejecución.

7.5. Optimización de operadores
Identificación de los operadores presentes en el modelo.

Código implementado con MicroMutableOpResolver<N>.

Comparación del tamaño del firmware.

Comparación de los tiempos de inferencia.

Discusión de los resultados.

7.6. Generalización del modelo
Resultados correspondientes a los diferentes intervalos de entrenamiento.

Gráficas comparativas.

Análisis del comportamiento fuera del intervalo de entrenamiento.

7.7. Comparación de estrategias de cuantización
Archivos .tflite generados mediante las cinco estrategias.

Comparación de sus tamaños.

Comparación de precisión mediante RMSE y MAE.

Análisis de los tipos de datos, operadores y tensores.

Evidencias de inspección de los archivos FlatBuffer.

Evaluación de compatibilidad con TensorFlow Lite Micro.

7.8. Influencia del conjunto de calibración
Resultados de los cuatro modelos INT8.

Comparación de los parámetros scale y zero_point.

Gráficas de predicción.

Análisis de los errores introducidos por la calibración.

7.9. Cuestionario
1. ¿Por qué resulta conveniente entrenar una red neuronal en una computadora y ejecutar únicamente las inferencias en un microcontrolador? ¿Qué limitaciones de memoria, procesamiento y energía justifican esta separación?

2. ¿Cuál es la función de los pesos, los sesgos y las funciones de activación de una red neuronal? ¿Qué ocurriría si se eliminaran las funciones de activación no lineales de las capas ocultas?

3. ¿Qué es FlatBuffers y por qué TensorFlow Lite utiliza este formato para almacenar modelos de aprendizaje automático? ¿Qué información contiene el archivo .tflite además de los pesos de la red neuronal?

4. ¿Cuáles son las diferencias entre cuantización de rango dinámico, cuantización Float16 y cuantización entera INT8? ¿Por qué estas estrategias pueden producir modelos con diferentes tamaños, precisión y compatibilidad de ejecución?

5. ¿Qué representan los parámetros scale y zero_point en un tensor cuantizado? ¿Cómo se utilizan para convertir valores entre las representaciones de punto flotante e INT8? ¿Qué ocurre cuando un valor de entrada excede el rango representable?

6. ¿Qué función cumple la tensor arena en TensorFlow Lite Micro? ¿Qué sucede si la memoria reservada es insuficiente? ¿Por qué el tamaño del archivo .tflite no permite determinar directamente la cantidad de RAM necesaria para ejecutar inferencias?

7. ¿Qué diferencias existen entre la memoria Flash y la memoria RAM de un microcontrolador? ¿Qué elementos de una aplicación TinyML se almacenan en cada una y cómo influyen en el tamaño del firmware?

8. ¿Cuál es la diferencia entre AllOpsResolver y MicroMutableOpResolver<N>? ¿Por qué registrar únicamente los operadores utilizados puede ayudar a optimizar una aplicación TinyML?

9. ¿Por qué es necesario realizar múltiples mediciones para estimar el tiempo de inferencia? ¿Qué factores del hardware, del software y de la representación numérica del modelo pueden producir diferencias entre los tiempos de ejecución?

10. ¿Cómo influyen el intervalo de entrenamiento y el conjunto representativo utilizado durante la cuantización en la precisión de un modelo? ¿Cómo pueden distinguirse los errores de generalización de los errores producidos por una calibración inadecuada?

7.10. Conclusiones
Discusión sobre las ventajas, limitaciones y consideraciones necesarias para implementar modelos de aprendizaje automático en microcontroladores.

Las conclusiones deberán abordar los compromisos entre tamaño, precisión, memoria y tiempo de inferencia, así como la influencia de las diferentes estrategias de cuantización sobre estas características.

7.11. Archivos y evidencias
El estudiante deberá entregar:

Notebook de Python utilizado durante los experimentos.

Archivos .tflite generados.

Código fuente de Arduino.

Archivos model.h correspondientes.

Gráficas y tablas de resultados.

Evidencias de compilación y ejecución en el microcontrolador.

Reporte técnico en formato PDF.

Subir dos archivos, el reporte en pdf y el archivo zip con el resto de archivos para reproducir el proyecto.