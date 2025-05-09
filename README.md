📊 Análisis del Entrenamiento de Phi-2 para Tutoría Matemática

Un recorrido por cómo evolucionó el entrenamiento de nuestro modelo de IA

📋 Resumen General
Entrenamos un modelo de inteligencia artificial (Phi-2) para que funcione como tutor de matemáticas. Aquí están los detalles básicos:

Modelo: Microsoft Phi-2 (un modelo de lenguaje relativamente pequeño, con 1.5 mil millones de parámetros)
Computadora utilizada: GPU NVIDIA GeForce GTX 1050 Ti (una tarjeta gráfica de gama media)
Tiempo que tardó: 32.5 horas (1 día, 8 horas y 33 minutos)
Cantidad de ciclos completos: 250 épocas

🧠 ¿Qué pasó durante el entrenamiento?
Observamos que el entrenamiento se dividió naturalmente en tres etapas:
1️⃣ Etapa de Aprendizaje Rápido (Épocas 1-50)
Mostrar imagen

Lo que pasó: El modelo aprendió muy rápidamente al principio
Mejora: 96% de todo el aprendizaje ocurrió aquí
Como si fuera: Un estudiante aprendiendo las bases de un tema nuevo - los avances iniciales son enormes

2️⃣ Etapa de Refinamiento (Épocas 51-100)

Lo que pasó: El modelo siguió mejorando, pero más lentamente
Mejora: Aproximadamente un 3% adicional
Como si fuera: Un estudiante que ya conoce los fundamentos y ahora está refinando detalles

3️⃣ Etapa de Estabilización (Épocas 101-250)
Mostrar imagen

Lo que pasó: Pocas mejoras a pesar de ser la etapa más larga
Mejora: Solo 1% adicional, a pesar de representar el 60% del tiempo total
Como si fuera: Un estudiante repasando lo que ya sabe bien, con pequeñas mejoras ocasionales

💡 ¿Qué aprendimos?

El modelo aprende principalmente al principio. El 96% de la mejora ocurrió en el primer 20% del tiempo.
Más tiempo no siempre significa mejores resultados. Después de la época 100, las mejoras fueron mínimas.
El hardware modesto es suficiente. Una GPU de gama media pudo entrenar este modelo.
El modelo final funciona bien. La pérdida (error) se redujo exitosamente de 2.61 a 0.057.

✅ Recomendaciones para futuros entrenamientos

Entrenar menos tiempo: 100 épocas probablemente sean suficientes en lugar de 250
Ajustar la velocidad de aprendizaje: Podría optimizarse para concentrar más recursos en las fases tempranas
Usar mejores ejemplos: La calidad de los datos de entrenamiento podría mejorarse
Evaluar periódicamente: Verificar el progreso durante el entrenamiento para poder parar cuando se estabilice

📚 Glosario de Términos
TérminoDefinición sencillaImportanciaModeloPrograma de IA entrenado para realizar tareas específicasEs lo que estamos creandoPhi-2Modelo de lenguaje pequeño creado por MicrosoftEl modelo base que estamos adaptandoGPUProcesador especializado que acelera cálculos matemáticosHardware necesario para el entrenamientoÉpocaUn ciclo completo donde el modelo ve todos los datos de entrenamientoUnidad para medir el progreso del entrenamientoLoss (Pérdida)Medida de cuánto se equivoca el modeloPrincipal indicador de progreso (menor = mejor)Learning Rate (Tasa de aprendizaje)Qué tan grandes son los ajustes que hace el modeloDetermina la velocidad y estabilidad del aprendizajeGrad Norm (Norma del gradiente)Medida de cuánto cambian los parámetros del modeloIndica momentos de ajustes importantesLoRATécnica para ajustar modelos grandes usando menos recursosPermite entrenar modelos grandes en GPU modestasFine-tuningProceso de adaptar un modelo pre-entrenado a una tarea específicaLo que estamos haciendo con Phi-2ParámetrosValores ajustables dentro del modeloLos "conocimientos" del modelo (1.5B = 1,500 millones)
📈 Los Números Importantes
MétricaValor InicialValor FinalMejoraPérdida (Loss)2.61410.05797.82%Memoria GPU utilizada-3087.42 MB77% de 4GB disponiblesTiempo total-32.5 horas-Muestras procesadas por segundo-0.137-
🧪 ¿Qué Significan las Métricas que Vimos?
Loss (Pérdida)

Qué es: El error del modelo, cuánto se equivoca
Comportamiento observado: Bajó rápidamente al principio, luego se estabilizó
Por qué importa: Es la principal medida de qué tan bien está aprendiendo el modelo

Grad Norm (Norma del Gradiente)

Qué es: Indica qué tan grandes son los cambios en los parámetros del modelo
Comportamiento observado: Tuvo picos altos al principio (especialmente en la época 25)
Por qué importa: Nos muestra cuándo el modelo está haciendo ajustes importantes

Learning Rate (Tasa de Aprendizaje)

Qué es: Controla el tamaño de los ajustes que hace el modelo
Comportamiento observado: Subió gradualmente hasta la época 31, luego descendió
Por qué importa: Balancear entre aprender rápido y no desestabilizar el aprendizaje

💾 Detalles Técnicos (para referencia)
Uso de Memoria GPU:

Memoria asignada: 2326.09 MB
Memoria reservada: 3228.00 MB
Memoria máxima asignada: 3087.42 MB

Métricas finales:

epochs = 250.0
total_flos = 100852232GF
train_loss = 0.2402
train_samples_per_second = 0.137
train_steps_per_second = 0.009

Ubicación del modelo guardado: models/phi2-matematicas-tutor

Este análisis está diseñado para ser accesible tanto para principiantes como para personas con experiencia en aprendizaje automático
