# 📊 Análisis del Entrenamiento de Phi-2 para Tutoría Matemática
> *Un recorrido por cómo evolucionó el entrenamiento de nuestro modelo de IA*

## 📋 Resumen General

Entrenamos un modelo de inteligencia artificial (Phi-2) para que funcione como tutor de matemáticas. Aquí están los detalles básicos:

* **Modelo:** Microsoft Phi-2 (un modelo de lenguaje relativamente pequeño, con 1.5 mil millones de parámetros)
* **Computadora utilizada:** GPU NVIDIA GeForce GTX 1050 Ti (una tarjeta gráfica de gama media)
* **Tiempo que tardó:** 32.5 horas (1 día, 8 horas y 33 minutos)
* **Cantidad de ciclos completos:** 250 épocas


# 🔄 ¿Qué son las épocas en el entrenamiento de IA?

## Concepto básico

Una **época** en el entrenamiento de inteligencia artificial es un ciclo completo en el que el modelo procesa **todo** el conjunto de datos de entrenamiento una vez. Es como si el modelo "leyera" todo el libro de datos disponible de principio a fin.

## 📚 Analogía para entenderlo mejor

Imagina que estás estudiando para un examen importante con un libro de texto:

- **Una época** = Leer el libro completo una vez
- **Múltiples épocas** = Releer el mismo libro varias veces para entenderlo mejor

Cada vez que relees el libro (cada época), comprendes mejor el material, haces conexiones nuevas y recuerdas más detalles.

## 🧠 ¿Por qué se entrena en épocas?

### 1. Aprendizaje gradual

Los modelos de IA no suelen aprender todo en una sola pasada. Necesitan ver los datos múltiples veces para:
- Detectar patrones sutiles
- Reforzar lo aprendido
- Ajustar sus parámetros internos

### 2. Limitaciones prácticas

- La **memoria del ordenador** es limitada, no siempre puede procesar todos los datos a la vez
- Dividir en épocas permite procesar conjuntos de datos enormes

### 3. Seguimiento del progreso

- Las épocas proporcionan puntos de control naturales para evaluar cómo progresa el entrenamiento
- Al final de cada época, podemos medir el rendimiento y decidir si continuar

## 📊 ¿Cuántas épocas son necesarias?

Este es un número que varía según:

- **Complejidad del modelo**: Modelos más complejos pueden requerir más épocas
- **Cantidad de datos**: Con más datos, a veces se necesitan menos épocas
- **Tarea a aprender**: Tareas difíciles requieren más épocas
- **Calidad de los datos**: Datos de alta calidad pueden requerir menos épocas

En general, observamos que:

1. **Épocas iniciales**: Gran mejora (el modelo aprende conceptos básicos)
2. **Épocas intermedias**: Mejora moderada (el modelo refina lo aprendido)
3. **Épocas finales**: Mejora mínima o nula (punto de saturación)
## 🧠 ¿Qué pasó durante el entrenamiento?

Observamos que el entrenamiento se dividió naturalmente en tres etapas:

### 1️⃣ Etapa de Aprendizaje Rápido (Épocas 1-50)
![Evolución de la pérdida](https://github.com/user-attachments/assets/e5f4dd4a-d717-4dd6-98ed-06b7caab31a0)

* **Lo que pasó:** El modelo aprendió muy rápidamente al principio
* **Mejora:** 96% de todo el aprendizaje ocurrió aquí
* **Como si fuera:** Un estudiante aprendiendo las bases de un tema nuevo - los avances iniciales son enormes

### 2️⃣ Etapa de Refinamiento (Épocas 51-100)
* **Lo que pasó:** El modelo siguió mejorando, pero más lentamente
* **Mejora:** Aproximadamente un 3% adicional
* **Como si fuera:** Un estudiante que ya conoce los fundamentos y ahora está refinando detalles

### 3️⃣ Etapa de Estabilización (Épocas 101-250)
![Métricas combinadas](https://github.com/user-attachments/assets/acf16f27-aa84-4f24-9abd-b9b0293c8ddf)

* **Lo que pasó:** Pocas mejoras a pesar de ser la etapa más larga
* **Mejora:** Solo 1% adicional, a pesar de representar el 60% del tiempo total
* **Como si fuera:** Un estudiante repasando lo que ya sabe bien, con pequeñas mejoras ocasionales

## 💡 ¿Qué aprendimos?

1. **El modelo aprende principalmente al principio.** El 96% de la mejora ocurrió en el primer 20% del tiempo.
2. **Más tiempo no siempre significa mejores resultados.** Después de la época 100, las mejoras fueron mínimas.
3. **El hardware modesto es suficiente.** Una GPU de gama media pudo entrenar este modelo.
4. **El modelo final funciona bien.** La pérdida (error) se redujo exitosamente de 2.61 a 0.057.

## ✅ Recomendaciones para futuros entrenamientos

1. **Entrenar menos tiempo:** 100 épocas probablemente sean suficientes en lugar de 250
2. **Ajustar la velocidad de aprendizaje:** Podría optimizarse para concentrar más recursos en las fases tempranas
3. **Usar mejores ejemplos:** La calidad de los datos de entrenamiento podría mejorarse
4. **Evaluar periódicamente:** Verificar el progreso durante el entrenamiento para poder parar cuando se estabilice

## 📚 Glosario de Términos

| Término | Definición sencilla | Importancia |
|---------|---------------------|-------------|
| **Modelo** | Programa de IA entrenado para realizar tareas específicas | Es lo que estamos creando |
| **Phi-2** | Modelo de lenguaje pequeño creado por Microsoft | El modelo base que estamos adaptando |
| **GPU** | Procesador especializado que acelera cálculos matemáticos | Hardware necesario para el entrenamiento |
| **Época** | Un ciclo completo donde el modelo ve todos los datos de entrenamiento | Unidad para medir el progreso del entrenamiento |
| **Loss (Pérdida)** | Medida de cuánto se equivoca el modelo | Principal indicador de progreso (menor = mejor) |
| **Learning Rate (Tasa de aprendizaje)** | Qué tan grandes son los ajustes que hace el modelo | Determina la velocidad y estabilidad del aprendizaje |
| **Grad Norm (Norma del gradiente)** | Medida de cuánto cambian los parámetros del modelo | Indica momentos de ajustes importantes |
| **LoRA** | Técnica para ajustar modelos grandes usando menos recursos | Permite entrenar modelos grandes en GPU modestas |
| **Fine-tuning** | Proceso de adaptar un modelo pre-entrenado a una tarea específica | Lo que estamos haciendo con Phi-2 |
| **Parámetros** | Valores ajustables dentro del modelo | Los "conocimientos" del modelo (1.5B = 1,500 millones) |

## 📈 Los Números Importantes

| Métrica | Valor Inicial | Valor Final | Mejora |
|---------|---------------|-------------|--------|
| **Pérdida (Loss)** | 2.6141 | 0.057 | 97.82% |
| **Memoria GPU utilizada** | - | 3087.42 MB | 77% de 4GB disponibles |
| **Tiempo total** | - | 32.5 horas | - |
| **Muestras procesadas por segundo** | - | 0.137 | - |

## 🧪 ¿Qué Significan las Métricas que Vimos?

### Loss (Pérdida)
* **Qué es:** El error del modelo, cuánto se equivoca
* **Comportamiento observado:** Bajó rápidamente al principio, luego se estabilizó
* **Por qué importa:** Es la principal medida de qué tan bien está aprendiendo el modelo

### Grad Norm (Norma del Gradiente)
* **Qué es:** Indica qué tan grandes son los cambios en los parámetros del modelo
* **Comportamiento observado:** Tuvo picos altos al principio (especialmente en la época 25)
* **Por qué importa:** Nos muestra cuándo el modelo está haciendo ajustes importantes

### Learning Rate (Tasa de Aprendizaje)
* **Qué es:** Controla el tamaño de los ajustes que hace el modelo
* **Comportamiento observado:** Subió gradualmente hasta la época 31, luego descendió
* **Por qué importa:** Balancear entre aprender rápido y no desestabilizar el aprendizaje

## 💾 Detalles Técnicos (para referencia)

**Uso de Memoria GPU:**
* Memoria asignada: 2326.09 MB
* Memoria reservada: 3228.00 MB
* Memoria máxima asignada: 3087.42 MB

**Métricas finales:**
* epochs = 250.0
* total_flos = 100852232GF
* train_loss = 0.2402
* train_samples_per_second = 0.137
* train_steps_per_second = 0.009

**Ubicación del modelo guardado:** `models/phi2-matematicas-tutor`

---

*Este análisis está diseñado para ser accesible tanto para principiantes como para personas con experiencia en aprendizaje automático
