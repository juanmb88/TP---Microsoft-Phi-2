# Componentes del proyecto

## config.py - Variables de entorno

- Actúa como un repositorio central de configuraciones del proyecto.
- Contiene constantes y parámetros que se usarán en todo el proyecto.
- Centraliza aspectos como rutas, hiperparámetros y configuraciones específicas.
- Facilita cambiar la configuración en un solo lugar sin tocar el código en múltiples archivos.

## utils.py - Funciones de utilidad

Este archivo contiene funciones útiles para el proyecto, incluyendo dos nuevas funciones:

- `load_model_from_gguf`: Ayuda a cargar modelos en formato GGUF (el formato utilizado por LM Studio y el modelo Mathstral).
- `validate_math_response`: Una función básica para validar respuestas matemáticas, útil durante la evaluación del modelo.

Además, el archivo mantiene otras funciones que cubren las necesidades del proyecto, como:

- Crear directorios necesarios
- Cargar el tokenizador
- Calcular parámetros entrenables
- Formatear conversaciones
- Cargar datasets JSON
- Guardar adaptadores LoRA
- Encontrar capas lineales

## datamodule.py - Módulo de manejo de datos

Este módulo se encarga de la carga, preprocesamiento y manejo de los datos del proyecto. Sus funcionalidades principales incluyen:

1. Carga de datos:
   - Funciones para cargar el dataset JSON de conceptos matemáticos.
   - Soporte para dos formatos: pares instrucción-respuesta y conversaciones completas.
   - `prepare_dataset()`: Carga el dataset JSON y lo convierte a pares de instrucción-respuesta.

2. Preprocesamiento:
   - Tokenización de los textos usando el tokenizador del modelo.
   - Formateo adecuado según el template de chat definido en `config.py`.
   - Preparación de etiquetas para el entrenamiento.
   - `preprocess_function()`: Tokeniza los pares instrucción-respuesta.
   - `preprocess_conversations()`: Tokeniza las conversaciones completas.

3. Manejo del dataset:
   - División en conjuntos de entrenamiento y prueba.
   - Análisis estadístico del dataset (longitudes, palabras clave).
   - `get_tokenized_dataset()`: Obtiene el dataset tokenizado listo para entrenamiento.
   - `split_dataset()`: Divide los datos en conjuntos de entrenamiento y evaluación.
   - `analyze_dataset_stats()`: Analiza estadísticas básicas del dataset (longitudes, temas, etc.).

4. Análisis de contenido:
   - Detección de temas matemáticos frecuentes en el dataset.
   - Estadísticas sobre longitudes de preguntas y respuestas.

El flujo de trabajo típico con este módulo sería:

1. Cargar el dataset JSON usando `prepare_conversations_dataset()`.
2. Tokenizar los datos con `get_tokenized_dataset()`.
3. Opcionalmente, dividir en conjuntos de entrenamiento y prueba con `split_dataset()`.
4. Analizar estadísticas con `analyze_dataset_stats()`.

## Configuración de los drivers de la GPU

En este punto, fue necesario configurar los drivers de la placa de video para aprovechar la aceleración por GPU. El diagnóstico inicial mostró que CUDA no estaba disponible para PyTorch. Para solucionar esto, se siguieron los siguientes pasos:

1. Instalar los drivers NVIDIA más recientes para la GPU.
2. Reinstalar PyTorch con soporte CUDA usando el comando apropiado de pytorch.org:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verificar que CUDA Toolkit esté instalado (versión compatible con PyTorch).
4. Asegurarse de que la GPU sea compatible con la versión de CUDA requerida.

## train.py - Script de entrenamiento

Este es el componente central del proyecto de fine-tuning de un modelo de lenguaje para conceptos matemáticos. Su función principal es tomar un modelo base pre-entrenado y ajustarlo utilizando un dataset específico de conversaciones matemáticas.

El flujo de ejecución de `train.py` es el siguiente:

1. Detección de hardware: Identifica si hay GPU disponible para acelerar el entrenamiento.
2. Preparación del entorno: Crea los directorios necesarios para guardar el modelo y checkpoints.
3. Carga del modelo base: Intenta cargar un modelo de lenguaje grande de acceso libre (como Falcon-7B, OPT-6.7B o similar) como punto de partida.
4. Aplicación de LoRA: Implementa la técnica Low-Rank Adaptation, que permite ajustar el modelo de manera eficiente sin modificar todos sus parámetros, reduciendo drásticamente los requisitos de memoria y tiempo.
5. Carga y preprocesamiento del dataset: Prepara los datos de conversaciones matemáticas, los tokeniza y los divide en conjuntos de entrenamiento y evaluación.
6. Configuración del entrenamiento: Establece hiperparámetros como la tasa de aprendizaje, tamaño de batch y número de épocas según la configuración definida en `config.py`.
7. Proceso de entrenamiento: Ejecuta el ciclo de entrenamiento donde el modelo aprende de los ejemplos proporcionados, con evaluaciones periódicas.
8. Guardado del modelo: Al finalizar, guarda los adaptadores LoRA y el tokenizador en la ubicación especificada para uso posterior.

Durante este proceso, se requirió ejecutar el comando `huggingface-cli login --token "mi_token_aqui"` para autenticarse con Hugging Face y acceder a los modelos restringidos.

## evaluate.py - Script de evaluación

Este script permitirá probar el modelo de manera interactiva una vez finalizado el entrenamiento. Sus características principales incluyen:

- Modo interactivo: Permite tener una conversación en tiempo real con el modelo matemático.
- Carga inteligente del modelo: Detecta automáticamente si es un modelo completo o un adaptador LoRA.
- Soporte para GPU/CPU: Optimiza la inferencia según el hardware disponible.
- Interfaz mejorada: Utiliza la biblioteca `rich` para una visualización más agradable.
- Manejo de errores robusto: Proporciona mensajes claros si hay problemas.

Además, `evaluate.py` ofrece funcionalidades adicionales como:

- Guardar conversaciones a archivos JSON.
- Iniciar nuevas conversaciones.
- Evaluar por lotes con un conjunto de preguntas predefinidas.

