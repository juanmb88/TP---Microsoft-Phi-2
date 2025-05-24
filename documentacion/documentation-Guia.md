# Guía para Fine-Tuning de LLM con Dataset de Conceptos Matemáticos

Esta guía proporcionará los pasos necesarios para realizar el fine-tuning de un modelo de lenguaje grande (LLM) utilizando un dataset de conceptos matemáticos en formato de conversación.

## Estructura de Carpetas

Mantener la siguiente estructura de carpetas para el proyecto:

```
mi-proyecto-finetuning/
├── dataset/
│   └── matematicas.json  # Aca va el dataset de conceptos matemáticos
├── models/
│   └── (aquí se guardarán los modelos)
├── checkpoints/
│   └── (aquí se guardarán los checkpoints)
├── config.py
├── datamodule.py
├── train.py
├── utils.py
├── evaluate.py
└── requirements.txt
```

## Paso 1: Configuración (`config.py`)

Este archivo de configuración `config.py` es para adaptarlo al dataset y especificar los hiperparámetros de entrenamiento.
- Actúa como un repositorio central de configuraciones del proyecto.
- Contiene constantes y parámetros que se usarán en todo el proyecto.
- Centraliza aspectos como rutas, hiperparámetros y configuraciones específicas.
- Facilita cambiar la configuración en un solo lugar sin tocar el código en múltiples archivos.

## Paso 2: Utilidades (`utils.py`)

Este archivo mantiene todas o la mayoría de las funciones en `utils.py`, pero añade soporte para el formato de chat y funciones adicionales como `load_model_from_gguf` y `validate_math_response`.

- `load_model_from_gguf`: Ayuda a cargar modelos en formato GGUF (el formato utilizado por LM Studio y el modelo Micreosoft-_Phi2).
- `validate_math_response`: Una función básica para validar respuestas matemáticas, útil durante la evaluación del modelo.

## Paso 3: Preparación de Datos (`datamodule.py`)

Este acrchivo  `datamodule.py`  es para manejar tu formato específico de datos, incluyendo la carga del dataset JSON, preprocesamiento, tokenización y división en conjuntos de entrenamiento y prueba.

El flujo de trabajo de este módulo sería:

1. Cargar el dataset JSON usando `prepare_conversations_dataset()`.
2. Tokenizar los datos con `get_tokenized_dataset()`.
3. Opcionalmente, dividir en conjuntos de entrenamiento y prueba con `split_dataset()`.
4. Analizar estadísticas con `analyze_dataset_stats()`.


## Paso 4: Script de Entrenamiento (`train.py`)

 `train.py` es para cargar el modelo base, aplicar la técnica de adaptación LoRA, preparar el dataset tokenizado y ejecutar el proceso de entrenamiento.

Este es el componente central del proyecto de fine-tuning de un modelo de lenguaje para conceptos matemáticos. Su función principal es tomar un modelo base pre-entrenado y ajustarlo utilizando un dataset específico de conversaciones matemáticas. este es el archivo que se corre por terminal para evaluar si esta todo correctamente instalado y funcional sobre la IA.


## Paso 5: Script de Evaluación (`evaluate.py`)

Actualiza el script de evaluación `evaluate.py` para permitir probar el modelo de manera interactiva una vez finalizado el entrenamiento, con soporte para mantener una conversación en formato de chat. este es el archivo que se corre para ejecutar la IA.

## Paso 6: Preparar el Dataset

Asegúrate de que tu archivo JSON `matematicas.json` esté ubicado en la carpeta `dataset/` y tenga la estructura adecuada con pares de mensajes "user" y "assistant".

## Paso 7: Configurar Requirements


Resumen dependencias necesarias especificadas en el archivo `requirements.txt`.

## Paso 8: Ejecutar el Fine-Tuning

Ejecuta el script de entrenamiento `train.py` para adaptar el modelo base a tu dataset de conceptos matemáticos.

## Paso 9: Probar el Modelo Entrenado

Una vez completado el entrenamiento, prueba tu asistente matemático ejecutando el script `evaluate.py` y manteniendo una conversación interactiva con el modelo.

## Consideraciones Específicas para tu Dataset Matemático

- Considera usar un modelo base especializado en matemáticas, como Microsoft-Phi2.
- Utiliza una temperatura de generación más baja (0.3-0.5) durante la inferencia para obtener respuestas matemáticas precisas.
- Verifica que el modelo no genere fórmulas incorrectas después del entrenamiento.
- Asegúrate de que la longitud de contexto (`MAX_LENGTH`) sea suficiente para capturar explicaciones matemáticas completas.
- Considera aumentar el número de épocas de entrenamiento si tu dataset es relativamente pequeño.



