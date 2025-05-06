#requerimientos.txt

# Bibliotecas principales para fine-tuning de Phi-2
#torch>=2.0.0
#transformers>=4.34.0
#peft>=0.5.0
#accelerate>=0.21.0
#bitsandbytes>=0.41.0
#datasets>=2.14.0

# Procesamiento de datos
#pandas>=1.5.0
#numpy>=1.24.0
#tqdm>=4.66.0

# Visualización y UI para evaluate.py
#rich>=13.5.0

# Utilidades
#matplotlib>=3.7.0
#fsspec>=2023.0.0
#huggingface_hub>=0.18.0
#safetensors>=0.4.0



#Microsoft-Phi2/
#│
#├── dataset/
#│   └── matematicas.json  # Tu dataset ya está aquí
#│
#├── models/               # Carpeta vacía (se llenará durante el entrenamiento)
#│
#├── checkpoints/          # Carpeta vacía (se llenará durante el entrenamiento)
#│
#├── config.py             # Configuración adaptada para Phi-2
#├── utils.py              # Funciones de utilidad
#├── datamodule.py         # Manejo de datos
#├── train.py              # Script principal de entrenamiento
#└── evaluate.py           # Para evaluar el modelo después del entrenamiento