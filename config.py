"""
Configuración para el proceso de fine-tuning de un modelo matemático
utilizando Microsoft Phi-2 (2.7B parámetros) optimizado para GPU de 4GB
"""

# Rutas
MODEL_PATH = "models/"
CHECKPOINT_PATH = "checkpoints/"
DATASET_PATH = "./dataset/matematica.json"

# Modelo base a utilizar - Phi-2 de Microsoft (2.7B parámetros)
BASE_MODEL = "microsoft/phi-2"  

# Configuración de entrenamiento optimizada para GPU con memoria limitada
LEARNING_RATE = 1e-2         # Tasa de aprendizaje ligeramente menor para más estabilidad
NUM_EPOCHS = 5                 # Más épocas para compensar batch size pequeño
BATCH_SIZE = 1                  # Batch size mínimo para ahorrar memoria
GRADIENT_ACCUMULATION_STEPS = 4  # Acumulación alta para simular batch size más grande
WARMUP_RATIO = 0.1              # Aumentar calentamiento
MAX_STEPS = -1                # Ajusta según el tamaño de tu dataset
SAVE_STEPS = 50                # Cada cuántos pasos guardar un checkpoint
LOGGING_STEPS = 25              # Logging más frecuente
MAX_LENGTH = 512                # Longitud máxima de contexto



# Configuración LoRA optimizada para Phi-2
LORA_R = 4                      # Rank reducido para ahorrar memoria
LORA_ALPHA = 8                 # Alpha
LORA_DROPOUT = 0.1             # Dropout para regularización
LORA_TARGET_MODULES = [         # Módulos específicos para Phi-2
    "q_proj",
    "k_proj", 
    "v_proj",
    "dense",
    
]

# Configuración de cuantización agresiva para ahorrar memoria
BITS = 4                      # Cuantización de 4-bit
DOUBLE_QUANT = True             # Doble cuantización para máximo ahorro de memoria
USE_NESTED_QUANT = True         # Cuantización anidada adicional

# Configuración del formato para conversación adaptada a Phi-2
CHAT_TEMPLATE = """<|user|>{{#each messages}}{{#ifEquals role "user"}}{{content}}{{/ifEquals}}{{/each}}<|assistant|>{{#each messages}}{{#ifEquals role "assistant"}}{{content}}{{/ifEquals}}{{/each}}"""

# Nombre del modelo fine-tuned
MODEL_NAME = "phi2-matematicas-tutor"

# Configuraciones adicionales para optimización de memoria
USE_FP16 = True                 # Usar precisión mixta (FP16) para ahorrar memoria
USE_CPU_OFFLOADING = True       # Habilitar offloading a CPU si es necesario
TORCH_COMPILE = False           # Desactivar torch.compile (puede causar problemas de memoria)
OPTIM = "adamw_bnb_8bit"      # Optimizador que ahorra memoria
LR_SCHEDULER = "linear"         # Scheduler de tasa de aprendizaje


