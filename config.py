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
#LEARNING_RATE = 1e-4            # Tasa de aprendizaje ligeramente menor para más estabilidad
#NUM_EPOCHS = 5                  # Más épocas para compensar batch size pequeño
#BATCH_SIZE = 1                  # Batch size mínimo para ahorrar memoria
#GRADIENT_ACCUMULATION_STEPS = 16  # Acumulación alta para simular batch size más grande
#WARMUP_STEPS = 100              # Pasos de calentamiento
#MAX_STEPS = 1000                # Ajusta según el tamaño de tu dataset
#SAVE_STEPS = 200                # Cada cuántos pasos guardar un checkpoint
#LOGGING_STEPS = 25              # Logging más frecuente
#MAX_LENGTH = 512                # Longitud máxima de contexto

LEARNING_RATE = 8e-5            # Ajustado para mejor convergencia
NUM_EPOCHS = 15                 # Reducido de 250 a 15 épocas (suficiente con LoRA)
BATCH_SIZE = 2                  # Aumentado si es posible con tu GPU
GRADIENT_ACCUMULATION_STEPS = 8  # Ajustado considerando batch_size nuevo
WARMUP_RATIO = 0.03             # En lugar de warmup_steps, usar ratio (3%)
MAX_STEPS = -1                  # Usar -1 para que se calcule automáticamente según épocas
SAVE_STEPS = 50                 # Guardar checkpoints más frecuentemente
LOGGING_STEPS = 10              # Logging más frecuente para mejor monitoreo
MAX_LENGTH = 1024  


# Configuración LoRA optimizada para Phi-2
LORA_R = 8                      # Rank reducido para ahorrar memoria
LORA_ALPHA = 16                 # Alpha
LORA_DROPOUT = 0.05             # Dropout para regularización
LORA_TARGET_MODULES = [         # Módulos específicos para Phi-2
    "q_proj",
    "k_proj", 
    "v_proj",
    "dense",
    "fc1",
    "fc2"
]

# Configuración de cuantización agresiva para ahorrar memoria
BITS = 8                        # Cuantización de 4-bit
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
OPTIM = "paged_adamw_8bit"      # Optimizador que ahorra memoria
LR_SCHEDULER = "cosine"         # Scheduler de tasa de aprendizaje


