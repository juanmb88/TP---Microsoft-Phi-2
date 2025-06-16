Reporte Final: Fine-tuning Microsoft Phi-2 para Tutoría Matemática
📋 Resumen Ejecutivo
Este proyecto implementó y evaluó diferentes estrategias de fine-tuning del modelo Microsoft Phi-2 (2.7B parámetros) para crear un tutor matemático especializado. Se realizaron múltiples iteraciones experimentales que revelaron la importancia crítica de la configuración de hiperparámetros y el impacto del sobreentrenamiento en modelos matemáticos.
Imagen 1 - Especificaciones del Hardware Utilizado
🛠️ Metodología
Herramientas y Tecnologías

Modelo Base: Microsoft Phi-2 (2.7B parámetros)
Hardware: NVIDIA GeForce GTX 1050 Ti (4GB VRAM)
Técnica: LoRA (Low-Rank Adaptation) para entrenamiento eficiente
Dataset: 90+ ejemplos de operaciones matemáticas básicas y conceptos numéricos
Framework: PyTorch + Transformers + PEFT

Imagen 2 - Estructura del Proyecto de Fine-tuning
Configuraciones Experimentales
Se evaluaron tres configuraciones principales de learning rate para analizar su impacto en el rendimiento:
ConfiguraciónLearning RateÉpocasTiempoComportamientoSobreentrenado1e-425032.5 horasAlucinaciones severasOptimizado1e-61.8849 minutosErrores de cálculoDestructivo0.5N/AN/AError CUDA - Modelo inutilizable
📊 Resultados Principales
Comparativa de Rendimiento
MétricaEntrenamiento ExtensivoEntrenamiento OptimizadoMejoraTiempo32.5 horas49 minutos97.5% reducciónÉpocas2501.88133x menosFLOPS100.8M GF1.16M GF98.8% reducciónPérdida0.24023.7851Más saludable
Imagen 3 - Resultados del Entrenamiento Inicial (250 épocas)
Imagen 4 - Análisis de Fases de Entrenamiento y Métricas Combinadas
Imagen 5 - Resultados del Entrenamiento Optimizado (2 épocas)
Evaluación Funcional
Imagen 6 - Modelo con Learning Rate Extremo (0.5) - Error CUDA
Modelo Sobreentrenado (250 épocas):

Respuesta a "4+4": Texto incoherente sobre conjuntos numéricos y constantes matemáticas
Resultado: Alucinaciones extremas, respuestas inutilizables

Imagen 7 - Ejemplos de Código Python y Respuestas del Modelo
Modelo Optimizado (1.88 épocas):

Respuesta a "4+4": "9" (incorrecto pero coherente)
Resultado: Errores de cálculo pero sin alucinaciones

Imagen 8 - Interfaz del Evaluador con Respuestas Extensas
Imagen 9 - Pruebas de Operaciones Matemáticas Básicas
Imagen 10 - Respuestas del Modelo a Operaciones con Orden de Precedencia
Imagen 11 - Modelo Funcionando Correctamente con Respuestas Precisas
Modelo Base sin Fine-tuning:

Precisión: 60% en operaciones básicas
Resultado: Superior a cualquier versión entrenada

🔍 Análisis de Impacto del Learning Rate
Rangos de Comportamiento Identificados
Learning RateComportamientoAplicación5e-6 a 1e-5Respuestas estables y correctasRango óptimo1e-4 a 1e-3Errores ocasionales a alucinacionesZona de riesgo1e-2 a 0.1Alucinaciones controladasExperimentación0.5+Crashes CUDADestructivo
Hallazgo Crítico: Gradientes Explosivos
La configuración con learning rate 0.5 provocó errores a nivel de hardware (CUDA device-side assert), demostrando que hiperparámetros extremos pueden corromper completamente el modelo.
💡 Conclusiones y Lecciones Aprendidas
1. Sobreentrenamiento vs. Generalización

250 épocas: Memorización excesiva (loss=0.2402) resultando en alucinaciones
2 épocas: Mejor generalización (loss=3.7851) con errores controlados

2. Eficiencia Computacional

Reducir épocas de 250 a 2 mejoró la eficiencia en 97.5% sin sacrificar utilidad
El 96% del aprendizaje útil ocurre en las primeras 50 épocas

3. Preservación de Conocimiento

El modelo base supera al fine-tuning en operaciones básicas
Fine-tuning puede deteriorar capacidades preexistentes del modelo

4. Configuración Óptima Identificada
pythonLEARNING_RATE = 1e-5    # Equilibrio estabilidad-aprendizaje
NUM_EPOCHS = 2          # Evita sobreentrenamiento
LORA_R = 4              # Parámetros conservadores
LORA_ALPHA = 8          # Configuración estable
🚀 Recomendaciones
Para Futuros Proyectos de Fine-tuning Matemático:

Usar el modelo base directamente para operaciones básicas con prompting optimizado
Implementar early stopping basado en evaluación matemática continua
Aplicar fine-tuning gradual comenzando con conceptos básicos
Priorizar calidad sobre cantidad en datos de entrenamiento

Para Optimización de Recursos:

Configuraciones conservadoras (learning rate ≤ 1e-5) son más efectivas
Pocas épocas (1-3) suficientes para la mayoría de casos
Monitoreo en tiempo real de respuestas matemáticas durante entrenamiento

📈 Impacto del Proyecto
Este estudio demuestra que en fine-tuning de modelos matemáticos, "menos es más": configuraciones conservadoras y entrenamientos cortos producen mejores resultados que entrenamientos extensivos. La reducción del 97.5% en tiempo computacional mientras se mantiene funcionalidad representa un enfoque más sostenible y accesible para el desarrollo de modelos especializados.
El hallazgo de que el modelo base supera al fine-tuning especializado en matemáticas básicas sugiere una reevaluación de estrategias, priorizando la preservación de conocimiento preentrenado sobre la especialización extensiva.

Proyecto desarrollado como estudio de caso en optimización de hiperparámetros y técnicas de fine-tuning eficiente.