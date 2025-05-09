# Análisis del Entrenamiento de Phi-2 para Tutoría Matemática

## Información General

| Característica | Valor |
|----------------|-------|
| **Modelo Base** | Microsoft Phi-2 (1.5B parámetros) |
| **Hardware** | NVIDIA GeForce GTX 1050 Ti (4GB VRAM, 768 núcleos CUDA) |
| **Épocas totales** | 250.0 |
| **Duración total** | 1 día, 8 horas, 33 minutos (1953.67 minutos) |
| **Loss final** | 0.2402 |

## Uso de Memoria GPU

| Tipo de Memoria | Tamaño (MB) |
|-----------------|-------------|
| Memoria asignada | 2326.09 MB |
| Memoria reservada | 3228.00 MB |
| Memoria máxima asignada | 3087.42 MB |

## Evolución del Entrenamiento

El proceso de entrenamiento evidenció tres fases claramente distinguibles:

### Fase 1: Aprendizaje Rápido (Épocas 1-50)
- **Pérdida inicial**: 2.6141
- **Pérdida al final de fase**: 0.1035
- **Mejora**: 96.04%
- Se observó un descenso acelerado de la pérdida
- La norma del gradiente mostró un pico significativo alrededor de la época 25 (valor 2.988)
- El learning rate alcanzó su máximo (~0.0001) cerca de la época 31

### Fase 2: Ajuste Fino (Épocas 51-100)
- **Pérdida inicial**: 0.1035
- **Pérdida al final de fase**: 0.0644
- **Mejora**: 37.78% (sobre el valor de pérdida de la Fase 1)
- La curva de pérdida empezó a aplanarse significativamente
- La norma del gradiente se estabilizó con valores menores, generalmente por debajo de 1.0
- El learning rate comenzó su descenso gradual

### Fase 3: Estabilización (Épocas 101-250)
- **Pérdida inicial**: 0.0644
- **Pérdida al final de fase**: 0.057
- **Mejora**: 11.49% (sobre el valor de pérdida de la Fase 2)
- La pérdida se mantuvo relativamente constante con mejoras marginales
- El learning rate disminuyó hasta valores prácticamente nulos
- La norma del gradiente se estabilizó en valores bajos, con pequeñas fluctuaciones

## Visualización de Métricas

![Evolución de la pérdida](https://github.com/user-attachments/assets/e5f4dd4a-d717-4dd6-98ed-06b7caab31a0)

*Figura 1: Evolución de la función de pérdida durante el entrenamiento. Nótese la rápida disminución inicial y la estabilización a partir de la época 100.*

![Evolución de métricas combinadas](https://github.com/user-attachments/assets/acf16f27-aa84-4f24-9abd-b9b0293c8ddf)

*Figura 2: Evolución combinada de la norma del gradiente y el learning rate. Obsérvese los picos iniciales del gradiente y la curva descendente del learning rate.*

## Análisis de Métricas Clave

### Loss (Función de Pérdida)
- **Comportamiento**: Disminución dramática en las primeras 50 épocas, seguida de mejoras moderadas hasta la época 100, y finalmente mejoras marginales en las épocas restantes.
- **Significado**: Indica que el modelo aprendió rápidamente los patrones principales del dataset, para luego refinar su conocimiento de manera más sutil.
- **Observación clave**: El 96.04% de la mejora total ocurrió en la primera fase (primeras 50 épocas).

### Grad Norm (Norma del Gradiente)
- **Comportamiento**: Mostró picos significativos durante la fase de aprendizaje rápido (especialmente en la época 25), para luego estabilizarse en valores más bajos.
- **Significado**: Los picos indican momentos de ajustes importantes en los pesos del modelo.
- **Observación clave**: La estabilización de la norma del gradiente a partir de la época 100 sugiere que el modelo dejó de realizar cambios significativos en sus parámetros.

### Learning Rate (Tasa de Aprendizaje)
- **Comportamiento**: Siguió un patrón de calentamiento inicial y decaimiento coseno, alcanzando su máximo alrededor de la época 31.
- **Significado**: Este esquema permitió un aprendizaje rápido al inicio, seguido de ajustes más finos a medida que el modelo convergía.
- **Observación clave**: La progresiva disminución del learning rate contribuyó a la estabilización del modelo en las últimas fases.

## Conclusiones y Recomendaciones

### Conclusiones
1. El entrenamiento fue exitoso, logrando una reducción total de pérdida del 97.82% (de 2.6141 a 0.057).
2. La mayor parte del aprendizaje significativo (96.04%) ocurrió durante las primeras 50 épocas.
3. La fase de ajuste fino (épocas 51-100) añadió mejoras moderadas (37.78% adicional).
4. La fase de estabilización (épocas 101-250) contribuyó con mejoras marginales (11.49%) a pesar de representar el 60% del tiempo total de entrenamiento.
5. El modelo Phi-2 demostró poder ser fine-tuned efectivamente con hardware de gama media (GTX 1050 Ti).

### Recomendaciones
1. **Optimización de épocas**: Para futuros entrenamientos de este tipo, considerar limitar a 100-150 épocas en lugar de 250, ya que la mejora después de la época 100 fue mínima.
2. **Ajuste de learning rate**: La configuración actual demostró ser efectiva, pero podría experimentarse con un decaimiento más rápido después de la época 50.
3. **Evaluación intermedia**: Implementar evaluaciones periódicas durante el entrenamiento para detectar más tempranamente el punto de estabilización.
4. **Uso de memoria**: El modelo utilizó aproximadamente el 77% de la memoria disponible de la GPU (3087.42 MB de 4GB), lo que sugiere que aún hay margen para optimizaciones o ajustes adicionales.
5. **Dataset**: Revisar la calidad del dataset para entrenamientos futuros, ya que la rápida convergencia inicial sugiere que podría beneficiarse de ejemplos más diversos o complejos.

---

*Modelo guardado en:* `models/phi2-matematicas-tutor`

*Fecha de generación del análisis: 9 de Mayo de 2025*
