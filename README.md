## Este es el resultado del entrenamiento sobre una placa grafica  GTX 1050 Ti (4GB VRAM, 768 núcleos CUDA)
#  Modelo: microsoft/phi-2 (1.5B parámetros)
#  === USO FINAL DE MEMORIA ===
#  GPU: NVIDIA GeForce GTX 1050 Ti
#  Memoria asignada: 2326.09 MB
#  Memoria reservada: 3228.00 MB
#  Memoria máxima asignada: 3087.42 MB

✅ ¡Entrenamiento completado!
***** train metrics *****
  epoch                    =             250.0
  total_flos               =       100852232GF
  train_loss               =            0.2402
  train_time               =   1 day, 8:33:25.44
  train_samples_per_second =             0.137
  train_steps_per_second   =             0.009

Guardando modelo final...
Adaptador LoRA guardado en: models/phi2-matematicas-tutor
Tokenizador guardado en: models/phi2-matematicas-tutor

✅ ¡Proceso completo! Modelo guardado en models/phi2-matematicas-tutor

   loss: la pérdida (error del modelo), que disminuye de manera constante, lo cual indica que el modelo está aprendiendo.

   grad_norm: la norma del gradiente, útil para diagnosticar problemas como el desvanecimiento o explosión del gradiente.

   learning_rate: la tasa de aprendizaje, que parece estar controlada por un planificador de tipo cosine decay o similar, disminuyendo progresivamente.


![fd2822cc-a0b6-4e5d-bea7-091da17a4f68](https://github.com/user-attachments/assets/e5f4dd4a-d717-4dd6-98ed-06b7caab31a0)


la gráfica de la evolución de la pérdida durante el entrenamiento. Como se puede ver, la pérdida disminuye rápidamente al principio y luego se estabiliza a partir de la época ~100, lo cual sugiere que el modelo ya aprendió lo esencial y podría no beneficiarse mucho de continuar entrenando más allá de ese punto.


![d59504b5-700b-4c11-a58f-90816fc4383e](https://github.com/user-attachments/assets/acf16f27-aa84-4f24-9abd-b9b0293c8ddf)


Norma del gradiente: muestra picos iniciales más altos (especialmente en la época 25), pero se estabiliza a valores bajos después de la época 100, lo que indica que el modelo ya no está realizando grandes actualizaciones.

Tasa de aprendizaje: desciende gradualmente durante el entrenamiento, lo cual es típico en estrategias para permitir una convergencia más estable al final.
