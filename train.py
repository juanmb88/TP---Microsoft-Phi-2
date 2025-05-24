"""
Script principal para el fine-tuning de Phi-2 para conceptos matemáticos
Optimizado para GPUs con memoria limitada (4GB)
"""

import os
import sys
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from datamodule import get_tokenized_dataset, split_dataset, analyze_dataset_stats
from utils import (
    create_directories, 
    load_tokenizer, 
    print_trainable_parameters, 
    save_model_adapter,
    find_target_modules_for_phi2,
    optimize_model_for_gpu_with_4gb,
    print_gpu_utilization,
    clear_gpu_memory
)
import config









def main():
    # Optimizaciones para GPU con memoria limitada
    optimize_model_for_gpu_with_4gb()
    
    # Limpiar memoria
    clear_gpu_memory()
    
    # Imprimir estado inicial de GPU
    print("\n=== ESTADO INICIAL DE GPU ===")
    print_gpu_utilization()
    
    print("\n=== INICIANDO FINE-TUNING DE PHI-2 PARA MATEMÁTICAS ===\n")
    
    # Crear directorios necesarios
    create_directories()
    
    # Tiempo de inicio
    start_time = time.time()
    
    print(f"\n1. Preparando configuración para: {config.BASE_MODEL}")
    
    # Configuración de cuantización para ahorrar memoria
    print("\n2. Configurando cuantización de 4-bit...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config.DOUBLE_QUANT,
        bnb_4bit_quant_type="nf4",
    )
    
    # Cargar el tokenizer
    print("\n3. Cargando tokenizador...")
    tokenizer = load_tokenizer(config.BASE_MODEL)
    
    # Asegurarse de que el dataset está vacío para liberar memoria
    dataset = None
    clear_gpu_memory()
    
    # Cargar el modelo base con cuantización
    try:
        print("\n4. Cargando Phi-2 con cuantización de 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_flash_attention_2=False,  # Desactivar para compatibilidad
        )
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {str(e)}")
        # Intentar cargar con configuración más conservadora
        try:
            print("Intentando cargar con configuración alternativa...")
            model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
            print("✅ Modelo cargado con configuración alternativa")
        except Exception as e2:
            print(f"❌ Error fatal al cargar modelo: {str(e2)}")
            return
    
    # Imprimir uso de memoria después de cargar el modelo
    print("\n=== USO DE MEMORIA DESPUÉS DE CARGAR MODELO ===")
    print_gpu_utilization()
    
    # Preparar el modelo para entrenamiento de 4-bit
    print("\n5. Preparando modelo para entrenamiento...")
    model = prepare_model_for_kbit_training(model)
    
    # Configuración de LoRA adaptada para Phi-2
    print("\n6. Aplicando configuración LoRA...")
    
    # Si los módulos target no están definidos, encontrarlos automáticamente
    target_modules = config.LORA_TARGET_MODULES
    if not target_modules:
        target_modules = find_target_modules_for_phi2(model)
        print(f"Módulos target detectados automáticamente: {target_modules}")
    
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    
    # Obtener modelo con LoRA aplicado
    model = get_peft_model(model, lora_config)
    
    # Mostrar información de parámetros entrenables
    print_trainable_parameters(model)
    
    # Liberar memoria antes de cargar dataset
    clear_gpu_memory()
    
    # Preparar dataset
    print("\n7. Preparando dataset...")
    try:
        tokenized_dataset = get_tokenized_dataset(tokenizer, use_conversation_format=True)
        print(f"✅ Dataset tokenizado con {len(tokenized_dataset)} ejemplos")
        
        # Analizar estadísticas del dataset
        print("\n8. Analizando dataset...")
        dataset_stats = analyze_dataset_stats(tokenized_dataset)
        
        # Dividir en train/test
        train_dataset, eval_dataset = split_dataset(tokenized_dataset, test_size=0.1)
        
    except Exception as e:
        print(f"❌ Error al preparar el dataset: {str(e)}")
        print("Sugerencia: Verifica que el archivo dataset/matematicas.json existe y tiene formato correcto.")
        return
    
    # Configuración de entrenamiento para GPU con memoria limitada
    print("\n9. Configurando parámetros de entrenamiento...")
    
    training_args = TrainingArguments(
        output_dir=config.CHECKPOINT_PATH,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        # warmup_steps=config.WARMUP_STEPS,
        warmup_ratio=config.WARMUP_RATIO,
        # max_radio=config.WARMUP_RATIO,
        save_steps=config.SAVE_STEPS,
        logging_steps=25,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        optim=config.OPTIM,
        lr_scheduler_type=config.LR_SCHEDULER,
        weight_decay=0.01,
        fp16=config.USE_FP16,
        bf16=False,  # Phi-2 no soporta bien bf16
        gradient_checkpointing=True,  # Ahorrar memoria activando gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",          
        eval_steps=50,                        
        load_best_model_at_end=True,          
        metric_for_best_model="eval_loss",    
        greater_is_better=False, 
        seed=42,
        data_seed=42,
        logging_first_step=True,
    )
    
    # Crear data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Inicializar Trainer
    print("\n10. Inicializando entrenador...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer

    )
    
    # Calcular pasos totales
    total_steps = min(
        config.MAX_STEPS,
        len(train_dataset) * config.NUM_EPOCHS // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS)
    )
    
    # Iniciar entrenamiento
    print("\n11. ¡INICIANDO ENTRENAMIENTO!")
    print("=" * 60)
    print(f"Modelo: {config.BASE_MODEL}")
    print(f"GPU: {'Disponible' if torch.cuda.is_available() else 'No disponible'}")
    print(f"Épocas totales: {config.NUM_EPOCHS}")
    print(f"Tamaño de lote: {config.BATCH_SIZE} (x{config.GRADIENT_ACCUMULATION_STEPS} acumulación)")
    print(f"Tasa de aprendizaje: {config.LEARNING_RATE}")
    print(f"Pasos totales: {total_steps}")
    print("=" * 60)
    
    print("\nEl entrenamiento puede tardar varias horas dependiendo de tu hardware.")
    print("Puedes monitorear el progreso en la terminal.")
    print("Presiona Ctrl+C en cualquier momento para detener (se guardará el último checkpoint).")
    print("\nIniciando...")
    
    try:
        # Entrenar modelo
        train_result = trainer.train()
        
        # Calcular tiempo total de entrenamiento
        train_time = (time.time() - start_time) / 60  # en minutos
        print(f"\n✅ ¡Entrenamiento completado en {train_time:.2f} minutos!")
        
        # Guardar métricas
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Guardar estado del trainer
        trainer.save_state()
        
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
        print("Guardando estado actual...")
        trainer.save_state()
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        print("Intentando guardar el estado actual...")
        try:
            trainer.save_state()
        except:
            print("No se pudo guardar el estado")
        return
    
    # Guardar modelo y adaptador LoRA
    print("\n12. Guardando modelo final...")
    final_model_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    save_model_adapter(model, final_model_path)
    
    # Guardar también el tokenizer con la plantilla de chat
    try:
        tokenizer.save_pretrained(final_model_path)
        print(f"Tokenizador guardado en: {final_model_path}")
    except Exception as e:
        print(f"Error al guardar tokenizador: {str(e)}")
    
    print(f"\n✅ ¡Proceso completo! Modelo guardado en {final_model_path}")
    print("\nPuedes usar el modelo fine-tuned para conceptos matemáticos ejecutando:")
    print(f"  python evaluate.py")
    
    # Mostrar uso final de memoria
    print("\n=== USO FINAL DE MEMORIA ===")
    print_gpu_utilization()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error general: {str(e)}")
        import traceback
        traceback.print_exc()