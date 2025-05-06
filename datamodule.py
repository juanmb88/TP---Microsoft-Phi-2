"""
Módulo para cargar y preparar los datos de entrenamiento para el tutor matemático con Phi-2
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
import numpy as np
from utils import (
    format_chat_to_instruction_response, 
    format_conversation, 
    load_json_dataset, 
    load_tokenizer,
    clear_gpu_memory
)
from config import MAX_LENGTH, DATASET_PATH

def prepare_dataset():
    """Preparar y cargar el dataset de conceptos matemáticos"""
    try:
        # Cargar desde un archivo JSON
        data = load_json_dataset(DATASET_PATH)
        
        if not data:
            raise ValueError("Dataset vacío o no disponible")
        
        # Convertir los datos al formato necesario
        formatted_data = []
        
        for item in data:
            if "messages" in item:
                # Extraer la instrucción (pregunta) y respuesta
                instruction_response = format_chat_to_instruction_response(item["messages"])
                if instruction_response:
                    formatted_data.append(instruction_response)
        
        # Verificar que tenemos datos
        if not formatted_data:
            raise ValueError("No se pudieron extraer pares de instrucción-respuesta válidos del dataset")
        
        # Convertir a Dataset de HuggingFace
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        print(f"Dataset de matemáticas cargado con {len(dataset)} ejemplos")
        return dataset
    
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return None

def prepare_conversations_dataset():
    """Preparar el dataset manteniendo el formato de conversación"""
    try:
        # Cargar desde un archivo JSON
        data = load_json_dataset(DATASET_PATH)
        
        if not data:
            raise ValueError("Dataset vacío o no disponible")
        
        # Mantener el formato de conversación completo
        formatted_data = []
        
        for item in data:
            if "messages" in item:
                formatted_data.append({"messages": item["messages"]})
        
        # Verificar que tenemos datos
        if not formatted_data:
            raise ValueError("No se pudieron extraer conversaciones válidas del dataset")
        
        # Convertir a Dataset de HuggingFace
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        print(f"Dataset de conversaciones matemáticas cargado con {len(dataset)} ejemplos")
        return dataset
    
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return None

def preprocess_function(examples, tokenizer):
    """Preprocesar y tokenizar los ejemplos de instrucción-respuesta para Phi-2"""
    # Formatear los ejemplos para Phi-2
    formatted_texts = []
    
    for instruction, response in zip(examples['instruction'], examples['response']):
        # Crear una conversación en formato de lista de mensajes
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        
        # Formatear la conversación específicamente para Phi-2
        formatted_text = format_conversation(messages, tokenizer)
        formatted_texts.append(formatted_text)
    
    # Tokenizar los textos con manejo de memoria
    try:
        # Tokenizar en lotes pequeños para evitar problemas de memoria
        batch_size = 4
        all_input_ids = []
        all_attention_mask = []
        
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i+batch_size]
            
            batch_tokenized = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            all_input_ids.extend(batch_tokenized["input_ids"].tolist())
            all_attention_mask.extend(batch_tokenized["attention_mask"].tolist())
            
            # Liberar memoria después de cada lote
            clear_gpu_memory()
        
        tokenized = {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(all_attention_mask)
        }
        
        # Preparar las etiquetas (igual que los input_ids para entrenamiento de lenguaje)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    except Exception as e:
        print(f"Error durante tokenización: {str(e)}")
        raise e

def preprocess_conversations(examples, tokenizer):
    """Preprocesar y tokenizar las conversaciones completas para Phi-2"""
    try:
        formatted_texts = []
        
        for messages in examples['messages']:
            formatted_text = format_conversation(messages, tokenizer)
            formatted_texts.append(formatted_text)
        
        # Tokenizar los textos con manejo de memoria
        batch_size = 4
        all_input_ids = []
        all_attention_mask = []
        
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i+batch_size]
            
            batch_tokenized = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            all_input_ids.extend(batch_tokenized["input_ids"].tolist())
            all_attention_mask.extend(batch_tokenized["attention_mask"].tolist())
            
            # Liberar memoria después de cada lote
            clear_gpu_memory()
        
        tokenized = {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(all_attention_mask)
        }
        
        # Preparar las etiquetas (igual que los input_ids para entrenamiento de lenguaje)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    except Exception as e:
        print(f"Error durante tokenización de conversaciones: {str(e)}")
        raise e

def get_tokenized_dataset(tokenizer, use_conversation_format=True):
    """Obtener el dataset tokenizado"""
    if use_conversation_format:
        dataset = prepare_conversations_dataset()
        if dataset is None:
            raise ValueError("No se pudo cargar el dataset de conversaciones")
        
        # Procesar el dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_conversations(examples, tokenizer),
            batched=True,
            remove_columns=['messages'],
            desc="Tokenizando dataset de conversaciones"
        )
    else:
        dataset = prepare_dataset()
        if dataset is None:
            raise ValueError("No se pudo cargar el dataset")
        
        # Procesar el dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=['instruction', 'response'],
            desc="Tokenizando dataset"
        )
    
    return tokenized_dataset

def split_dataset(dataset, test_size=0.1, seed=42):
    """Dividir el dataset en conjuntos de entrenamiento y evaluación"""
    try:
        # Calcular tamaños
        dataset_size = len(dataset)
        test_samples = max(1, int(dataset_size * test_size))
        train_samples = dataset_size - test_samples
        
        # Dividir el dataset
        splits = dataset.train_test_split(
            test_size=test_samples,
            train_size=train_samples,
            seed=seed
        )
        
        train_dataset = splits["train"]
        test_dataset = splits["test"]
        
        print(f"Dataset dividido: {train_samples} ejemplos de entrenamiento, {test_samples} ejemplos de prueba")
        
        return train_dataset, test_dataset
    except Exception as e:
        print(f"Error al dividir dataset: {str(e)}")
        # En caso de error, retornar el dataset completo como entrenamiento y una pequeña muestra como validación
        small_val = dataset.select(range(min(3, len(dataset))))
        return dataset, small_val

def analyze_dataset_stats(dataset):
    """Analizar estadísticas básicas del dataset de matemáticas"""
    try:
        print("\n--- Estadísticas del Dataset ---")
        
        # Analizar según el formato del dataset
        if "input_ids" in dataset.column_names:
            # Dataset ya tokenizado
            avg_length = np.mean([len(ids) for ids in dataset["input_ids"]])
            max_length = max([len(ids) for ids in dataset["input_ids"]])
            
            print(f"Total de ejemplos: {len(dataset)}")
            print(f"Longitud promedio de secuencias: {avg_length:.1f} tokens")
            print(f"Secuencia más larga: {max_length} tokens")
            
            if max_length > MAX_LENGTH:
                print(f"⚠️ ADVERTENCIA: La secuencia más larga ({max_length}) excede MAX_LENGTH ({MAX_LENGTH})")
        
        elif "instruction" in dataset.column_names and "response" in dataset.column_names:
            # Dataset de instrucción-respuesta
            instruction_lengths = [len(text.split()) for text in dataset["instruction"]]
            response_lengths = [len(text.split()) for text in dataset["response"]]
            
            avg_instruction_len = sum(instruction_lengths) / len(instruction_lengths)
            avg_response_len = sum(response_lengths) / len(response_lengths)
            max_instruction_len = max(instruction_lengths)
            max_response_len = max(response_lengths)
            
            print(f"Total de ejemplos: {len(dataset)}")
            print(f"Longitud promedio de instrucciones: {avg_instruction_len:.1f} palabras")
            print(f"Longitud promedio de respuestas: {avg_response_len:.1f} palabras")
            print(f"Instrucción más larga: {max_instruction_len} palabras")
            print(f"Respuesta más larga: {max_response_len} palabras")
            
        elif "messages" in dataset.column_names:
            # Dataset en formato de conversación
            message_counts = [len(conv) for conv in dataset["messages"]]
            avg_messages = sum(message_counts) / len(message_counts)
            
            print(f"Total de conversaciones: {len(dataset)}")
            print(f"Promedio de mensajes por conversación: {avg_messages:.1f}")
            print(f"Conversación más larga: {max(message_counts)} mensajes")
        
        return {
            "num_examples": len(dataset)
        }
    
    except Exception as e:
        print(f"Error al analizar estadísticas: {str(e)}")
        return {}