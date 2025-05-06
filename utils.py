"""
Funciones de utilidad para el proceso de fine-tuning con Phi-2
"""

import os
import torch
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb

def create_directories():
    """Crear las carpetas necesarias si no existen"""
    from config import MODEL_PATH, CHECKPOINT_PATH
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def clear_gpu_memory():
    """Liberar memoria GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_tokenizer(model_name):
    """Cargar el tokenizer del modelo base"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Configurar el tokenizer
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.unk_token
        
        # Phi-2 usa tokenizador CodeLlama que ya tiene tokens especiales
        special_tokens = {
            "user_token": "<|user|>",
            "assistant_token": "<|assistant|>",
            "system_token": "<|system|>",
            "end_token": "<|endoftext|>"
        }
        
        # Intentar añadir tokens especiales si no existen
        try:
            current_tokens = tokenizer.get_vocab()
            for token_name, token_value in special_tokens.items():
                if token_value not in current_tokens:
                    print(f"Añadiendo token especial: {token_value}")
                    tokenizer.add_special_tokens({token_name: token_value})
        except Exception as e:
            print(f"Advertencia al configurar tokens especiales: {e}")
        
        # Configurar la plantilla de chat si existe
        from config import CHAT_TEMPLATE
        if CHAT_TEMPLATE and hasattr(tokenizer, "chat_template"):
            tokenizer.chat_template = CHAT_TEMPLATE
            
        return tokenizer
    except Exception as e:
        print(f"Error al cargar tokenizador: {str(e)}")
        # Intentar cargar un tokenizador más compatible como fallback
        try:
            return AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
        except:
            return AutoTokenizer.from_pretrained("gpt2")

def print_gpu_utilization():
    """Mostrar uso de memoria GPU"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria asignada: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memoria reservada: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Memoria máxima asignada: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    else:
        print("No se detectó GPU disponible")

def print_trainable_parameters(model):
    """Imprime el número de parámetros entrenables vs total"""
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"Parámetros entrenables: {trainable_params:,d} ({100 * trainable_params / all_param:.2f}% del total)")
    print(f"Total de parámetros: {all_param:,d}")
    print(f"Tamaño del modelo: {all_param * 2 / 1024**2:.2f} MB (FP16) / {all_param * 4 / 1024**2:.2f} MB (FP32)")

def format_chat_to_instruction_response(messages):
    """Convertir un formato de chat a pares de instrucción-respuesta"""
    # Asumimos que los mensajes vienen en pares usuario-asistente
    if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        return {
            "instruction": messages[0]["content"],
            "response": messages[1]["content"]
        }
    return None

def format_conversation(messages, tokenizer=None):
    """Formatea una conversación para Phi-2"""
    formatted_text = ""
    
    for msg in messages:
        if msg["role"] == "user":
            formatted_text += f"<|user|>{msg['content']}"
        elif msg["role"] == "assistant":
            formatted_text += f"<|assistant|>{msg['content']}"
        elif msg["role"] == "system":
            formatted_text += f"<|system|>{msg['content']}"
    
    # Añadir token de fin si es necesario
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        if not formatted_text.endswith(tokenizer.eos_token):
            formatted_text += tokenizer.eos_token
    
    return formatted_text

def load_json_dataset(file_path):
    """Cargar un dataset en formato JSON de chat matemático"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Dataset cargado: {len(data)} ejemplos")
        return data
    except FileNotFoundError:
        print(f"ADVERTENCIA: Archivo {file_path} no encontrado.")
        # Crear estructura mínima para pruebas
        return []
    except Exception as e:
        print(f"Error al cargar el dataset: {str(e)}")
        return []

def save_model_adapter(model, output_dir):
    """Guardar el adaptador LoRA del modelo"""
    try:
        model.save_pretrained(output_dir)
        print(f"Adaptador LoRA guardado en: {output_dir}")
        return True
    except Exception as e:
        print(f"Error al guardar el adaptador: {str(e)}")
        return False

def find_target_modules_for_phi2(model):
    """Encontrar módulos específicos para LoRA en Phi-2"""
    target_modules = set()
    for name, module in model.named_modules():
        if name.endswith("q_proj") or name.endswith("k_proj") or name.endswith("v_proj") or \
           name.endswith("dense") or name.endswith("fc1") or name.endswith("fc2"):
            module_name = name.split(".")[-1]
            target_modules.add(module_name)
    
    return list(target_modules)

def optimize_model_for_gpu_with_4gb():
    """Optimizaciones específicas para GPU con 4GB"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False  # Desactivar TF32 para ahorrar memoria
        torch.backends.cudnn.allow_tf32 = False
        
        # Configurar cuDNN para ahorrar memoria
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Limitar cache de CUDA
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"