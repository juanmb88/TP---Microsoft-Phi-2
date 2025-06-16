"""
Script para evaluar el modelo Phi-2 entrenado en conceptos matemáticos
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich import print as rprint
import time
from utils import clear_gpu_memory, optimize_model_for_gpu_with_4gb
import config

# Configurar consola para mejor visualización
console = Console()

def load_model(model_path=None):
    """Cargar el modelo base con el adaptador LoRA"""
    # Aplicar optimizaciones para GPU con memoria limitada
    optimize_model_for_gpu_with_4gb()
    
    # Limpiar memoria antes de cargar
    clear_gpu_memory()
    
    if model_path is None:
        model_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    
    if not os.path.exists(model_path):
        console.print(f"[bold red]Error:[/bold red] No se encuentra el modelo en {model_path}")
        console.print("[yellow]¿Has completado el entrenamiento con train.py?[/yellow]")
        return None, None
    
    console.print(f"[bold green]Cargando modelo desde:[/bold green] {model_path}")
    
    try:
        # Configurar cuantización para inferencia
        quantization_config = None
        if torch.cuda.is_available():
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
            console.print("[cyan]Usando cuantización de 4-bit para inferencia...[/cyan]")
        
        # Verificar si existe configuración PEFT
        is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        # Definir el dispositivo
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        device_type = "GPU + CPU" if torch.cuda.is_available() else "Solo CPU"
        
        console.print(f"[bold cyan]Dispositivo de inferencia:[/bold cyan] {device_type}")
        
        if is_peft_model:
            # Cargar como modelo PEFT (LoRA)
            console.print("[yellow]Detectado modelo con adaptadores LoRA.[/yellow]")
            
            # Cargar configuración PEFT
            peft_config = PeftConfig.from_pretrained(model_path)
            
            # Cargar modelo base
            console.print(f"[cyan]Cargando modelo base: {peft_config.base_model_name_or_path}[/cyan]")
            
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                **(quantization_config or {}),
            )
            
            # Cargar adaptadores
            console.print("[cyan]Aplicando adaptadores LoRA...[/cyan]")
            model = PeftModel.from_pretrained(base_model, model_path)
            
        else:
            # Cargar modelo normal
            console.print("[yellow]Cargando modelo completo...[/yellow]")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                **(quantization_config or {}),
            )
        
        # Cargar tokenizador
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            # Fallback: usar tokenizador de Phi-2 si no se encuentra en el modelo
            console.print("[yellow]Usando tokenizador de Phi-2 como alternativa...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        # Asegurar que hay token de padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        console.print("[bold green]Modelo cargado correctamente[/bold green]")
        return model, tokenizer
    
    except Exception as e:
        console.print(f"[bold red]Error al cargar el modelo:[/bold red] {str(e)}")
        return None, None

def generate_response(model, tokenizer, user_message, conversation_history=None, system_prompt=None):
    """Generar respuesta para una pregunta matemática - VERSIÓN CORREGIDA"""
    if model is None or tokenizer is None:
        return "Error: Modelo no cargado correctamente", conversation_history
    
    if conversation_history is None:
        conversation_history = []
        
        # Agregar system prompt si existe
        if system_prompt:
            conversation_history.append({"role": "system", "content": system_prompt})
    
    # Añadir el mensaje del usuario a la conversación
    conversation_history.append({"role": "user", "content": user_message})
    
    try:
        # 🔧 CORRECCIÓN 1: Formato limpio sin tokens especiales problemáticos
        input_text = ""
        
        # Solo procesar los últimos 2-3 intercambios para evitar degradación
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        
        for msg in recent_history:
            if msg["role"] == "system":
                input_text += f"Sistema: {msg['content']}\n"
            elif msg["role"] == "user":
                input_text += f"Pregunta: {msg['content']}\n"
            elif msg["role"] == "assistant":
                input_text += f"Respuesta: {msg['content']}\n"
        
        # 🔧 CORRECCIÓN 2: Formato final limpio
        # Si es la primera pregunta o no hay assistant previo
        if not any(msg["role"] == "assistant" for msg in recent_history):
            input_text += "Respuesta: "
        else:
            # Para preguntas siguientes, usar formato consistente
            input_text = input_text.rstrip() + "\nRespuesta: "
        
        # 🔧 CORRECCIÓN 3: Tokenizar con configuración segura
        inputs = tokenizer(
            input_text, 
            return_tensors="pt",
            max_length=128,          # Limitar contexto
            truncation=True,
            padding=False
        ).to(model.device)
        
        # 🔧 CORRECCIÓN 4: Configuración de generación optimizada
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,                    # Aumentar ligeramente para respuestas completas
                min_new_tokens=1,                     # Asegurar al menos 1 token
                temperature=0.3,                      # Muy determinista para matemáticas
                do_sample=True,
                top_p=0.9,                          # Más restrictivo
                top_k=50,                            # Limitar vocabulario
                repetition_penalty=1.1,              # Suave para evitar degradación
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2               # Evitar repeticiones
            )
        
        # 🔧 CORRECCIÓN 5: Extracción limpia de respuesta
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la nueva respuesta
        if "Respuesta: " in generated_text:
            # Dividir por "Respuesta: " y tomar la última parte
            parts = generated_text.split("Respuesta: ")
            response = parts[-1].strip()
        else:
            # Fallback: tomar todo después del input original
            response = generated_text[len(input_text):].strip()
        
        # 🔧 CORRECCIÓN 6: Limpiar respuesta de tokens residuales
        response = clean_response(response)
        
        # Validar que la respuesta no esté vacía
        if not response or len(response.strip()) == 0:
            response = "No pude generar una respuesta válida."
        
        # Añadir la respuesta a la conversación
        conversation_history.append({"role": "assistant", "content": response})
        
        return response, conversation_history
    
    except Exception as e:
        error_msg = f"Error al generar respuesta: {str(e)}"
        print(f"Error: {error_msg}")
        print(f"Input text: {input_text[:200]}...")
        conversation_history.append({"role": "assistant", "content": "Error en la generación"})
        return "Error en la generación", conversation_history
    

    

def clean_response(response):
    """Limpiar respuesta de tokens problemáticos"""
    import re
    
    # 🧹 LIMPIEZA 1: Remover tokens especiales problemáticos
    problematic_tokens = [
        r'<\|user\|>.*',           # Todo después de <|user|>
        r'<\|assistant\|>.*',      # Todo después de <|assistant|>
        r'<\|system\|>.*',         # Todo después de <|system|>
        r'<\|endofgeneration\|>.*', # Todo después de endofgeneration
        r'<\|.*\|>',               # Cualquier token especial
        r'""".*',                  # Comillas triples problemáticas
    ]
    
    for pattern in problematic_tokens:
        response = re.sub(pattern, '', response, flags=re.DOTALL)
    
    # 🧹 LIMPIEZA 2: Remover patrones específicos observados
    response = re.sub(r'\d+\s*%\s*\d+\?\s*<\|.*', '', response)  # "54 % 37?<|ass"
    response = re.sub(r'<\|.*', '', response)                     # Cualquier token incompleto
    
    # 🧹 LIMPIEZA 3: Limpiar espacios y caracteres raros
    response = response.strip()
    response = re.sub(r'\s+', ' ', response)  # Múltiples espacios a uno
    response = re.sub(r'^\W+', '', response)  # Caracteres raros al inicio
    
    # 🧹 LIMPIEZA 4: Para respuestas matemáticas, mantener solo lo esencial
    if re.match(r'^\d+\.?\d*\s*$', response.strip()):  # Si es solo un número
        response = response.strip()
    
    return response


def display_welcome():
    """Mostrar mensaje de bienvenida"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]EVALUADOR DE MODELO MATEMÁTICO PHI-2[/bold cyan]\n\n"
        "Este programa te permite interactuar con el modelo matemático Phi-2 fine-tuned.\n"
        "Ingresa tus preguntas sobre conceptos matemáticos y obtén respuestas generadas por el modelo.\n\n"
        "[yellow]Escribe 'salir' para terminar, 'nueva' para iniciar una nueva conversación,[/yellow]\n"
        "[yellow]o 'guardar' para guardar la conversación actual.[/yellow]",
        title="Bienvenido/a",
        border_style="green"
    ))

def interactive_mode(model, tokenizer):
    """Modo interactivo para evaluar el modelo"""
    display_welcome()
    
    if model is None or tokenizer is None:
        return
    
    # System prompt para matemáticas
    system_prompt = """Eres un asistente matemático especializado. Tu objetivo es explicar conceptos 
    matemáticos de manera clara, precisa y didáctica. Ayudas a los estudiantes a entender
    temas complejos a través de explicaciones concisas y ejemplos ilustrativos. Tu conocimiento
    abarca desde aritmética básica hasta cálculo, álgebra, geometría, probabilidad y estadística."""
    
    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})
    
    conversation_count = 1
    
    while True:
        console.print("\n[bold green]Tu pregunta:[/bold green] (escribe 'salir' para terminar, 'nueva' para reiniciar, 'guardar' para guardar)", style="green")
        user_input = input("> ").strip()
        
        if user_input.lower() == "salir":
            console.print("[bold cyan]¡Gracias por utilizar el evaluador de modelo matemático![/bold cyan]")
            break
        
        elif user_input.lower() == "nueva":
            conversation_history = []
            if system_prompt:
                conversation_history.append({"role": "system", "content": system_prompt})
            conversation_count += 1
            console.print("[bold cyan]Nueva conversación iniciada[/bold cyan]")
            continue
            
        elif user_input.lower() == "guardar":
            filename = f"conversacion_matematica_{conversation_count}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, ensure_ascii=False, indent=2)
            console.print(f"[bold green]Conversación guardada en:[/bold green] {filename}")
            continue
            
        elif not user_input:
            continue
        
        # Generar respuesta
        console.print("[bold cyan]Generando respuesta...[/bold cyan]")
        response, conversation_history = generate_response(
            model, tokenizer, user_input, conversation_history
        )
        
        # Mostrar respuesta con formato
        console.print("\n[bold green]Respuesta:[/bold green]")
        console.print(Panel(Markdown(response), border_style="cyan"))

def batch_evaluate(model, tokenizer, test_file=None):
    """Evaluación por lotes usando un archivo de preguntas de prueba"""
    if test_file is None:
        test_file = "test_questions.json"
    
    if not os.path.exists(test_file):
        console.print(f"[bold yellow]Archivo de pruebas {test_file} no encontrado.[/bold yellow]")
        console.print("Creando archivo de ejemplo...")
        
        # Crear archivo de ejemplo
        test_questions = [
            "¿Qué son los números naturales?",
            "Explica el concepto de derivada",
            "¿Cómo se resuelve una ecuación cuadrática?",
            "¿Qué es el teorema de Pitágoras?",
            "Explica qué es una integral definida"
        ]
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_questions, f, ensure_ascii=False, indent=2)
        
        console.print(f"[bold green]Archivo de ejemplo creado:[/bold green] {test_file}")
    
    # Cargar preguntas de prueba
    with open(test_file, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    
    console.print(f"[bold cyan]Evaluando modelo con {len(test_questions)} preguntas...[/bold cyan]")
    
    # Archivo para guardar resultados
    results_file = f"evaluacion_resultados_{time.strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("# Resultados de Evaluación del Modelo Matemático Phi-2\n\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"[cyan]Pregunta {i}/{len(test_questions)}:[/cyan] {question}")
            
            # Generar respuesta
            response, _ = generate_response(model, tokenizer, question)
            
            # Guardar en archivo
            f.write(f"## Pregunta {i}: {question}\n\n")
            f.write(f"### Respuesta:\n\n{response}\n\n")
            f.write("---\n\n")
            
            # Mostrar respuesta resumida
            short_response = response[:100] + "..." if len(response) > 100 else response
            console.print(f"[green]Respuesta:[/green] {short_response}\n")
    
    console.print(f"[bold green]Evaluación completada. Resultados guardados en:[/bold green] {results_file}")





def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Evaluador de modelo matemático Phi-2")
    parser.add_argument("--batch", action="store_true", help="Ejecutar evaluación por lotes")
    parser.add_argument("--test-file", type=str, help="Archivo JSON con preguntas de prueba")
    parser.add_argument("--model-path", type=str, help="Ruta al modelo entrenado")
    args = parser.parse_args()
    
    # Cargar modelo
    model, tokenizer = load_model(args.model_path)
    
    if model is None or tokenizer is None:
        console.print("[bold red]No se pudo cargar el modelo. Finalizando programa.[/bold red]")
        return
    
    # Modo de evaluación
    if args.batch:
        batch_evaluate(model, tokenizer, args.test_file)
    else:
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Programa interrumpido por el usuario[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error inesperado:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()