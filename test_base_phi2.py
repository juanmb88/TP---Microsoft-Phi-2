# simple_math_solver.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Cargando modelo base Phi-2...")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

def solve_math(question):
    # Usar el formato que sabemos que funciona
    prompt = f"Calcula {question} = "
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            do_sample=False  # Evitar warnings
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo la parte numérica de la respuesta
    response = full_response[len(prompt):].strip()
    
    # Limpiar texto adicional, quedarse solo con números
    import re
    numeric_match = re.search(r"\d+\.?\d*", response)
    if numeric_match:
        clean_answer = numeric_match.group(0)
    else:
        clean_answer = response
    
    return clean_answer

# Interfaz simple
while True:
    question = input("\nIngresa una operación matemática (o 'salir' para terminar): ")
    
    if question.lower() in ['salir', 'exit', 'quit']:
        break
    
    answer = solve_math(question)
    print(f"Respuesta: {answer}")