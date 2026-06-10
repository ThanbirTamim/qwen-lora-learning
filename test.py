import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(base_model, "./adapter")

model.eval()

print("\n🔥 Chat ready (type exit to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"""<|im_start|>system
    You are a precise assistant. Always answer factual questions directly.<|im_end|>
    <|im_start|>user
    {user_input}<|im_end|>
    <|im_start|>assistant
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.0,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nBot:", result.split("assistant")[-1].strip(), "\n")