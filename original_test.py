import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading ORIGINAL base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

model.eval()

print("\n🔥 Original Qwen Chat Ready (type exit to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Bye 👋")
        break

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    response = response.split("assistant")[-1].strip()

    print("\nBot:", response, "\n")