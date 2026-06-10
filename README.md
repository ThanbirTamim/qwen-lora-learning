# 📘 Qwen2.5 LoRA Fine-Tuning (Hands-on Learning Project)

This project demonstrates how to fine-tune a large language model using **LoRA (Low-Rank Adaptation)** on a local machine (MacBook), using the Qwen2.5 Instruct model.

It is designed for **learning LoRA step-by-step**, not production scale training.

---

# 🚀 What this project does

- Loads a pre-trained LLM (Qwen2.5-3B-Instruct)
- Applies LoRA adapters to attention layers
- Trains on a small instruction dataset
- Saves only lightweight adapter weights
- Runs interactive chat inference

---

# 🧠 Key Concepts Covered

- LoRA (Low-Rank Adaptation)
- PEFT (Parameter Efficient Fine-Tuning)
- Instruction tuning
- Chat format prompting (Qwen style)
- Adapter-based model updates
- Local LLM training on Mac (MPS)

---

# 📁 Project Structure

```
qwen-lora-learning/
├── train.py
├── test.py
├── requirements.txt
└── data/
    └── train.json
```
---

# 📊 Dataset Format

```
[
  {
    "instruction": "What is LoRA?",
    "response": "LoRA trains small adapter weights instead of the entire model."
  }
]
```

---

# ⚙️ Installation

## 1. Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
## 2. Install dependencies
```
pip install -r requirements.txt
```
---

# 🏋️ Training

Run LoRA fine-tuning:

```python train.py```

### What happens during training:

- Loads Qwen2.5-3B model
- Injects LoRA into attention layers:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
- Converts dataset into Qwen chat format
- Trains only LoRA parameters
- Saves adapter to ./adapter/

---

# 💾 Output After Training

```
adapter/
├── adapter_config.json
├── adapter_model.safetensors
```

👉 This is your trained LoRA module (very small size)

---

# 💬 Run Chat Inference

```python test.py```

Example:

You: Who is Thanbir Tamim?
Bot: Thanbir Tamim is a senior software engineer at Softzino Technologies.

---

# 🧠 How It Works

```
Dataset
   ↓
Chat Format Conversion
   ↓
Tokenization
   ↓
Frozen Qwen Base Model
   ↓
LoRA Layers Injected
   ↓
Training Only LoRA Weights
   ↓
Save Adapter
   ↓
Inference = Base Model + LoRA Adapter
```

---

# 🔥 Why LoRA is Powerful

- Full model is NOT trained
- Only small adapter weights are updated
- Very low GPU/RAM usage
- Easy to switch adapters
- Fast experimentation

---

# ⚠️ Important Notes

- Dataset is very small → model may not generalize well
- LoRA is not a database (it learns patterns, not facts)
- Better results require:
  - More data (50–500+ examples)
  - Repeated instruction patterns
  - Proper chat formatting

---

# 🧪 Recommended Improvements

- Increase dataset size
- Use Qwen chat template consistently
- Tune LoRA rank (r=8 → r=16)
- Try Qwen2.5-0.5B for faster experiments
- Experiment with QLoRA (4-bit training)

---

# 👨‍💻 Learning Outcome

After this project you understand:

- How LoRA modifies LLM behavior
- How adapters work
- How instruction tuning works
- How inference = Base Model + Adapter

---

# 🚀 Next Steps

- Merge LoRA into base model
- Convert model to Ollama format
- Build FastAPI inference server
- Add multi-turn memory chat
- Build RAG + LoRA hybrid system
