from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
SAVE_DIR = "./model"

print(f"Downloading {MODEL_ID} to {SAVE_DIR} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(SAVE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
)
model.save_pretrained(SAVE_DIR)

print(f"Done! Model saved to {os.path.abspath(SAVE_DIR)}")
