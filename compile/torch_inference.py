from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaConfig

model_path = "llama-2-13b-chat-hf"

model = LlamaForCausalLM.from_pretrained(model_path)
model.eval()

tokenizer = LlamaTokenizer.from_pretrained(model_path)

input_ids = tokenizer.encode("hello", return_tensors="pt")
output = model.generate(input_ids, num_beams=1, do_sample=False, max_length=100)
print(tokenizer.batch_decode(output, skip_special_tokens=False))
