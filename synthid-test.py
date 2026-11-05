import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig

model_name = "LiquidAI/LFM2.5-230M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
).eval()
print("model loaded")
messages = [
    {"role": "system", "content": "You are a text generator - you generate text based on user inputs."},
    {"role": "user", "content": "Hello, how do transformers generate text?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("where the fuck are we getting stuck")
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

synthid_keys = [42, 1337, 2026, 99, 7, 88, 12, 34, 56, 78, 90, 11, 22, 33, 44, 55, 66, 77, 123, 456] # List of 20-30 random integers

watermark_config = WatermarkingConfig()

print('generating response')
with torch.inference_mode():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        watermarking_config=watermark_config 
    )
print('response generated')
# Strip input prompt tokens from output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Generated response:")
print(response)



#FOR DETECTING

from transformers import WatermarkDetector


detector = WatermarkDetector(model_config = model.config,
                            device = model.device,
                            watermarking_config = watermark_config,
                            )

# 3. Check a piece of input text
# Returns a statistical p-value. A very small p-value (e.g., < 0.01) 
# strongly rejects the null hypothesis, meaning the text is likely watermarked.


output = detector(tokenizer([response], return_tensors="pt").input_ids, return_dict=True)
print(f"P-value of generated text: {output.p_value}, confidence: {output.confidence}, prediction: {output.prediction}")

output2 = detector(tokenizer(["I'm just a little guy please don't hurt me."], return_tensors="pt").input_ids, return_dict=True)
print(f"P-value of human text: {output2.p_value}, confidence: {output2.confidence}, prediction: {output2.prediction}")
