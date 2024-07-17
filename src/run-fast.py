import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import load_model



def define_model_and_tokenizer(model_id, num_gpus=1, max_gpu_memory=48):
    """ Define the model and tokenizer
    """
    model, tokenizer = load_model(
            model_id,
            device="cuda",
            num_gpus=num_gpus,
            max_gpu_memory=f"{max_gpu_memory}GiB",
            load_8bit=True,
            cpu_offloading=False,
            debug=False,
        )

    return model, tokenizer



model_id = "lmsys/longchat-7b-16k"

model, tokenizer = define_model_and_tokenizer(model_id)
model.cuda()

with open("3.txt", "r") as f:
    text1 = f.read()

tokenized1 = tokenizer(text1, return_tensors="pt")
input_ids = tokenized1.input_ids.cuda()
st = time.monotonic()

generated = model.generate(input_ids, max_new_tokens = 1, use_cache = True, return_dict_in_generate=True)

print("output of first call: ", tokenizer.decode(generated['sequences'][0], skip_special_tokens = True))

torch.cuda.synchronize()

kv = generated['past_key_values']

text2 = "The first topic we discussed was "
tokenized2 = tokenizer(text2, return_tensors="pt")
input_ids2 = tokenized2.input_ids.cuda()

attn = torch.cat((tokenized1["attention_mask"], tokenized2["attention_mask"]), -1).cuda()

generated2 = model.generate(input_ids2, past_key_values = kv, max_new_tokens = 50,  return_dict_in_generate = True, attention_mask = attn)
print("generated2[0]: ", generated2[0])

gen_text = tokenizer.decode(generated2['sequences'][0], skip_special_tokens = True)
print("generated_text: ", gen_text)
