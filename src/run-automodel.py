import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("lmsys/longchat-7b-16k", load_in_8bit=True, 
)
#model.cuda()

tokenizer = AutoTokenizer.from_pretrained("lmsys/longchat-7b-16k")

with open("x0.txt", "r") as f:
    text1 = f.read()

tokenized1 = tokenizer(text1, return_tensors="pt")
input_ids1 = tokenized1.input_ids.cuda()

gen_tokens = model.generate(
    input_ids1,
    use_cache=True,
    max_new_tokens = 1,
    return_dict_in_generate=True
)

torch.cuda.synchronize()


gen_text = tokenizer.decode(gen_tokens['sequences'][0], skip_special_tokens = True)
print("generated_text: ", gen_text)
kv = gen_tokens['past_key_values']


text2 = "What was the first topic we discussed? ASSISTANT:"
tokenized2 = tokenizer(text2, return_tensors="pt")
input_ids2 = tokenized2.input_ids.cuda()

attn = torch.cat((tokenized1["attention_mask"], tokenized2["attention_mask"]), -1)
generated2 = model.generate(input_ids2, past_key_values = kv, max_new_tokens = 100,  return_dict_in_generate = True, attention_mask = attn)
gen_text = tokenizer.decode(generated2['sequences'][0], skip_special_tokens = True)

#gen_text2 = tokenizer.decode(generated2[0])


print("generated_text: ", gen_text)
#print("generated_text2: ", gen_text2)
