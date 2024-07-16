from GEARLM import SimulatedGearLlamaForCausalLM
from GEARLM import CompressionConfig
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
import torch
import time
import os
import argparse
import pickle
import json

cache_dir = "../../../../cache"

parser = argparse.ArgumentParser(description="Evaluate GEAR with KV Cache")
parser.add_argument(
    "--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model name or path."
)
parser.add_argument(
    "--prompt_file", type=str, default="0.txt", help=""
)

parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--top_k", type=int, default=50, help="")
parser.add_argument("--compress_method", type=str, default="GEAR", help="")
parser.add_argument("--rank", type=float, default=0.0, help="")
parser.add_argument("--rankv", type=float, default=0.0, help="")
parser.add_argument("--prefillrank", type=float, default=0.0, help="rank compared with smaller dimension set to K cache.")
parser.add_argument("--prefillrankv", type=float, default=0.0, help="rank compared with smaller dimension set to V cache.")
parser.add_argument("--loop", type=int, default=0, help="")
parser.add_argument("--quantize_bit", type=int, default=8, help="")
parser.add_argument("--group_num", type=int, default=0, help="")
parser.add_argument("--group_size", type=int, default=0, help="")
parser.add_argument("--left", type=float, default=0.0, help="")
parser.add_argument("--attention_number", type=int, default=100, help="")
parser.add_argument("--streaming", action="store_true", default=False, help="")
parser.add_argument("--streaming_gap", type=int, default=0, help="")
parser.add_argument("--stream_grouping", action="store_true", default=False, help="Use streaming mode.")
parser.add_argument("--start_file", type=int)
parser.add_argument("--end_file", type=int)
parser.add_argument("--file_dir", type=str, default=None)
parser.add_argument("--results_dir", type=str, default=None)
parser.add_argument("--kvcache_dir", type=str, default=None)

args = parser.parse_args()

print("args: ", args)


compress_config = CompressionConfig(
    compress_method=args.compress_method,
    rank=args.rank,
    rankv=args.rankv,
    prefill_rank = args.prefillrank,
    prefill_rankv = args.prefillrankv,
    loop=args.loop,
    quantize_bit=args.quantize_bit,
    group_num=args.group_num,
    group_size = args.group_size,
    top_k=args.top_k,
    left=args.left,
    attention_number=args.attention_number,
    batch_num=args.batch_size,
    streaming=args.streaming,
    streaming_gap=args.streaming_gap,
    stream_grouping=args.stream_grouping,
)

if compress_config is not None:
        compress_config.copy_for_all_attention()
        compress_config.calculate_compress_ratio_list(4095, 4096)

print("compress_config: ", compress_config)

tokenizer = AutoTokenizer.from_pretrained(args.model)

model = SimulatedGearLlamaForCausalLM.from_pretrained(
    args.model,
    cache_dir=cache_dir,
    device_map="cuda",
    compress_config=compress_config,
    torch_dtype=torch.float16,
    use_cache = True,
)

for i in range(args.start_file, args.end_file):
    with open(f"{args.file_dir}/{i}.txt", "r") as f:
        text = f.read()
    tokenized1 = tokenizer(text, return_tensors="pt")
    input_ids1 = tokenized1.input_ids.cuda()
    #print("input_ids1.shape: ", input_ids1.shape)

    gen_tokens = model.generate(
        input_ids1[:, :-1],
        use_cache = True,
        max_new_tokens = 1,
        return_dict_in_generate = True
    )
    torch.cuda.synchronize()
    kv = gen_tokens['past_key_values']

    print("past_key_values shape for ", i, ": (",
          len(kv), ", ",
          len(kv[0]), ", ",
         kv[0][0].shape, ")")

    #torch.save(kv, f"{args.kvcache_dir}/kv{i}.pt")

    gen_text = tokenizer.decode(gen_tokens['sequences'][0], skip_special_tokens = True)
    #print("gen_text after first generate on ", i, ".txt: ", gen_text)

    text2 = "The first topic we discussed was "
    tokenized2 = tokenizer(text2, return_tensors = "pt")
    input_ids2 = tokenized2.input_ids.cuda()
    
    generated2 = model.generate(
        input_ids2,
        max_new_tokens = 100,
        past_key_values = kv,
        return_dict_in_generate = True,
        attention_mask = tokenized1["attention_mask"].cuda()
    )

    print('-'*50)
    gen_text = tokenizer.decode(generated2['sequences'][0], skip_special_tokens = True)
    with open(f"{args.results_dir}/{i}_result.txt", "w") as f:
        f.write(gen_text)
    print("gen_text after second generate on ", i, ".txt: \n", gen_text)
    print('-'*50)
