#Note this has to be run in GEAR/GenerationBench/GenerationTest directory

python run-gear-kvcache-param.py --rank 1 --rankv 1 --left 0.0001 --quantize_bit 2 --group_size 8 --loop 3 --top_k 2 --compress_method GEAR --streaming --streaming_gap 64

