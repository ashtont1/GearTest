python gear-generate-responses.py --model lmsys/longchat-7b-16k --rank 1 --rankv 1 --left 0.0001 --quantize_bit 8 --group_size 8 --loop 3 --top_k 2 --compress_method GEAR --streaming --streaming_gap 64 --start_file 0 --end_file 50 --file_dir /dataheart/ashton/CacheGen2/CacheGen/test_data/16_prompts --results_dir results_dir/qb-8

