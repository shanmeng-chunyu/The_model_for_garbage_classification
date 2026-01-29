
$env:HF_HUB_ENDPOINT = "https://hf-mirror.com"
$env:HF_ENDPOINT = "https://hf-mirror.com"


llamafactory-cli train `
    --stage sft `
    --resume_from_checkpoint saves\Qwen2-VL-2B-Instruct\lora\garbage_structed_finetune\checkpoint-5800 `
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct `
    --do_predict True `
    --finetuning_type lora `
    --quantization_bit 4 `
    --dataset_dir data `
    --eval_dataset garbage_multimodal_structed_test `
    --template qwen2_vl `
    --preprocessing_num_workers 1 `
    --cutoff_len 1024 `
    --per_device_eval_batch_size 2 `
    --predict_with_generate True `
    --max_new_tokens 512 `
    --top_p 0.7 `
    --temperature 0.95 `
    --output_dir saves\Qwen2-VL-2B-Instruct\lora\eval_results_finetune_structed_test `
    --report_to none `
    --trust_remote_code True