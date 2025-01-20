# How to test manually

uv run ./musubi-tuner/cache_latents.py --dataset_config "./test/config/dataset.toml" --vae "C:\Users\berna\Downloads\pytorch_model.pt" --vae_chunk_size 32 --vae_tiling

uv run ./musubi-tuner/cache_text_encoder_outputs.py --dataset_config "./test/config/dataset.toml"  --text_encoder1 "C:\Users\berna\Downloads\llava_llama3_fp16.safetensors" --text_encoder2 "C:\Users\berna\Downloads\clip_l.safetensors" --batch_size 1

uv run accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ./musubi-tuner/hv_train_network.py  --dit C:/Users/berna/Downloads/mp_rank_00_model_states_fp8.safetensors --dataset_config ./test/config/dataset.toml --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora --network_dim 32 --timestep_sampling shift --discrete_flow_shift 7.0 --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 --output_dir ./test/output --output_name name-of-lora