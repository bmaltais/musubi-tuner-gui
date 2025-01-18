# How to test manually

uv run cache_latents.py --dataset_config "./test/dataset/darius kawasaki" --vae "path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt" --vae_chunk_size 32 --vae_tiling

uv run ./musubi-tuner/cache_latents.py --dataset_config "./test/config/dataset.toml" --vae "C:\Users\berna\Downloads\pytorch_model.pt" --vae_chunk_size 32 --vae_tiling

uv run ./musubi-tuner/cache_text_encoder_outputs.py --dataset_config "./test/config/dataset.toml"  --text_encoder1 "C:\Users\berna\Downloads\llava_llama3_fp16.safetensors" --text_encoder2 "C:\Users\berna\Downloads\clip_l.safetensors" --batch_size 1
