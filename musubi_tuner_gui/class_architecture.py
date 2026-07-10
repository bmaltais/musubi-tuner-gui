from dataclasses import dataclass, field


@dataclass
class ArchitectureSpec:
    key: str
    label: str
    train_script: str
    cache_latents_script: str
    cache_teo_script: str
    model_field_groups: list = field(default_factory=list)
    unsupported_shared_args: set = field(default_factory=set)
    extra_args: list = field(default_factory=list)


REGISTRY = {
    "hunyuanvideo": ArchitectureSpec(
        key="hunyuanvideo",
        label="HunyuanVideo",
        train_script="hv_train_network.py",
        cache_latents_script="cache_latents.py",
        cache_teo_script="cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "dit_dtype",
            "hv_extras",
            "dual_text_encoder",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[],
    ),
    "wan": ArchitectureSpec(
        key="wan",
        label="Wan 2.1/2.2",
        train_script="wan_train_network.py",
        cache_latents_script="wan_cache_latents.py",
        cache_teo_script="wan_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "wan_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[
            "task",
            "dit_high_noise",
            "timestep_boundary",
            "t5",
            "clip",
            "fp8_scaled",
            "fp8_t5",
            "vae_cache_cpu",
        ],
    ),
    "qwen_image": ArchitectureSpec(
        key="qwen_image",
        label="Qwen-Image",
        train_script="qwen_image_train_network.py",
        cache_latents_script="qwen_image_cache_latents.py",
        cache_teo_script="qwen_image_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "single_text_encoder",
            "model_version",
            "qwen_image_extras",
            "fp8_vl",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[
            "text_encoder",
            "fp8_vl",
            "model_version",
            "num_layers",
            "remove_first_image_from_target",
        ],
    ),
    "zimage": ArchitectureSpec(
        key="zimage",
        label="Z-Image",
        train_script="zimage_train_network.py",
        cache_latents_script="zimage_cache_latents.py",
        cache_teo_script="zimage_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "single_text_encoder",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=["text_encoder"],
    ),
    "flux_2": ArchitectureSpec(
        key="flux_2",
        label="FLUX.2",
        train_script="flux_2_train_network.py",
        cache_latents_script="flux_2_cache_latents.py",
        cache_teo_script="flux_2_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "single_text_encoder",
            "model_version",
            "flux_2_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=["text_encoder", "fp8_text_encoder", "model_version"],
    ),
    "flux_kontext": ArchitectureSpec(
        key="flux_kontext",
        label="FLUX.1 Kontext",
        train_script="flux_kontext_train_network.py",
        cache_latents_script="flux_kontext_cache_latents.py",
        cache_teo_script="flux_kontext_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "dual_text_encoder",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=["text_encoder1", "text_encoder2", "fp8_t5"],
    ),
    "hv_1_5": ArchitectureSpec(
        key="hv_1_5",
        label="HunyuanVideo 1.5",
        train_script="hv_1_5_train_network.py",
        cache_latents_script="hv_1_5_cache_latents.py",
        cache_teo_script="hv_1_5_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "dit_dtype",
            "single_text_encoder",
            "fp8_vl",
            "image_encoder",
            "hv_1_5_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[
            "text_encoder",
            "fp8_vl",
            "hv15_task",
            "byt5",
            "image_encoder",
            "vae_enable_patch_conv",
            "vae_sample_size",
        ],
    ),
    "framepack": ArchitectureSpec(
        key="framepack",
        label="FramePack",
        train_script="fpack_train_network.py",
        cache_latents_script="fpack_cache_latents.py",
        cache_teo_script="fpack_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "hv_extras",
            "dual_text_encoder",
            "fp8_common",
            "image_encoder",
            "framepack_extras",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[
            "image_encoder",
            "latent_window_size",
            "f1",
            "bulk_decode",
            "one_frame",
        ],
    ),
    "kandinsky5": ArchitectureSpec(
        key="kandinsky5",
        label="Kandinsky 5",
        train_script="kandinsky5_train_network.py",
        cache_latents_script="kandinsky5_cache_latents.py",
        cache_teo_script="kandinsky5_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "kandinsky5_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=["kandinsky5_task", "text_encoder_clip", "text_encoder_qwen"],
    ),
    "hidream_o1": ArchitectureSpec(
        key="hidream_o1",
        label="HiDream-O1-Image",
        train_script="hidream_o1_train_network.py",
        cache_latents_script="hidream_o1_cache_pixel.py",
        cache_teo_script="hidream_o1_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "hidream_o1_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[
            "hidream_task",
            "hidream_model_type",
            "fp8_te",
            "dino_loss_weight",
        ],
    ),
    "ideogram4": ArchitectureSpec(
        key="ideogram4",
        label="Ideogram4",
        train_script="ideogram4_train_network.py",
        cache_latents_script="ideogram4_cache_latents.py",
        cache_teo_script="ideogram4_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "dit_dtype",
            "single_text_encoder",
            "ideogram4_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=[
            "text_encoder",
            "unconditional_dit",
            "sampler_preset",
            "initial_sigma",
            "use_unconditional_dit_for_lora_sampling",
            "validate_caption_structure",
            "warn_on_caption_issues",
        ],
    ),
    "krea2": ArchitectureSpec(
        key="krea2",
        label="Krea 2",
        train_script="krea2_train_network.py",
        cache_latents_script="krea2_cache_latents.py",
        cache_teo_script="krea2_cache_text_encoder_outputs.py",
        model_field_groups=[
            "dit_vae",
            "single_text_encoder",
            "krea2_extras",
            "fp8_common",
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=["text_encoder", "turbo_dit", "turbo_dit_cache"],
    ),
}

DEFAULT_ARCHITECTURE = "hunyuanvideo"


def get_architecture(key: str) -> ArchitectureSpec:
    return REGISTRY.get(key, REGISTRY[DEFAULT_ARCHITECTURE])


def architecture_choices() -> list:
    return [(spec.label, spec.key) for spec in REGISTRY.values()]
