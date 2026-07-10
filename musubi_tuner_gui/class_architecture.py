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
        model_field_groups=["dit_vae", "hv_extras", "flow_matching", "perf"],
        unsupported_shared_args=set(),
        extra_args=[],
    ),
    "wan": ArchitectureSpec(
        key="wan",
        label="Wan 2.1/2.2",
        train_script="wan_train_network.py",
        cache_latents_script="wan_cache_latents.py",
        cache_teo_script="wan_cache_text_encoder_outputs.py",
        model_field_groups=["dit_vae", "wan_extras", "flow_matching", "perf"],
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
        model_field_groups=["dit_vae", "single_text_encoder", "flow_matching", "perf"],
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
            "flow_matching",
            "perf",
        ],
        unsupported_shared_args=set(),
        extra_args=["text_encoder", "fp8_text_encoder", "model_version"],
    ),
}

DEFAULT_ARCHITECTURE = "hunyuanvideo"


def get_architecture(key: str) -> ArchitectureSpec:
    return REGISTRY.get(key, REGISTRY[DEFAULT_ARCHITECTURE])


def architecture_choices() -> list:
    return [(spec.label, spec.key) for spec in REGISTRY.values()]
