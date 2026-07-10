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
        model_field_groups=["dit_vae_te", "flow_matching", "perf"],
        unsupported_shared_args=set(),
        extra_args=[],
    ),
}

DEFAULT_ARCHITECTURE = "hunyuanvideo"


def get_architecture(key: str) -> ArchitectureSpec:
    return REGISTRY.get(key, REGISTRY[DEFAULT_ARCHITECTURE])


def architecture_choices() -> list:
    return [(spec.label, spec.key) for spec in REGISTRY.values()]
