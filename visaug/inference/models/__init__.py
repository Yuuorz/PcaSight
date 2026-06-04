from .patch_llava import inject_llava_pca

def inject_s1mini_pca(model, args):
    """Lazy import patch_s1mini to avoid requiring transformers.models.qwen3 in LLaVA env."""
    from .patch_s1mini import inject_s1mini_pca as _inject
    return _inject(model, args)