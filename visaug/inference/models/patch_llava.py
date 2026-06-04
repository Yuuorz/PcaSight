"""
Model-specific intercept patching for LLaVA.
"""
import torch
from PCAPullAdapter import PCAPullAdapter

def inject_llava_pca(model, args, start_layer=2, end_layer=31):
    model.config._attn_implementation = "eager"
    if hasattr(model, 'model'):
        model.model.config._attn_implementation = "eager"

    print(f"=== Injecting Uncentered PCA Subspace into LLaVA ===")
    for i, layer in enumerate(model.model.layers):
        if start_layer <= i <= end_layer:
            zone_weight = 0.5 if i <= 8 else 1.0

            adap = PCAPullAdapter(layer.self_attn.config)
            adap.load_state_dict(layer.self_attn.state_dict())

            adap._pca_k = args.pca_k
            adap._scale = args.scale
            adap._zone_weight = zone_weight
            
            adap = adap.half().cuda()
            layer.self_attn = adap
    return model