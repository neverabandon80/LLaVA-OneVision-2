import logging
import torch

# Filter out _extra_state keys from sharded state dict to handle old checkpoints
def filter_extra_state(sharded_state_dict, modules_to_filter=None, ops_to_filter=None):
    """Filter out _extra_state keys from sharded state dict to handle old checkpoints.
    
    Args:
        sharded_state_dict: The sharded state dict to filter
        modules_to_filter: List of module paths to filter (e.g., ["vision_model"])
        ops_to_filter: List of operator names to filter (e.g., ["pre_layernorm"])
    Example:    
        # Filter only the _extra_state of pre_layernorm in vision_model.
        load_kwargs['sharded_state_dict'] = filter_extra_state(
            load_kwargs['sharded_state_dict'],
            modules_to_filter=["vision_model"],
            ops_to_filter=["pre_layernorm"]
        )

        # Filter all _extra_state in vision_model.
        load_kwargs['sharded_state_dict'] = filter_extra_state(
            load_kwargs['sharded_state_dict'],
            modules_to_filter=["vision_model"]
        )

        # Filter all _extra_state of pre_layernorm
        load_kwargs['sharded_state_dict'] = filter_extra_state(
            load_kwargs['sharded_state_dict'],
            ops_to_filter=["pre_layernorm"]
        )

        # Filter all _extra_state
        load_kwargs['sharded_state_dict'] = filter_extra_state(load_kwargs['sharded_state_dict'])                    
    """
    if isinstance(sharded_state_dict, dict):
        filtered_dict = {}
        for key, value in sharded_state_dict.items():
            # Only check string keys for _extra_state
            if isinstance(key, str) and "_extra_state" in key:
                # Check if this key should be filtered
                should_filter = False
                
                # Both modules_to_filter and ops_to_filter specified: must match both
                if modules_to_filter and ops_to_filter:
                    module_match = any(module in key for module in modules_to_filter)
                    op_match = any(f".{op}." in key or key.endswith(f".{op}._extra_state") for op in ops_to_filter)
                    should_filter = module_match and op_match
                # Only modules_to_filter specified: match any module
                elif modules_to_filter:
                    should_filter = any(module in key for module in modules_to_filter)
                # Only ops_to_filter specified: match any op
                elif ops_to_filter:
                    should_filter = any(f".{op}." in key or key.endswith(f".{op}._extra_state") for op in ops_to_filter)
                # No filters specified: filter all
                else:
                    should_filter = True
                
                if not should_filter:
                    filtered_dict[key] = filter_extra_state(value, modules_to_filter, ops_to_filter)
            else:
                filtered_dict[key] = filter_extra_state(value, modules_to_filter, ops_to_filter)
        return filtered_dict
    elif isinstance(sharded_state_dict, list):
        return [filter_extra_state(item, modules_to_filter, ops_to_filter) for item in sharded_state_dict]
    else:
        return sharded_state_dict


def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: type, modules_to_filter=None, ops_to_filter=None
):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys: Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
        modules_to_filter: List of module paths to filter (e.g., ["vision_model"])
        ops_to_filter: List of operator names to filter (e.g., ["pre_layernorm"])
     
    Example:    
        # Filter only the _extra_state of pre_layernorm in vision_model.
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: _load_state_dict_hook_ignore_extra_state(
                module, incompatible_keys, 
                modules_to_filter=["vision_model"], 
                ops_to_filter=["pre_layernorm"]
            )
        )

        # Filter all _extra_state
        self.register_load_state_dict_post_hook(_load_state_dict_hook_ignore_extra_state)
    """
    for name, keys in incompatible_keys._asdict().items():
        for key in keys[::-1]:
            if "extra_state" in key:
                # Check if this key should be filtered
                should_filter = False
                
                # Both modules_to_filter and ops_to_filter specified: must match both
                if modules_to_filter and ops_to_filter:
                    module_match = any(module in key for module in modules_to_filter)
                    op_match = any(f".{op}." in key or key.endswith(f".{op}._extra_state") for op in ops_to_filter)
                    should_filter = module_match and op_match
                # Only modules_to_filter specified: match any module
                elif modules_to_filter:
                    should_filter = any(module in key for module in modules_to_filter)
                # Only ops_to_filter specified: match any op
                elif ops_to_filter:
                    should_filter = any(f".{op}." in key or key.endswith(f".{op}._extra_state") for op in ops_to_filter)
                # No filters specified: filter all
                else:
                    should_filter = True
                
                if should_filter:
                    logging.getLogger(__name__).warning(
                        f"_extra_state key {key} being removed from {name}"
                    )
                    keys.remove(key)