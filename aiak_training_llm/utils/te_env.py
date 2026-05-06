import os

def add_te_filter_module(module_name):
    """Append module to TE_FILTER_MODULE_OPT (no deduplication)"""
    env_val = os.environ.get("TE_FILTER_MODULE_OPT", "")
    
    if ";" in env_val:
        modules, ops = env_val.split(";", 1)
    else:
        modules, ops = env_val, ""

    if modules:
        modules += "," + module_name
    else:
        modules = module_name

    os.environ["TE_FILTER_MODULE_OPT"] = f"{modules};{ops}"


def add_te_filter_op(op_name):
    """Append op to TE_FILTER_MODULE_OPT (no deduplication)"""
    env_val = os.environ.get("TE_FILTER_MODULE_OPT", "")
    
    if ";" in env_val:
        modules, ops = env_val.split(";", 1)
    else:
        modules, ops = env_val, ""

    if ops:
        ops += "," + op_name
    else:
        ops = op_name

    os.environ["TE_FILTER_MODULE_OPT"] = f"{modules};{ops}"


def add_te_filter_module_and_op(module_name, op_name):
    """Add module and op simultaneously (no deduplication)"""
    add_te_filter_module(module_name)
    add_te_filter_op(op_name)
    
def get_te_filter_modules_and_ops() -> tuple[list[str], list[str]]:
    """
    Get modules and ops lists from TE_FILTER_MODULE_OPT.
    Raises ValueError if modules and ops have different lengths.
    No deduplication.
    
    Returns:
        tuple[list[str], list[str]]: modules list and ops list with the same length
    """
    env_val = os.environ.get("TE_FILTER_MODULE_OPT", "").strip()

    # Split into module part and op part
    if ";" in env_val:
        modules_part, ops_part = env_val.split(";", 1)
    else:
        modules_part, ops_part = env_val, ""

    # Parse into clean lists (no deduplication, keep empty strings if present)
    modules = [m.strip() for m in modules_part.split(",")] if modules_part else []
    ops = [o.strip() for o in ops_part.split(",")] if ops_part else []

    # CRITICAL: raise error if lengths do NOT match
    if len(modules) != len(ops):
        raise ValueError(
            f"TE_FILTER_MODULE_OPT: modules count ({len(modules)}) != ops count ({len(ops)}). "
            "Modules and ops must be paired 1:1."
        )

    return modules, ops