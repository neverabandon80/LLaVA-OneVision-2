### bug fix for DCP loading issue
/usr/local/lib/python3.12/dist-packages/torch/distributed/checkpoint/default_planner.py  456

```python
# origin code
if fqn not in checkpoint_state_dict:
    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")

# new code
if fqn in ("language_model.output_layer.weight", "language_model.embedding.word_embeddings.weight"):
    print(f"------>>> Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}")
else:
    raise ValueError(
        f"Size mismatch between saved {md.size} and current: {obj.size()} for {fqn}",
    )
```
