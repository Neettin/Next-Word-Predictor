import json

with open(r"C:\Users\nitin gupta\next_word_predictor_app\model_config.json") as f:
    config = json.load(f)

def patch_config(obj):
    if isinstance(obj, dict):
        # Remove all Keras 3 only keys
        obj.pop("optional", None)
        obj.pop("batch_shape", None)
        obj.pop("quantization_config", None)

        # Fix dtype: if it's a dict (Keras 3 DTypePolicy), replace with plain string
        if "dtype" in obj and isinstance(obj["dtype"], dict):
            obj["dtype"] = obj["dtype"].get("config", {}).get("name", "float32")

        for k, v in list(obj.items()):
            patch_config(v)

    elif isinstance(obj, list):
        for item in obj:
            patch_config(item)

patch_config(config)

with open(r"C:\Users\nitin gupta\next_word_predictor_app\model_config.json", "w") as f:
    json.dump(config, f)

print("Done! Config patched successfully.")