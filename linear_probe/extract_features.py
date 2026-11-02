"""
Extract features from Qwen3-VL-2B-Instruct for every photo in manifest.csv.
STATUS: written without access to the model weights, so it is UNTESTED. Run it on 2 photos first, fix API
mismatches (transformers version names), then run on all 60. Keep the SAME model revision and the SAME
frozen identification prompt as the RegionalPack-VLM evaluation.

manifest.csv columns: photo_id, product_id (0..19), condition (straight|tilt|occluded), path,
                      primary (1 for the 18 primary products, 0 for the apple-juice pair),
                      model_correct (1/0, from the FROZEN Set A scorer), failure_type (optional)
Output: features.npz with one array (N, D) per site, rows in manifest order.
Sites:  vision_out                  mean-pooled output of the vision tower (+merger)
        llm_img_L{l}                mean of hidden states over the IMAGE tokens at LLM layer l   ("evidence")
        llm_last_L{l}               hidden state at the LAST prompt token at layer l              ("decision")
        pixel_thumb, color_hist     trivial baselines (no model), to show the probe is not just reading colour
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"        # same pinned revision as the original evaluation
IDENT_PROMPT = "What product is shown in this image? If you can identify the specific brand, name it. If you cannot identify the specific brand with confidence, say so explicitly rather than guessing."
# Frozen evaluation pixel budget (configs/eval_config.yaml and benchmark_environment.json).
FROZEN_MIN_PIXELS = 65536
FROZEN_MAX_PIXELS = 1003520

def _describe(o, prefix=""):
    if torch.is_tensor(o):
        print(f"STRUCT {prefix} tensor {tuple(o.shape)} {o.dtype}")
        return
    if isinstance(o, (tuple, list)):
        print(f"STRUCT {prefix} {type(o).__name__} len={len(o)}")
        for i, item in enumerate(o):
            _describe(item, prefix + f"[{i}]")
        return
    print(f"STRUCT {prefix} {type(o).__name__}")
    for name in ("pooler_output", "last_hidden_state", "deepstack_features"):
        if hasattr(o, name):
            _describe(getattr(o, name), prefix + "." + name)

def _merger_tensor(o):
    """Spatial-merger output: the image embeddings the language model receives.

    The vision module's last_hidden_state is pre-merger. Its forward hook can
    also return a tuple, and pooler_output can itself be a one-element split.
    """
    if torch.is_tensor(o):
        return o
    if hasattr(o, "pooler_output") and o.pooler_output is not None:
        return _merger_tensor(o.pooler_output)
    if isinstance(o, (tuple, list)):
        objects = [item for item in o if hasattr(item, "pooler_output")]
        if objects:
            return _merger_tensor(objects[0])
        tensors = []
        for item in o:
            if torch.is_tensor(item) or isinstance(item, (tuple, list)) or hasattr(item, "pooler_output"):
                tensors.append(_merger_tensor(item))
        if not tensors:
            raise TypeError(f"vision hook output has no tensor: {type(o)}")
        wide = [t for t in tensors if t.shape[-1] >= 2048]
        chosen = wide if wide else tensors
        if len(chosen) == 1:
            return chosen[0]
        # One image is sometimes a 1-tuple after torch.split. Concatenate pieces of one sequence.
        return torch.cat([t.reshape(-1, t.shape[-1]) for t in chosen], dim=0)
    if hasattr(o, "last_hidden_state"):
        return _merger_tensor(o.last_hidden_state)
    raise TypeError(f"unrecognised vision output: {type(o)}")

def pooled(o):
    tensor = _merger_tensor(o)
    return tensor.float().reshape(-1, tensor.shape[-1]).mean(0).cpu().numpy()

def apply_frozen_pixel_budget(processor):
    image_processor = processor.image_processor
    previous = getattr(image_processor, "size", None)
    if previous is not None and hasattr(previous, "longest_edge"):
        previous.longest_edge = FROZEN_MAX_PIXELS
        if hasattr(previous, "shortest_edge"):
            previous.shortest_edge = FROZEN_MIN_PIXELS
    elif isinstance(previous, dict):
        previous["longest_edge"] = FROZEN_MAX_PIXELS
        previous["shortest_edge"] = FROZEN_MIN_PIXELS
    else:
        image_processor.size = {"longest_edge": FROZEN_MAX_PIXELS, "shortest_edge": FROZEN_MIN_PIXELS}
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = FROZEN_MAX_PIXELS
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = FROZEN_MIN_PIXELS

def color_hist(img, bins=8):
    a = np.asarray(img.convert("RGB").resize((128, 128))).reshape(-1, 3)
    h = np.concatenate([np.histogram(a[:, k], bins=bins, range=(0, 255))[0] for k in range(3)]).astype(np.float32)
    return h / h.sum()

def main(manifest, out, limit=None):
    df = pd.read_csv(manifest)
    if limit: df = df.head(limit)
    if torch.cuda.is_available():
        device_map, dtype = "cuda", torch.float16
        print("device", torch.cuda.get_device_name(0), "dtype", dtype)
    else:
        # This machine has no CUDA device. The frozen evaluation ran float16 on a T4.
        device_map, dtype = "cpu", torch.float32
        print("device cpu dtype float32; CUDA is not available, so this is not the Colab T4")
    local = Path(__file__).resolve().parents[1] / "models" / "Qwen3-VL-2B-Instruct"
    if (local / "model.safetensors").is_file() and (local / "config.json").is_file():
        source, load_kwargs = str(local), dict(local_files_only=True, device_map=device_map, attn_implementation="sdpa")
        print("loading pinned local snapshot", source)
    else:
        source, load_kwargs = MODEL_ID, dict(revision=REVISION, device_map=device_map, attn_implementation="sdpa")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(source, dtype=dtype, **load_kwargs).eval()
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = Qwen3VLForConditionalGeneration.from_pretrained(source, torch_dtype=dtype, **load_kwargs).eval()
    proc = AutoProcessor.from_pretrained(source, revision=None if source != MODEL_ID else REVISION, local_files_only=source != MODEL_ID)
    apply_frozen_pixel_budget(proc)
    n_layers = int(model.config.text_config.num_hidden_layers)
    merge = int(model.model.visual.spatial_merge_size)
    cap = []
    hook = model.model.visual.register_forward_hook(lambda m, i, o: cap.append(o))
    feats = {}
    def put(k, v): feats.setdefault(k, []).append(v)
    for _, r in df.iterrows():
        with Image.open(r["path"]) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": IDENT_PROMPT}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt")
        if hasattr(inputs, "pop"):
            inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)
        cap.clear()
        with torch.no_grad():
            o = model(**inputs, output_hidden_states=True)
        if len(o.hidden_states) != n_layers + 1:
            raise SystemExit(f"{r['photo_id']}: hidden_states={len(o.hidden_states)}, expected {n_layers + 1}")
        img_mask = inputs["input_ids"][0] == model.config.image_token_id
        n_tok = int(img_mask.sum().item())
        expected = int(inputs["image_grid_thw"][0].prod().item()) // (merge * merge)
        if n_tok != expected or n_tok <= 0:
            raise SystemExit(f"{r['photo_id']}: image tokens {n_tok}, grid expects {expected}")
        if len(cap) != 1:
            raise SystemExit(f"{r['photo_id']}: vision hook fired {len(cap)} times")
        if not feats:
            _describe(cap[0])
        vision = pooled(cap[0])
        if not np.isfinite(vision).all():
            raise SystemExit(f"{r['photo_id']}: vision_out is not finite")
        put("vision_out", vision)
        for l, hs in enumerate(o.hidden_states):                       # 0 = embeddings, last = final layer
            if l == 0: continue
            h = hs[0].float()
            img_vec = h[img_mask].mean(0)
            last_vec = h[-1]
            if not torch.isfinite(img_vec).all() or not torch.isfinite(last_vec).all():
                raise SystemExit(f"{r['photo_id']}: non-finite hidden state at layer {l}")
            put(f"llm_img_L{l}", img_vec.cpu().numpy())
            put(f"llm_last_L{l}", last_vec.cpu().numpy())
        print(
            "check", r["photo_id"],
            f"image={img.size[0]}x{img.size[1]}",
            f"pixels={tuple(inputs['pixel_values'].shape)}",
            f"image_tokens={n_tok}",
            f"hidden_states={len(o.hidden_states)}",
            f"vision_dim={vision.shape[0]}",
        )
        put("pixel_thumb", np.asarray(img.resize((16, 16))).astype(np.float32).ravel() / 255.0)
        put("color_hist", color_hist(img))
        print("done", r["photo_id"])
    hook.remove()
    np.savez_compressed(out, **{k: np.stack(v) for k, v in feats.items()})
    print("saved", out, {k: np.stack(v).shape for k, v in list(feats.items())[:3]})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", default="features.npz")
    ap.add_argument("--limit", type=int); a = ap.parse_args(); main(a.manifest, a.out, a.limit)
