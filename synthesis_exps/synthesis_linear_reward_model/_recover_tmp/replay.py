import json
import os

path = r"C:\Users\Asus\.cursor\projects\c-Users-Asus-Downloads-ttp-final-v2\agent-transcripts\eb196e86-e84b-4a2e-906a-516aabeb7873\eb196e86-e84b-4a2e-906a-516aabeb7873.jsonl"
out_dir = r"c:\Users\Asus\Downloads\ttp - final_ v2\_recover_tmp"
os.makedirs(out_dir, exist_ok=True)


def is_mlp(text: str) -> bool:
    t = text.lower()
    return (
        "mlp_returns" in t
        or "calibrate_mlp_scale" in t
        or "train_mlp_ttp" in t
        or "shared mlp" in t
        or "synthetic_mlp_" in t
    )


snapshots = {}
files = {}
with open(path, "r", encoding="utf-8") as f:
    for li, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        content = obj.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            inp = block.get("input", {})
            p = inp.get("path", "")
            if "final_ttp_" not in p:
                continue
            base = os.path.basename(p)
            if name == "Write" and "contents" in inp:
                newc = inp["contents"]
                if base in files and not is_mlp(files[base]) and is_mlp(newc):
                    snapshots[base] = files[base]
                    print(f"SNAPSHOT {base} before MLP Write at {li}, len={len(files[base])}")
                files[base] = newc
            elif name == "StrReplace" and base in files:
                old, new = inp.get("old_string"), inp.get("new_string")
                if old is None or new is None:
                    continue
                if old not in files[base]:
                    print(f"WARN miss {base} @{li}")
                    continue
                if inp.get("replace_all"):
                    trial = files[base].replace(old, new)
                else:
                    trial = files[base].replace(old, new, 1)
                if not is_mlp(files[base]) and is_mlp(trial):
                    snapshots[base] = files[base]
                    print(f"SNAPSHOT {base} before MLP StrReplace at {li}, len={len(files[base])}")
                files[base] = trial

print("Snapshot keys:", list(snapshots.keys()))
for base, text in snapshots.items():
    fp = os.path.join(out_dir, "LINEAR_" + base)
    with open(fp, "w", encoding="utf-8") as out:
        out.write(text)
    print(f"Wrote {fp} ({len(text)} bytes)")
    for line2 in text.splitlines():
        if "default=" in line2 and ("out_" in line2 or "synthetic" in line2):
            print(" ", line2.strip())
    print("  _segment_returns", "_segment_returns" in text)
    print("  consensus 0.5", "consensus_coef=0.5" in text)
    print("  lr_theta 0.05", "lr_theta: float = 0.05" in text or "lr_theta=0.05" in text)

# Also dump current non-MLP files if never converted via Write but via gradual replace
for base in [
    "final_ttp_synthetic_shared_core.py",
    "final_ttp_synthetic_shared_all.py",
    "final_ttp_pair_overlap_shared.py",
    "final_ttp_partial_adversary_shared.py",
    "final_ttp_expert_ratio_sweep_shared.py",
    "final_ttp_synthetic_wk_ablation.py",
]:
    if base not in snapshots and base in files and not is_mlp(files[base]):
        fp = os.path.join(out_dir, "LINEAR_" + base)
        with open(fp, "w", encoding="utf-8") as out:
            out.write(files[base])
        print(f"FINAL-NONMLP {base} -> {fp}")
