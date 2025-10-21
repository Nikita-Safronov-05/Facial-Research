from pathlib import Path
import torch
import sys

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent

print(f"Script dir: {script_dir}")
print(f"Repo root:  {repo_root}")
print(f"CWD:        {Path.cwd()}")

candidate_rel = [
    Path('encoder_files/encoder_data/retrained_best_autoencoder_full.pth'),
    Path('encoder_files/retrained_best_autoencoder_full.pth'),
]

# Build absolute candidates (both relative to repo root and script dir just in case)
ckpt_candidates = []
for rel in candidate_rel:
    ckpt_candidates.append(repo_root / rel)
    if script_dir != repo_root:
        ckpt_candidates.append(script_dir / rel.name)  # rarely needed

# Scan Unused for legacy copies
unused_dir = repo_root / 'Unused'
if unused_dir.is_dir():
    ckpt_candidates.extend(unused_dir.rglob('retrained_best_autoencoder_full.pth'))

print("Checking candidates:")
for p in ckpt_candidates:
    print(" -", p, "exists=" + str(p.exists()))

ckpt = None
for p in ckpt_candidates:
    if not p.exists():
        continue
    try:
        ckpt_obj = torch.load(p, map_location='cpu')
        if isinstance(ckpt_obj, dict):
            if 'latent_dim' in ckpt_obj:
                ckpt = ckpt_obj
                print(f"Loaded checkpoint: {p}")
                break
            else:
                print(f"File loaded but 'latent_dim' key missing: {p} (keys: {list(ckpt_obj.keys())[:10]})")
        else:
            print(f"Unexpected checkpoint object type from {p}: {type(ckpt_obj)}")
    except Exception as e:
        print(f"Failed loading {p}: {e}")

if ckpt is None:
    print('\nNo checkpoint with latent_dim found among candidates.')
    sys.exit(1)

print('\nCheckpoint metadata:')
print(' latent_dim        =', ckpt.get('latent_dim'))
print(' input_dim         =', ckpt.get('input_dim'))
print(' encoder_layers    =', ckpt.get('encoder_layers'))
print(' decoder_layers    =', ckpt.get('decoder_layers'))
print(' activation_name   =', ckpt.get('activation_name'))
print(' epochs_trained    =', ckpt.get('epochs_trained'))
print(' final_train_loss  =', ckpt.get('final_train_loss'))
print(' test_loss         =', ckpt.get('test_loss'))
