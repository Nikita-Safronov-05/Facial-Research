import re
from pathlib import Path

def list_face_objs(folder: Path):
    if not folder.is_dir():
        return []
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".obj"]

    def idx(p: Path):
        m = re.search(r"simface_(\d+)\.obj$", p.name, flags=re.IGNORECASE)
        return int(m.group(1)) if m else 0

    return sorted(files, key=idx)


def parse_obj_vertices(obj_path: Path):
    """Parse vertex lines (v x y z) from an OBJ, preserving order; returns a flat list of floats [x1,y1,z1,...]."""
    flat = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    # parts: ['v', 'x', 'y', 'z'] possibly more
                    try:
                        flat.append(f"{float(parts[1]):.6f}")
                        flat.append(f"{float(parts[2]):.6f}")
                        flat.append(f"{float(parts[3]):.6f}")
                    except ValueError:
                        # Skip malformed vertex line
                        continue
            # Stop early if we reached faces to avoid unnecessary parsing
            elif line.startswith("f ") and flat:
                # We assume all 'v ' lines come before 'f ' lines in our generated files
                # so once we hit 'f', vertices are done
                break
    return flat


def write_xyz_csv(seed_dir: Path):
    faces_dir = seed_dir / "faces"
    obj_paths = list_face_objs(faces_dir)
    if not obj_paths:
        print(f"[skip] No OBJ files found in {faces_dir}")
        return False

    out_csv = seed_dir / "XYZ.csv"
    if out_csv.exists():
        print(f"[skip] Exists: {out_csv}")
        return True

    with out_csv.open("w", encoding="utf-8", newline="\n") as out:
        for i, objp in enumerate(obj_paths, start=1):
            flat = parse_obj_vertices(objp)
            if not flat:
                print(f"  [warn] No vertices in {objp}")
                continue
            out.write(",".join(flat))
            out.write("\n")
            if i % 200 == 0:
                print(f"  wrote {i}/{len(obj_paths)} rows -> {out_csv}")

    print(f"[done] Saved {len(obj_paths)}x{len(flat)} XYZ to {out_csv}")
    return True

def main():
    repo_dir = Path(__file__).resolve().parent
    processed_root = repo_dir / "processed"

    for seed in range(1, 101):
        seed_dir = processed_root / f"sim_2000_d4_{seed}"
        if not seed_dir.is_dir():
            print(f"[skip] Missing: {seed_dir}")
            continue
        print(f"[seed {seed:03d}] Processing {seed_dir}")
        write_xyz_csv(seed_dir)

    print("All done.")


if __name__ == "__main__":
    main()
