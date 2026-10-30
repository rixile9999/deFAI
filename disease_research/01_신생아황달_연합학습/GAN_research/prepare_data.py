"""Audit sources and freeze patient-disjoint train/validation/test partitions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "NeoJaundice" / "data"
CROP = (.37, .39, .63, .65)  # Central skin patch; fixed before looking at test performance.


def prepare(output, seed=42, source=SOURCE):
    source = Path(source)
    if (output / "real_data.npz").exists() or (output / "audit.json").exists():
        raise FileExistsError("A completed preparation already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source / "metadata.csv", dtype={"patient_id": str})
    if frame.groupby("patient_id")[["blood(mg/dL)", "Treatment"]].nunique().to_numpy().max() != 1:
        raise ValueError("Conflicting outcomes within a patient")
    original_rows = len(frame)
    source_hashes = {name: sha256((source/"images"/name).read_bytes()).hexdigest() for name in frame.image_idx}
    frame["sha256"] = frame.image_idx.map(source_hashes)
    cross_patient_hashes = frame.groupby("sha256").patient_id.nunique()
    cross_patient_hashes = set(cross_patient_hashes.index[cross_patient_hashes > 1])
    conflicting = frame.loc[frame.sha256.isin(cross_patient_hashes)].copy()
    conflicting.to_csv(output / "quarantined_cross_patient_duplicates.csv", index=False)
    frame = frame.loc[~frame.sha256.isin(cross_patient_hashes)].copy()
    # A zero target has unverified meaning in the source; quarantine it explicitly.
    excluded = frame.loc[frame["blood(mg/dL)"] <= 0].copy()
    excluded.to_csv(output / "quarantined_zero_targets.csv", index=False)
    frame = frame.loc[frame["blood(mg/dL)"] > 0].copy().reset_index(drop=True)
    duplicate_rows = frame.loc[frame.sha256.duplicated()].copy()
    duplicate_rows.to_csv(output / "removed_within_patient_duplicates.csv", index=False)
    frame = frame.drop_duplicates("sha256").reset_index(drop=True)
    if not np.isfinite(frame[["blood(mg/dL)", "Treatment"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite outcome labels")
    missing = {k: int(frame[k].isna().sum()) for k in ["gestational_age", "age(day)", "weight"]}
    frame[list(missing)] = frame[list(missing)].astype(float)
    patients = frame.drop_duplicates("patient_id")
    tr_ids, other_ids = train_test_split(patients.patient_id, test_size=.30, random_state=seed,
                                        stratify=patients.Treatment)
    other = patients.set_index("patient_id").loc[other_ids]
    val_ids, test_ids = train_test_split(other_ids, test_size=.5, random_state=seed+1,
                                        stratify=other.Treatment)
    split = {p: name for name, ids in [("train", tr_ids), ("val", val_ids), ("test", test_ids)] for p in ids}
    frame["split"] = frame.patient_id.map(split)
    # Simulated institutions, NOT real hospitals. Keep one patient in one client.
    client = {}
    rng = np.random.default_rng(seed+2)
    for partition in ["train", "val", "test"]:
        subset = patients.loc[patients.patient_id.map(split) == partition]
        for label in (0, 1):
            ids = rng.permutation(subset.loc[subset.Treatment == label, "patient_id"].to_numpy())
            client.update({p: int(i % 3) for i, p in enumerate(ids)})
    frame["client"] = frame.patient_id.map(client)
    imputation = {}
    for site in range(3):
        imputation[site] = {}
        for column in missing:
            median = float(frame.loc[(frame.client == site) & (frame.split == "train"), column].median())
            if not np.isfinite(median):
                raise ValueError("No finite local training values for imputation")
            mask = (frame.client == site) & frame[column].isna()
            frame.loc[mask, column] = median
            imputation[site][column] = median
    if not np.isfinite(frame[list(missing)].to_numpy(float)).all():
        raise ValueError("Nonfinite input after local imputation")
    hashes, seen, images = {}, {}, []
    duplicates_within_patient = 0
    for row in frame.itertuples():
        path = source / "images" / row.image_idx
        digest = source_hashes[row.image_idx]
        if digest in seen:
            if seen[digest] != row.patient_id:
                raise ValueError(f"Duplicate image crosses patient identifiers: {row.image_idx}")
            duplicates_within_patient += 1
        seen[digest] = row.patient_id
        hashes[row.image_idx] = digest
        with Image.open(path) as original:
            im = original.convert("RGB")
            w, h = im.size
            im = im.crop(tuple(int(x*scale) for x, scale in zip(CROP, (w, h, w, h))))
            images.append(np.asarray(im.resize((64, 64), Image.Resampling.LANCZOS)))
    meta = np.stack([(frame.gender == "F").to_numpy(float), frame.gestational_age/40,
                     frame["age(day)"]/30, frame.weight/4000], axis=1).astype("float32")
    np.savez(output / "real_data.npz", images=np.stack(images), meta=meta,
             blood=frame["blood(mg/dL)"].to_numpy("float32"), treatment=frame.Treatment.to_numpy("float32"),
             patient=frame.patient_id.to_numpy(dtype=str), split=frame.split.to_numpy(dtype=str),
             client=frame.client.to_numpy(dtype="int64"))
    frame.to_csv(output / "split_manifest.csv", index=False)
    audit = {"source": str(source), "metadata_sha256": sha256((source/"metadata.csv").read_bytes()).hexdigest(),
             "source_rows": original_rows, "retained_rows": len(frame), "retained_patients": frame.patient_id.nunique(),
             "quarantined_patients": excluded.patient_id.unique().tolist(),
             "zero_target_reason": "Unverified zero-valued outcome; excluded from primary analysis, not clinically adjudicated",
             "within_patient_duplicate_images": duplicates_within_patient,
             "removed_within_patient_duplicate_rows": len(duplicate_rows),
             "quarantined_cross_patient_hashes": len(cross_patient_hashes),
             "quarantined_cross_patient_rows": len(conflicting),
             "source_missing_rows": missing, "local_training_imputation_medians": imputation,
             "crop_fraction": CROP, "image_size": 64, "seed": seed,
             "split_counts": {name: {"patients": int(group.patient_id.nunique()), "images": len(group)}
                              for name, group in frame.groupby("split")}, "image_sha256": hashes,
             "all_source_image_sha256": source_hashes}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    # Training examples only in the review grid.
    rows = frame.index[frame.split == "train"].to_numpy()
    selected = rng.choice(rows, 40, replace=False)
    sheet = Image.new("RGB", (640, 400), "white"); draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(selected):
        x, y = (j % 8)*80, (j//8)*80
        sheet.paste(Image.fromarray(images[i]), (x, y))
        draw.text((x, y+64), f"{frame.loc[i, 'blood(mg/dL)']:.1f}", fill="black")
    sheet.save(output / "real_training_patches.jpg")
    print(json.dumps({k: v for k, v in audit.items() if "sha256" not in k}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    prepare(args.output, args.seed, args.source)
