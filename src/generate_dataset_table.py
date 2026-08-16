"""
generate_dataset_table.py — Produces the dataset table Reviewer 3
Comment 1 explicitly asked for: subjects/images per class, and the
train/test partitioning strategy actually used.

Run:
    python -m src.generate_dataset_table

Writes docs/dataset_table.md (Markdown, ready to paste into the
manuscript) and results/dataset_table.csv (raw numbers for the authors
to add age/sex/imaging-protocol columns from their IRB-approved subject
records — those fields are NOT contained in image filenames/pixels, so
this script cannot fabricate them; see the TODO columns in the output).
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.dataset import load_dataset_index
from src.utils import load_config


def main():
    cfg = load_config()
    class_names = cfg["data"]["classes"]
    samples = load_dataset_index(cfg["data"]["root_dir"], class_names)

    per_class_images = defaultdict(int)
    per_class_subjects = defaultdict(set)
    for s in samples:
        per_class_images[class_names[s.label]] += 1
        per_class_subjects[class_names[s.label]].add(s.subject_id)

    rows = []
    for c in class_names:
        rows.append({
            "Class": c,
            "N_subjects": len(per_class_subjects[c]),
            "N_images": per_class_images[c],
            "Age_range (yrs)": "TODO — pull from ADNI clinical/demographics table",
            "Sex (M/F)": "TODO — pull from ADNI clinical/demographics table",
            "Imaging protocol": "TODO — e.g. T1-weighted sagittal MPRAGE, scanner field strength",
        })
    df = pd.DataFrame(rows)

    total_row = {
        "Class": "Total",
        "N_subjects": sum(len(s) for s in per_class_subjects.values()),
        "N_images": sum(per_class_images.values()),
        "Age_range (yrs)": "-",
        "Sex (M/F)": "-",
        "Imaging protocol": "-",
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    Path("results").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)
    df.to_csv("results/dataset_table.csv", index=False)

    split_note = (
        f"\n\n**Partitioning strategy:** subject-wise {cfg['data']['n_folds']}-fold "
        f"cross-validation (GroupKFold on `subject_id`), repeated "
        f"{cfg['data']['n_repeats']} times; no subject's scans appear in both the "
        f"training and test partition of any fold "
        f"(enforced programmatically — see `src/dataset.py::subject_wise_kfold`).\n"
    )
    with open("docs/dataset_table.md", "w") as f:
        f.write("# Dataset Table (Reviewer 3, Comment 1)\n\n")
        f.write(df.to_markdown(index=False))
        f.write(split_note)

    print(df.to_string(index=False))
    print("\nWrote docs/dataset_table.md and results/dataset_table.csv")
    print("NOTE: age/sex/imaging-protocol columns must be filled in from your "
          "ADNI clinical metadata export — they cannot be derived from image pixels.")


if __name__ == "__main__":
    main()
