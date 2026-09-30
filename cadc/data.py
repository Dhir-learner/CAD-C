"""Load nodule shards written by cadc.preprocess and serve them for training."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from cadc.preprocess import HU_MAX, HU_MIN, PATCH

# Mean radiologist malignancy rating (1-5) thresholds. Nodules in between are
# ambiguous and left out, the usual convention for LIDC-IDRI.
BENIGN_MAX, MALIGNANT_MIN = 2.5, 3.5


def load_nodules(shard_dir, min_readers=1):
    """Return (patches, table) for every nodule with a clear benign/malignant label.

    patches is an int16 array (N, 64, 64, 64) in HU; table has one row per patch.
    """
    shards = sorted(Path(shard_dir).glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no .npz shards in {shard_dir}")
    rows, patches = [], []
    for path in shards:
        with np.load(path) as z:
            ratings_all = z["malignancy"]
            if len(ratings_all) == 0:
                continue
            shard_patches = z["patches"]
            for i, ratings in enumerate(ratings_all):
                ratings = ratings[ratings > 0]
                if len(ratings) < min_readers:
                    continue
                score = float(ratings.mean())
                if BENIGN_MAX < score < MALIGNANT_MIN:
                    continue
                rows.append(dict(
                    patient_id=str(z["patient_id"]),
                    shard=path.stem,
                    nodule=i,
                    n_readers=len(ratings),
                    malignancy=score,
                    diameter_mm=float(z["diameter_mm"][i]),
                    label=int(score >= MALIGNANT_MIN),
                ))
                patches.append(shard_patches[i])
    return np.stack(patches), pd.DataFrame(rows)


def assign_folds(table, n_folds, seed):
    """Add a 'fold' column. All nodules of a patient share a fold, so no patient
    appears in both training and validation."""
    table = table.copy()
    table["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(splitter.split(table, table["label"], table["patient_id"])):
        table.loc[table.index[idx], "fold"] = fold
    return table


class NoduleDataset(Dataset):
    """Crops a crop^3 cube from each 64^3 patch: random with flips and rotations
    when training, centred otherwise. Intensities are scaled to [0, 1]."""

    def __init__(self, patches, labels, train, crop=48):
        self.patches = patches
        self.labels = np.asarray(labels, dtype=np.float32)
        self.train = train
        self.crop = crop

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        x = self.patches[i]
        c, margin = self.crop, PATCH - self.crop
        if self.train:
            # torch's RNG is seeded per DataLoader worker; NumPy's is not.
            z, y, w = torch.randint(0, margin + 1, (3,)).tolist()
            x = x[z:z + c, y:y + c, w:w + c]
            flips = [ax for ax in range(3) if torch.rand(1) < 0.5]
            if flips:
                x = np.flip(x, flips)
            x = np.rot90(x, int(torch.randint(0, 4, (1,))), axes=(1, 2))
        else:
            s = margin // 2
            x = x[s:s + c, s:s + c, s:s + c]
        x = (x.astype(np.float32) - HU_MIN) / (HU_MAX - HU_MIN)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.tensor(self.labels[i])
