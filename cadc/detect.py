"""Find nodule candidates in a whole CT scan with MONAI's pretrained LUNA16 RetinaNet detector.

The detector ("lung_nodule_ct_detection" bundle, Apache-2.0) expects RAS orientation, 0.703125 x
0.703125 x 1.25 mm voxels and intensities scaled from [-1024, 300] HU to [0, 1]. This module builds
that input from our (row, col, slice) volume and maps the boxes back to (x=col, y=row, slice).

Download the bundle once:
    python -c "from monai.bundle import download; download(name='lung_nodule_ct_detection', bundle_dir='data/bundles')"

Note: LUNA16 is drawn from LIDC-IDRI, so on LIDC scans the detector is being run on data it may
have been trained on.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TARGET_SPACING = np.array([0.703125, 0.703125, 1.25])  # (x, y, z) mm
HU_RANGE = (-1024.0, 300.0)


def load_detector(bundle_dir, device, roi=(192, 192, 96), score_thresh=0.02):
    from monai.bundle import ConfigParser

    bundle_dir = Path(bundle_dir)
    parser = ConfigParser()
    parser.read_config(str(bundle_dir / "configs" / "inference.json"))
    parser["bundle_root"] = str(bundle_dir)
    parser["device"] = f"$torch.device('{device}')"
    network = parser.get_parsed_content("network")
    state = torch.load(bundle_dir / "models" / "model.pt", map_location=device)
    network.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    detector = parser.get_parsed_content("detector")
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(score_thresh=score_thresh, topk_candidates_per_level=1000,
                                         nms_thresh=0.22, detections_per_img=100)
    # Small windows keep memory within a 6 GB laptop GPU; stitching happens on the CPU.
    detector.set_sliding_window_inferer(roi_size=list(roi), overlap=0.25, sw_batch_size=1, mode="constant", device="cpu")
    detector.eval()
    return detector


@torch.no_grad()
def detect(detector, vol, spacing, device, min_score=0.3):
    """vol: HU volume indexed (row, col, slice), slices inferior -> superior; spacing (row, col, slice) mm.

    Returns candidates sorted by score: dicts with x (col), y (row), slice, score, diameter_mm.
    """
    # (row, col, slice) -> (x=col, y=row, z=slice). DICOM columns run to the patient's left and rows
    # to posterior (LPS), so flip both in-plane axes to get RAS.
    arr = torch.from_numpy(np.ascontiguousarray(vol.transpose(1, 0, 2)[::-1, ::-1])).float()
    src = np.array([spacing[1], spacing[0], spacing[2]], dtype=float)
    size = np.maximum(1, np.round(np.array(arr.shape) * src / TARGET_SPACING)).astype(int)
    x = arr.to(device)[None, None]
    x = F.interpolate(x, size=tuple(int(s) for s in size), mode="trilinear", align_corners=False)
    x = ((x.clamp(*HU_RANGE) - HU_RANGE[0]) / (HU_RANGE[1] - HU_RANGE[0]))[0]
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        out = detector([x], use_inferer=True)[0]
    boxes = out["box"].float().cpu().numpy()          # xyzxyz in resampled voxels
    scores = out["label_scores"].float().cpu().numpy()

    scale = np.array(arr.shape) / size                  # resampled voxel -> original voxel
    found = []
    for b, s in zip(boxes, scores):
        if s < min_score:
            continue
        c = (b[:3] + b[3:]) / 2 * scale
        xr, yr, z = c
        col = arr.shape[0] - 1 - xr                     # undo the RAS flips
        row = arr.shape[1] - 1 - yr
        size_mm = (b[3:] - b[:3]) * TARGET_SPACING
        found.append(dict(x=float(col), y=float(row), slice=float(z), score=float(s),
                          diameter_mm=float(np.mean(size_mm[:2]))))
    return sorted(found, key=lambda d: -d["score"])
