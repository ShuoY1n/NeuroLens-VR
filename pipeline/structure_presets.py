"""Structure presets for per-label mesh export.

The current dataset uses DKT31_CMA labels aligned to MNI152. In practice these
volumes often include a FreeSurfer/aseg-style subcortical label subset
(small integer IDs) plus cortical labels in the 1000/2000 ranges.

This module provides a curated set of "important" structures we want available
as individually toggleable meshes in the viewer.
"""

from __future__ import annotations


def important_structures() -> list[dict]:
    """Return a list of structure specs `{sourceLabelId, label, id?}`.

    Notes:
    - IDs are the common FreeSurfer aseg integer labels.
    - If a label is not present in a subject, export simply skips it.
    """
    # Keep this list small/curated; names come from Dataset/label_mappings.json.
    label_ids = [
        # Ventricles + CSF spaces
        4,
        43,
        5,
        44,
        14,
        15,
        # Deep gray nuclei
        10,
        49,
        11,
        50,
        12,
        51,
        13,
        52,
        26,
        58,
        # Medial temporal
        17,
        53,
        18,
        54,
        # Cerebellum
        7,
        46,
        8,
        47,
        # Brainstem / ventral diencephalon
        16,
        28,
    ]

    items = [{"sourceLabelId": lid} for lid in label_ids]

    # Give each a stable id string used by the viewer.
    for it in items:
        lid = int(it["sourceLabelId"])
        it["id"] = f"aseg_{lid}"
    return items

