"""Canonical class definitions and constants for the dacl10k dataset."""

DACL10K_CLASSES: tuple[str, ...] = (
    "Crack",
    "ACrack",
    "Wetspot",
    "Efflorescence",
    "Rust",
    "Rockpocket",
    "Hollowareas",
    "Cavity",
    "Spalling",
    "Restformwork",
    "ExposedRebars",
    "Bearing",
    "Drainage",
    "ExpansionJoint",
    "Graffiti",
    "Weathering",
    "Joint",
    "PEquipment",
    "WFlange",
)

CLASS_TO_IDX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(DACL10K_CLASSES)}

NUM_CLASSES: int = 19
