"""Attack sub-package."""
from fedsys.adversarial.attack.poison import (
    PoisonedBPRPairDataset,
    build_poisoned_dataloader,
    poisoned_num_users,
)
from fedsys.adversarial.attack.target import (
    select_target_item,
    select_target_from_clean_model,
    choose_target_genre,
    benign_segment_users,
)

__all__ = [
    "PoisonedBPRPairDataset",
    "build_poisoned_dataloader",
    "poisoned_num_users",
    "select_target_item",
    "select_target_from_clean_model",
    "choose_target_genre",
    "benign_segment_users",
]
