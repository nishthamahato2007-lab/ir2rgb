'''import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_INPUT_DIR = DATA_DIR / "train" / "input_thermal"
TRAIN_TARGET_DIR = DATA_DIR / "train" / "target_rgb"

VAL_INPUT_DIR = DATA_DIR / "val" / "input_thermal"
VAL_TARGET_DIR = DATA_DIR / "val" / "target_rgb"


# ============================================================
# LANDSAT DATASET
# ============================================================

class LandsatDataset(Dataset):

    def __init__(self, input_dir, target_dir):

        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)

        # Find all thermal patches
        self.input_files = sorted(
            self.input_dir.glob("*.npy")
        )

        # Find all RGB patches
        self.target_files = sorted(
            self.target_dir.glob("*.npy")
        )

        # ----------------------------------------------------
        # Check that input and target counts match
        # ----------------------------------------------------

        if len(self.input_files) != len(self.target_files):

            raise RuntimeError(
                f"Input/target mismatch:\n"
                f"Input patches: {len(self.input_files)}\n"
                f"Target patches: {len(self.target_files)}"
            )

        if len(self.input_files) == 0:

            raise RuntimeError(
                f"No patches found in {self.input_dir}"
            )

        print(
            f"Dataset loaded: {len(self.input_files)} samples"
        )

    # ========================================================
    # NUMBER OF SAMPLES
    # ========================================================

    def __len__(self):

        return len(self.input_files)

    # ========================================================
    # GET ONE SAMPLE
    # ========================================================

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Load thermal patch
        # ----------------------------------------------------

        thermal = np.load(
            self.input_files[index]
        ).astype(np.float32)

        # Shape:
        # (H, W)

        # Convert:
        # (H, W)
        #     ↓
        # (1, H, W)

        thermal = np.expand_dims(
            thermal,
            axis=0
        )

        # ----------------------------------------------------
        # Load RGB target
        # ----------------------------------------------------

        rgb = np.load(
            self.target_files[index]
        ).astype(np.float32)

        # Shape:
        # (H, W, 3)

        # PyTorch expects:
        # (C, H, W)

        rgb = np.transpose(
            rgb,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        thermal = torch.from_numpy(
            thermal
        )

        rgb = torch.from_numpy(
            rgb
        )

        # ----------------------------------------------------
        # Convert [0,1] → [-1,1]
        #
        # Pix2Pix generators normally use Tanh output.
        # Therefore both input and target are represented
        # in the same [-1,1] range.
        # ----------------------------------------------------

        thermal = thermal * 2.0 - 1.0

        rgb = rgb * 2.0 - 1.0

        return thermal, rgb


# ============================================================
# CREATE TRAIN DATASET
# ============================================================

def create_train_dataset():

    return LandsatDataset(
        TRAIN_INPUT_DIR,
        TRAIN_TARGET_DIR
    )


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def create_val_dataset():

    return LandsatDataset(
        VAL_INPUT_DIR,
        VAL_TARGET_DIR
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=4,
    num_workers=0
):

    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    train_dataset = create_train_dataset()

    val_dataset = create_val_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print("\nTrain samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    print("Batch size:", batch_size)

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    return train_loader, val_loader


# ============================================================
# TEST DATASET
# ============================================================

def test_dataset():

    print("\n" + "=" * 60)
    print("TESTING DATASET")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(
        batch_size=4
    )

    # --------------------------------------------------------
    # Get one training batch
    # --------------------------------------------------------

    thermal, rgb = next(
        iter(train_loader)
    )

    print("\nFirst training batch:")

    print(
        "Thermal shape:",
        thermal.shape
    )

    print(
        "RGB shape:",
        rgb.shape
    )

    print(
        "Thermal dtype:",
        thermal.dtype
    )

    print(
        "RGB dtype:",
        rgb.dtype
    )

    print(
        "Thermal range:",
        thermal.min().item(),
        "to",
        thermal.max().item()
    )

    print(
        "RGB range:",
        rgb.min().item(),
        "to",
        rgb.max().item()
    )

    # --------------------------------------------------------
    # Expected:
    #
    # Thermal:
    # [batch, 1, 256, 256]
    #
    # RGB:
    # [batch, 3, 256, 256]
    # --------------------------------------------------------

    assert thermal.ndim == 4
    assert rgb.ndim == 4

    assert thermal.shape[1] == 1
    assert rgb.shape[1] == 3

    assert thermal.shape[2] == 256
    assert thermal.shape[3] == 256

    assert rgb.shape[2] == 256
    assert rgb.shape[3] == 256

    print("\n✓ Dataset test passed")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_dataset()'''

import numpy as np
import torch

from pathlib import Path

from torch.utils.data import (
    Dataset,
    DataLoader
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DATA_DIR = (
    PROJECT_ROOT
    / "data"
)


# ============================================================
# PATHS
# ============================================================

TRAIN_INPUT_DIR = (
    DATA_DIR
    / "train"
    / "input_thermal"
)

TRAIN_TARGET_DIR = (
    DATA_DIR
    / "train"
    / "target_rgb"
)

VAL_INPUT_DIR = (
    DATA_DIR
    / "val"
    / "input_thermal"
)

VAL_TARGET_DIR = (
    DATA_DIR
    / "val"
    / "target_rgb"
)


# ============================================================
# DATASET
# ============================================================

class LandsatDataset(
    Dataset
):

    def __init__(
        self,
        input_dir,
        target_dir
    ):

        self.input_dir = Path(
            input_dir
        )

        self.target_dir = Path(
            target_dir
        )

        self.input_files = sorted(
            self.input_dir.glob(
                "*.npy"
            )
        )

        self.target_files = sorted(
            self.target_dir.glob(
                "*.npy"
            )
        )

        if len(
            self.input_files
        ) != len(
            self.target_files
        ):

            raise RuntimeError(
                f"Input/target mismatch\n"
                f"Input: "
                f"{len(self.input_files)}\n"
                f"Target: "
                f"{len(self.target_files)}"
            )

        if len(
            self.input_files
        ) == 0:

            raise RuntimeError(
                f"No patches found in "
                f"{self.input_dir}"
            )

        print(
            f"Dataset loaded: "
            f"{len(self.input_files)} samples"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(
        self
    ):

        return len(
            self.input_files
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Thermal
        # ----------------------------------------------------

        thermal = np.load(
            self.input_files[index]
        ).astype(
            np.float32
        )

        # H,W -> 1,H,W

        thermal = np.expand_dims(
            thermal,
            axis=0
        )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = np.load(
            self.target_files[index]
        ).astype(
            np.float32
        )

        # H,W,3 -> 3,H,W

        rgb = np.transpose(
            rgb,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        thermal = torch.from_numpy(
            thermal
        )

        rgb = torch.from_numpy(
            rgb
        )

        # ----------------------------------------------------
        # [0,1] -> [-1,1]
        # ----------------------------------------------------

        thermal = (
            thermal * 2.0
            - 1.0
        )

        rgb = (
            rgb * 2.0
            - 1.0
        )

        return {
            "thermal": thermal,
            "rgb": rgb
        }


# ============================================================
# DATASETS
# ============================================================

def create_train_dataset():

    return LandsatDataset(
        TRAIN_INPUT_DIR,
        TRAIN_TARGET_DIR
    )


def create_val_dataset():

    return LandsatDataset(
        VAL_INPUT_DIR,
        VAL_TARGET_DIR
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=4,
    num_workers=2
):

    train_dataset = (
        create_train_dataset()
    )

    val_dataset = (
        create_val_dataset()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# TEST
# ============================================================

def test_dataset():

    print("=" * 60)
    print("TESTING DATASET")
    print("=" * 60)

    train_loader, val_loader = (
        create_dataloaders(
            batch_size=4,
            num_workers=0
        )
    )

    batch = next(
        iter(train_loader)
    )

    thermal = batch[
        "thermal"
    ]

    rgb = batch[
        "rgb"
    ]

    print(
        "\nThermal shape:",
        thermal.shape
    )

    print(
        "RGB shape:",
        rgb.shape
    )

    print(
        "Thermal range:",
        thermal.min().item(),
        "to",
        thermal.max().item()
    )

    print(
        "RGB range:",
        rgb.min().item(),
        "to",
        rgb.max().item()
    )

    assert thermal.ndim == 4
    assert rgb.ndim == 4

    assert thermal.shape[1] == 1
    assert rgb.shape[1] == 3

    assert thermal.shape[2] == 256
    assert thermal.shape[3] == 256

    assert rgb.shape[2] == 256
    assert rgb.shape[3] == 256

    print(
        "\n✓ Dataset test passed"
    )


if __name__ == "__main__":

    test_dataset()