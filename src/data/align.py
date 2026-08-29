'''import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "val"
)


# ============================================================
# PATCH CONFIGURATION
# ============================================================

PATCH_SIZE = 256

STRIDE = 256

MIN_VALID_RATIO = 0.80

VAL_RATIO = 0.20


# ============================================================
# LOAD ONE SCENE
# ============================================================

def load_scene(
    scene_dir
):

    print(
        "\nLoading:",
        scene_dir.name
    )

    blue = np.load(
        scene_dir / "blue.npy"
    )

    green = np.load(
        scene_dir / "green.npy"
    )

    red = np.load(
        scene_dir / "red.npy"
    )

    thermal = np.load(
        scene_dir / "thermal_celsius.npy"
    )

    valid_mask = np.load(
        scene_dir / "valid_mask.npy"
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    shapes = [
        blue.shape,
        green.shape,
        red.shape,
        thermal.shape,
        valid_mask.shape
    ]

    if len(set(shapes)) != 1:

        raise ValueError(
            f"Dimension mismatch in {scene_dir.name}"
        )

    print(
        "Shape:",
        thermal.shape
    )

    return (
        blue,
        green,
        red,
        thermal,
        valid_mask
    )


# ============================================================
# CREATE RGB
# ============================================================

def create_rgb(
    blue,
    green,
    red
):

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    return rgb


# ============================================================
# NORMALIZE RGB
# ============================================================

def normalize_rgb(
    rgb
):

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    return rgb.astype(
        np.float32
    )


# ============================================================
# NORMALIZE THERMAL
# ============================================================

def normalize_thermal(
    thermal
):

    MIN_TEMP = -20.0

    MAX_TEMP = 60.0

    thermal_normalized = (
        thermal - MIN_TEMP
    ) / (
        MAX_TEMP - MIN_TEMP
    )

    thermal_normalized = np.clip(
        thermal_normalized,
        0.0,
        1.0
    )

    thermal_normalized = np.nan_to_num(
        thermal_normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return thermal_normalized.astype(
        np.float32
    )


# ============================================================
# FIND VALID PATCHES
# ============================================================

def find_valid_patches(
    valid_mask
):

    height, width = valid_mask.shape

    positions = []

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            mask_patch = valid_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            valid_ratio = np.mean(
                mask_patch
            )

            if valid_ratio >= MIN_VALID_RATIO:

                positions.append(
                    (y, x)
                )

    return positions


# ============================================================
# SAVE PATCHES
# ============================================================

def save_patches(
    positions,
    thermal,
    rgb,
    input_dir,
    target_dir,
    prefix,
    start_index
):

    saved = 0

    for local_index, (
        y,
        x
    ) in enumerate(positions):

        thermal_patch = thermal[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        rgb_patch = rgb[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
            :
        ]

        global_index = (
            start_index
            + local_index
        )

        input_path = (
            input_dir
            / f"{prefix}_{global_index:06d}.npy"
        )

        target_path = (
            target_dir
            / f"{prefix}_{global_index:06d}.npy"
        )

        np.save(
            input_path,
            thermal_patch
        )

        np.save(
            target_path,
            rgb_patch
        )

        saved += 1

        if saved % 100 == 0:

            print(
                f"  Saved {saved}/{len(positions)}"
            )

    return saved


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    train_input = (
        TRAIN_DIR
        / "input_thermal"
    )

    train_target = (
        TRAIN_DIR
        / "target_rgb"
    )

    val_input = (
        VAL_DIR
        / "input_thermal"
    )

    val_target = (
        VAL_DIR
        / "target_rgb"
    )

    train_input.mkdir(
        parents=True,
        exist_ok=True
    )

    train_target.mkdir(
        parents=True,
        exist_ok=True
    )

    val_input.mkdir(
        parents=True,
        exist_ok=True
    )

    val_target.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        train_input,
        train_target,
        val_input,
        val_target
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MULTI-SCENE PATCH GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    (
        train_input_dir,
        train_target_dir,
        val_input_dir,
        val_target_dir
    ) = create_directories()

    # --------------------------------------------------------
    # Find scenes
    # --------------------------------------------------------

    scene_dirs = sorted(
        [
            path
            for path in PROCESSED_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print(
        "\nScenes found:",
        len(scene_dirs)
    )

    if len(scene_dirs) == 0:

        raise RuntimeError(
            "No processed scenes found."
        )

    # --------------------------------------------------------
    # Process scenes
    # --------------------------------------------------------

    all_patches = []

    for scene_dir in scene_dirs:

        print("\n" + "-" * 60)

        print(
            "PROCESSING:",
            scene_dir.name
        )

        (
            blue,
            green,
            red,
            thermal,
            valid_mask
        ) = load_scene(
            scene_dir
        )

        rgb = create_rgb(
            blue,
            green,
            red
        )

        rgb = normalize_rgb(
            rgb
        )

        thermal = normalize_thermal(
            thermal
        )

        valid_mask = valid_mask.astype(
            bool
        )

        positions = find_valid_patches(
            valid_mask
        )

        print(
            "Valid patches:",
            len(positions)
        )

        # Store everything for splitting later

        for y, x in positions:

            all_patches.append(
                (
                    scene_dir.name,
                    y,
                    x,
                    thermal,
                    rgb
                )
            )

    # --------------------------------------------------------
    # Shuffle all patches
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMBINING DATASET")
    print("=" * 60)

    rng = np.random.default_rng(
        42
    )

    rng.shuffle(
        all_patches
    )

    total_patches = len(
        all_patches
    )

    val_count = int(
        total_patches
        * VAL_RATIO
    )

    val_patches = all_patches[
        :val_count
    ]

    train_patches = all_patches[
        val_count:
    ]

    print(
        "\nTotal patches:",
        total_patches
    )

    print(
        "Training patches:",
        len(train_patches)
    )

    print(
        "Validation patches:",
        len(val_patches)
    )

    # --------------------------------------------------------
    # Save training patches
    # --------------------------------------------------------

    print(
        "\nSaving training patches..."
    )

    for index, (
        scene_id,
        y,
        x,
        thermal,
        rgb
    ) in enumerate(
        train_patches
    ):

        thermal_patch = thermal[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        rgb_patch = rgb[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
            :
        ]

        np.save(
            train_input_dir
            / f"train_{index:06d}.npy",
            thermal_patch
        )

        np.save(
            train_target_dir
            / f"train_{index:06d}.npy",
            rgb_patch
        )

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"  Saved "
                f"{index + 1}/"
                f"{len(train_patches)}"
            )

    # --------------------------------------------------------
    # Save validation patches
    # --------------------------------------------------------

    print(
        "\nSaving validation patches..."
    )

    for index, (
        scene_id,
        y,
        x,
        thermal,
        rgb
    ) in enumerate(
        val_patches
    ):

        thermal_patch = thermal[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        rgb_patch = rgb[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
            :
        ]

        np.save(
            val_input_dir
            / f"val_{index:06d}.npy",
            thermal_patch
        )

        np.save(
            val_target_dir
            / f"val_{index:06d}.npy",
            rgb_patch
        )

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"  Saved "
                f"{index + 1}/"
                f"{len(val_patches)}"
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MULTI-SCENE PATCH GENERATION COMPLETE")
    print("=" * 60)

    print(
        "\nTraining patches:",
        len(train_patches)
    )

    print(
        "Validation patches:",
        len(val_patches)
    )

    print(
        "\nDataset ready for training."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()'''

import os
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

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
# CONFIGURATION
# ============================================================

PATCH_SIZE = 256

# 50% overlap
STRIDE = 128

# Patch must have at least 90% valid pixels
MIN_PATCH_VALID_RATIO = 0.90

# Skip entire scenes below 10% valid pixels
MIN_SCENE_VALID_RATIO = 0.10

# About 20% of scenes go to validation
VAL_SCENE_RATIO = 0.20

RANDOM_SEED = 42

# Prevent one scene from dominating the dataset
MAX_PATCHES_PER_SCENE = 400


# ============================================================
# EXPECTED PROCESSED FILES
# ============================================================

RGB_FILENAME = "rgb_normalized.npy"

THERMAL_FILENAME = (
    "thermal_normalized.npy"
)

MASK_FILENAME = (
    "valid_mask.npy"
)


# ============================================================
# CREATE / CLEAR OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories():

    directories = [
        TRAIN_INPUT_DIR,
        TRAIN_TARGET_DIR,
        VAL_INPUT_DIR,
        VAL_TARGET_DIR
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remove old patches only
        for file in directory.glob(
            "*.npy"
        ):

            file.unlink()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def scene_has_required_files(
    scene_dir
):

    required = [
        RGB_FILENAME,
        THERMAL_FILENAME,
        MASK_FILENAME
    ]

    for filename in required:

        file_path = (
            scene_dir
            / filename
        )

        if not file_path.exists():

            return False

    return True


# ============================================================
# LOAD SCENE
# ============================================================

def load_scene(
    scene_dir
):

    rgb_path = (
        scene_dir
        / RGB_FILENAME
    )

    thermal_path = (
        scene_dir
        / THERMAL_FILENAME
    )

    mask_path = (
        scene_dir
        / MASK_FILENAME
    )

    # --------------------------------------------------------
    # MEMORY MAPPING
    # --------------------------------------------------------
    #
    # These files are huge (~hundreds of MB each).
    # mmap_mode="r" prevents loading the entire scene into RAM.
    # --------------------------------------------------------

    rgb = np.load(
        rgb_path,
        mmap_mode="r"
    )

    thermal = np.load(
        thermal_path,
        mmap_mode="r"
    )

    valid_mask = np.load(
        mask_path,
        mmap_mode="r"
    )

    # --------------------------------------------------------
    # SHAPE CHECKS
    # --------------------------------------------------------

    if rgb.ndim != 3:

        raise RuntimeError(
            f"{scene_dir.name}: "
            f"RGB should be H,W,3. "
            f"Got {rgb.shape}"
        )

    if rgb.shape[2] != 3:

        raise RuntimeError(
            f"{scene_dir.name}: "
            f"RGB should have 3 channels. "
            f"Got {rgb.shape}"
        )

    if thermal.ndim != 2:

        raise RuntimeError(
            f"{scene_dir.name}: "
            f"thermal should be H,W. "
            f"Got {thermal.shape}"
        )

    if valid_mask.ndim != 2:

        raise RuntimeError(
            f"{scene_dir.name}: "
            f"mask should be H,W. "
            f"Got {valid_mask.shape}"
        )

    if (
        rgb.shape[:2]
        != thermal.shape
    ):

        raise RuntimeError(
            f"{scene_dir.name}: "
            f"RGB/thermal shape mismatch.\n"
            f"RGB: {rgb.shape}\n"
            f"Thermal: {thermal.shape}"
        )

    if (
        thermal.shape
        != valid_mask.shape
    ):

        raise RuntimeError(
            f"{scene_dir.name}: "
            f"thermal/mask shape mismatch.\n"
            f"Thermal: {thermal.shape}\n"
            f"Mask: {valid_mask.shape}"
        )

    return (
        rgb,
        thermal,
        valid_mask
    )


# ============================================================
# FIND VALID PATCH POSITIONS
# ============================================================

def find_valid_patch_positions(
    valid_mask
):

    height, width = (
        valid_mask.shape
    )

    positions = []

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            mask_patch = valid_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            valid_ratio = float(
                np.mean(
                    mask_patch
                )
            )

            if (
                valid_ratio
                >= MIN_PATCH_VALID_RATIO
            ):

                positions.append(
                    (y, x)
                )

    return positions


# ============================================================
# LIMIT PATCHES PER SCENE
# ============================================================

def limit_patch_positions(
    positions,
    rng
):

    if (
        len(positions)
        <= MAX_PATCHES_PER_SCENE
    ):

        return positions

    selected_indices = rng.choice(
        len(positions),
        size=MAX_PATCHES_PER_SCENE,
        replace=False
    )

    selected_indices = sorted(
        selected_indices.tolist()
    )

    return [
        positions[index]
        for index
        in selected_indices
    ]


# ============================================================
# SAVE PATCHES
# ============================================================

def save_scene_patches(
    scene_dir,
    input_dir,
    target_dir,
    rng
):

    (
        rgb,
        thermal,
        valid_mask
    ) = load_scene(
        scene_dir
    )

    # --------------------------------------------------------
    # Find suitable patch positions
    # --------------------------------------------------------

    positions = (
        find_valid_patch_positions(
            valid_mask
        )
    )

    original_count = len(
        positions
    )

    positions = limit_patch_positions(
        positions,
        rng
    )

    limited_count = len(
        positions
    )

    print(
        f"\n{scene_dir.name}"
    )

    print(
        f"  Valid patch positions: "
        f"{original_count}"
    )

    if (
        original_count
        > MAX_PATCHES_PER_SCENE
    ):

        print(
            f"  Capped to: "
            f"{limited_count}"
        )

    saved = 0

    # --------------------------------------------------------
    # Extract patches
    # --------------------------------------------------------

    for (
        y,
        x
    ) in positions:

        thermal_patch = np.array(
            thermal[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ],
            dtype=np.float32,
            copy=True
        )

        rgb_patch = np.array(
            rgb[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
                :
            ],
            dtype=np.float32,
            copy=True
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if (
            thermal_patch.shape
            != (
                PATCH_SIZE,
                PATCH_SIZE
            )
        ):

            continue

        if (
            rgb_patch.shape
            != (
                PATCH_SIZE,
                PATCH_SIZE,
                3
            )
        ):

            continue

        if not np.isfinite(
            thermal_patch
        ).all():

            continue

        if not np.isfinite(
            rgb_patch
        ).all():

            continue

        # ----------------------------------------------------
        # The preprocessor already normalized these to [0,1].
        # Clip only as numerical safety.
        # ----------------------------------------------------

        thermal_patch = np.clip(
            thermal_patch,
            0.0,
            1.0
        )

        rgb_patch = np.clip(
            rgb_patch,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Same filename for input and target
        # ----------------------------------------------------

        filename = (
            f"{scene_dir.name}"
            f"_y{y:05d}"
            f"_x{x:05d}"
            f".npy"
        )

        np.save(
            input_dir
            / filename,
            thermal_patch
        )

        np.save(
            target_dir
            / filename,
            rgb_patch
        )

        saved += 1

    print(
        f"  Saved patches: "
        f"{saved}"
    )

    return saved


# ============================================================
# FIND USABLE SCENES
# ============================================================

def get_usable_scenes():

    print(
        "\n" + "=" * 70
    )

    print(
        "CHECKING PROCESSED SCENES"
    )

    print(
        "=" * 70
    )

    if not PROCESSED_DIR.exists():

        raise RuntimeError(
            f"Processed directory "
            f"does not exist:\n"
            f"{PROCESSED_DIR}"
        )

    scene_dirs = sorted(
        [
            path
            for path
            in PROCESSED_DIR.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith(".")
            )
        ]
    )

    print(
        f"\nProcessed scene folders: "
        f"{len(scene_dirs)}"
    )

    usable_scenes = []

    skipped_scenes = []

    for scene_dir in scene_dirs:

        print(
            f"\n{scene_dir.name}"
        )

        # ----------------------------------------------------
        # FILE CHECK
        # ----------------------------------------------------

        if not scene_has_required_files(
            scene_dir
        ):

            print(
                "  SKIPPED: missing required "
                "processed files"
            )

            print(
                f"  Expected:"
            )

            print(
                f"    {RGB_FILENAME}"
            )

            print(
                f"    {THERMAL_FILENAME}"
            )

            print(
                f"    {MASK_FILENAME}"
            )

            skipped_scenes.append(
                scene_dir.name
            )

            continue

        # ----------------------------------------------------
        # SCENE VALID RATIO
        # ----------------------------------------------------

        valid_mask = np.load(
            scene_dir
            / MASK_FILENAME,
            mmap_mode="r"
        )

        valid_ratio = float(
            valid_mask.mean()
        )

        print(
            f"  Valid pixels: "
            f"{valid_ratio * 100:.2f}%"
        )

        if (
            valid_ratio
            < MIN_SCENE_VALID_RATIO
        ):

            print(
                "  SKIPPED: scene has "
                "too few valid pixels"
            )

            skipped_scenes.append(
                scene_dir.name
            )

            continue

        print(
            "  ✓ Usable"
        )

        usable_scenes.append(
            scene_dir
        )

    print(
        "\n" + "-" * 70
    )

    print(
        f"Usable scenes: "
        f"{len(usable_scenes)}"
    )

    print(
        f"Skipped scenes: "
        f"{len(skipped_scenes)}"
    )

    if skipped_scenes:

        print(
            "\nSkipped scenes:"
        )

        for name in skipped_scenes:

            print(
                f"  - {name}"
            )

    return usable_scenes


# ============================================================
# SPLIT SCENES
# ============================================================

def split_scenes(
    usable_scenes
):

    if len(
        usable_scenes
    ) < 5:

        raise RuntimeError(
            "Too few usable scenes "
            "for train/validation split."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    shuffled = list(
        usable_scenes
    )

    rng.shuffle(
        shuffled
    )

    val_count = round(
        len(shuffled)
        * VAL_SCENE_RATIO
    )

    # At least 2 validation scenes
    val_count = max(
        2,
        val_count
    )

    # At least 3 training scenes
    val_count = min(
        val_count,
        len(shuffled) - 3
    )

    val_scenes = shuffled[
        :val_count
    ]

    train_scenes = shuffled[
        val_count:
    ]

    return (
        train_scenes,
        val_scenes
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "LANDSAT MULTI-SCENE PATCH GENERATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPatch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride: "
        f"{STRIDE}"
    )

    print(
        f"Minimum patch valid ratio: "
        f"{MIN_PATCH_VALID_RATIO * 100:.0f}%"
    )

    print(
        f"Minimum scene valid ratio: "
        f"{MIN_SCENE_VALID_RATIO * 100:.0f}%"
    )

    print(
        f"Maximum patches per scene: "
        f"{MAX_PATCHES_PER_SCENE}"
    )

    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    prepare_output_directories()

    # ========================================================
    # GET USABLE SCENES
    # ========================================================

    usable_scenes = (
        get_usable_scenes()
    )

    if len(
        usable_scenes
    ) == 0:

        raise RuntimeError(
            "No usable processed scenes found."
        )

    # ========================================================
    # SCENE-LEVEL SPLIT
    # ========================================================

    (
        train_scenes,
        val_scenes
    ) = split_scenes(
        usable_scenes
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SCENE-LEVEL SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTraining scenes: "
        f"{len(train_scenes)}"
    )

    for scene in train_scenes:

        print(
            f"  TRAIN: "
            f"{scene.name}"
        )

    print(
        f"\nValidation scenes: "
        f"{len(val_scenes)}"
    )

    for scene in val_scenes:

        print(
            f"  VAL:   "
            f"{scene.name}"
        )

    # ========================================================
    # RNG
    # ========================================================

    train_rng = np.random.default_rng(
        RANDOM_SEED
    )

    val_rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    # ========================================================
    # TRAINING PATCHES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING TRAINING PATCHES"
    )

    print(
        "=" * 70
    )

    total_train = 0

    train_scene_counts = {}

    for scene_dir in train_scenes:

        count = save_scene_patches(
            scene_dir,
            TRAIN_INPUT_DIR,
            TRAIN_TARGET_DIR,
            train_rng
        )

        train_scene_counts[
            scene_dir.name
        ] = count

        total_train += count

    # ========================================================
    # VALIDATION PATCHES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING VALIDATION PATCHES"
    )

    print(
        "=" * 70
    )

    total_val = 0

    val_scene_counts = {}

    for scene_dir in val_scenes:

        count = save_scene_patches(
            scene_dir,
            VAL_INPUT_DIR,
            VAL_TARGET_DIR,
            val_rng
        )

        val_scene_counts[
            scene_dir.name
        ] = count

        total_val += count

    # ========================================================
    # VERIFY PAIRS
    # ========================================================

    train_inputs = sorted(
        TRAIN_INPUT_DIR.glob(
            "*.npy"
        )
    )

    train_targets = sorted(
        TRAIN_TARGET_DIR.glob(
            "*.npy"
        )
    )

    val_inputs = sorted(
        VAL_INPUT_DIR.glob(
            "*.npy"
        )
    )

    val_targets = sorted(
        VAL_TARGET_DIR.glob(
            "*.npy"
        )
    )

    if (
        len(train_inputs)
        != len(train_targets)
    ):

        raise RuntimeError(
            "Training input/target "
            "count mismatch."
        )

    if (
        len(val_inputs)
        != len(val_targets)
    ):

        raise RuntimeError(
            "Validation input/target "
            "count mismatch."
        )

    # --------------------------------------------------------
    # Verify matching filenames
    # --------------------------------------------------------

    train_input_names = [
        file.name
        for file in train_inputs
    ]

    train_target_names = [
        file.name
        for file in train_targets
    ]

    if (
        train_input_names
        != train_target_names
    ):

        raise RuntimeError(
            "Training input/target "
            "filenames do not match."
        )

    val_input_names = [
        file.name
        for file in val_inputs
    ]

    val_target_names = [
        file.name
        for file in val_targets
    ]

    if (
        val_input_names
        != val_target_names
    ):

        raise RuntimeError(
            "Validation input/target "
            "filenames do not match."
        )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING PATCH COUNTS"
    )

    print(
        "=" * 70
    )

    for (
        scene,
        count
    ) in train_scene_counts.items():

        print(
            f"{scene}: "
            f"{count}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION PATCH COUNTS"
    )

    print(
        "=" * 70
    )

    for (
        scene,
        count
    ) in val_scene_counts.items():

        print(
            f"{scene}: "
            f"{count}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nUsable scenes: "
        f"{len(usable_scenes)}"
    )

    print(
        f"Training scenes: "
        f"{len(train_scenes)}"
    )

    print(
        f"Validation scenes: "
        f"{len(val_scenes)}"
    )

    print(
        f"\nTraining patches: "
        f"{total_train}"
    )

    print(
        f"Validation patches: "
        f"{total_val}"
    )

    print(
        f"Total patches: "
        f"{total_train + total_val}"
    )

    print(
        "\nOutput directories:"
    )

    print(
        f"Train thermal: "
        f"{TRAIN_INPUT_DIR}"
    )

    print(
        f"Train RGB:     "
        f"{TRAIN_TARGET_DIR}"
    )

    print(
        f"Val thermal:   "
        f"{VAL_INPUT_DIR}"
    )

    print(
        f"Val RGB:       "
        f"{VAL_TARGET_DIR}"
    )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    if total_train == 0:

        raise RuntimeError(
            "No training patches generated."
        )

    if total_val == 0:

        raise RuntimeError(
            "No validation patches generated."
        )

    print(
        "\n✓ Patch generation successful."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()