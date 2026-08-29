'''import os
import sys

import numpy as np
import rasterio


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)


# ============================================================
# DIRECTORIES
# ============================================================

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# FIND FILE
# ============================================================

def find_file(
    scene_dir,
    keyword
):

    for filename in os.listdir(scene_dir):

        if keyword.lower() in filename.lower():

            return os.path.join(
                scene_dir,
                filename
            )

    raise FileNotFoundError(
        f"Could not find file containing "
        f"'{keyword}' in {scene_dir}"
    )


# ============================================================
# READ RASTER
# ============================================================

def read_raster(path):

    with rasterio.open(path) as src:

        data = src.read(1).astype(
            np.float32
        )

        profile = src.profile

        transform = src.transform

        crs = src.crs

    return (
        data,
        profile,
        transform,
        crs
    )


# ============================================================
# LANDSAT REFLECTANCE SCALING
# ============================================================

def scale_reflectance(dn):

    return (
        dn * 0.0000275
        - 0.2
    )


# ============================================================
# LANDSAT TEMPERATURE SCALING
# ============================================================

def scale_temperature(dn):

    return (
        dn * 0.00341802
        + 149.0
    )


# ============================================================
# CLOUD MASK
# ============================================================

def create_cloud_mask(qa):

    fill = (
        qa & (1 << 0)
    ) != 0

    dilated_cloud = (
        qa & (1 << 1)
    ) != 0

    cirrus = (
        qa & (1 << 2)
    ) != 0

    cloud = (
        qa & (1 << 3)
    ) != 0

    cloud_shadow = (
        qa & (1 << 4)
    ) != 0

    bad_pixels = (
        fill
        | dilated_cloud
        | cirrus
        | cloud
        | cloud_shadow
    )

    return ~bad_pixels


# ============================================================
# PROCESS ONE SCENE
# ============================================================

def process_scene(scene_id):

    print("\n" + "=" * 60)
    print("PROCESSING SCENE")
    print("=" * 60)

    print("\nScene:")
    print(scene_id)

    scene_raw_dir = os.path.join(
        RAW_DIR,
        scene_id
    )

    scene_processed_dir = os.path.join(
        PROCESSED_DIR,
        scene_id
    )

    # --------------------------------------------------------
    # Check raw directory
    # --------------------------------------------------------

    if not os.path.isdir(
        scene_raw_dir
    ):

        raise FileNotFoundError(
            f"Raw scene directory not found:\n"
            f"{scene_raw_dir}"
        )

    os.makedirs(
        scene_processed_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find bands
    # --------------------------------------------------------

    print(
        "\nFinding Landsat bands..."
    )

    b2_file = find_file(
        scene_raw_dir,
        "SR_B2"
    )

    b3_file = find_file(
        scene_raw_dir,
        "SR_B3"
    )

    b4_file = find_file(
        scene_raw_dir,
        "SR_B4"
    )

    b10_file = find_file(
        scene_raw_dir,
        "ST_B10"
    )

    qa_file = find_file(
        scene_raw_dir,
        "QA_PIXEL"
    )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    print(
        "\nReading Landsat bands..."
    )

    blue_dn, profile, transform, crs = read_raster(
        b2_file
    )

    green_dn, _, _, _ = read_raster(
        b3_file
    )

    red_dn, _, _, _ = read_raster(
        b4_file
    )

    thermal_dn, _, _, _ = read_raster(
        b10_file
    )

    qa, _, _, _ = read_raster(
        qa_file
    )

    print("✓ B2 loaded")
    print("✓ B3 loaded")
    print("✓ B4 loaded")
    print("✓ B10 loaded")
    print("✓ QA_PIXEL loaded")

    # --------------------------------------------------------
    # Scale RGB
    # --------------------------------------------------------

    print(
        "\nScaling RGB bands..."
    )

    blue = scale_reflectance(
        blue_dn
    )

    green = scale_reflectance(
        green_dn
    )

    red = scale_reflectance(
        red_dn
    )

    # --------------------------------------------------------
    # Scale thermal
    # --------------------------------------------------------

    print(
        "Scaling thermal band..."
    )

    thermal_kelvin = scale_temperature(
        thermal_dn
    )

    thermal_celsius = (
        thermal_kelvin
        - 273.15
    )

    # --------------------------------------------------------
    # Cloud mask
    # --------------------------------------------------------

    print(
        "Creating cloud mask..."
    )

    valid_mask = create_cloud_mask(
        qa.astype(np.uint16)
    )

    print(
        "Valid pixels:",
        np.sum(valid_mask),
        "/",
        valid_mask.size
    )

    # --------------------------------------------------------
    # Apply mask
    # --------------------------------------------------------

    blue[~valid_mask] = np.nan
    green[~valid_mask] = np.nan
    red[~valid_mask] = np.nan
    thermal_celsius[~valid_mask] = np.nan

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving processed data..."
    )

    np.save(
        os.path.join(
            scene_processed_dir,
            "blue.npy"
        ),
        blue
    )

    np.save(
        os.path.join(
            scene_processed_dir,
            "green.npy"
        ),
        green
    )

    np.save(
        os.path.join(
            scene_processed_dir,
            "red.npy"
        ),
        red
    )

    np.save(
        os.path.join(
            scene_processed_dir,
            "thermal_celsius.npy"
        ),
        thermal_celsius
    )

    np.save(
        os.path.join(
            scene_processed_dir,
            "valid_mask.npy"
        ),
        valid_mask
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nProcessed data saved to:"
    )

    print(
        scene_processed_dir
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "\nBlue:",
        np.nanmin(blue),
        "to",
        np.nanmax(blue)
    )

    print(
        "Green:",
        np.nanmin(green),
        "to",
        np.nanmax(green)
    )

    print(
        "Red:",
        np.nanmin(red),
        "to",
        np.nanmax(red)
    )

    print(
        "Thermal °C:",
        np.nanmin(thermal_celsius),
        "to",
        np.nanmax(thermal_celsius)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LANDSAT MULTI-SCENE PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Find all downloaded scenes
    # --------------------------------------------------------

    scene_ids = []

    for name in os.listdir(RAW_DIR):

        path = os.path.join(
            RAW_DIR,
            name
        )

        if os.path.isdir(path):

            scene_ids.append(name)

    scene_ids.sort()

    print(
        "\nScenes found:",
        len(scene_ids)
    )

    if len(scene_ids) == 0:

        raise RuntimeError(
            "No Landsat scenes found in data/raw"
        )

    # --------------------------------------------------------
    # Process each scene
    # --------------------------------------------------------

    successful = 0

    for scene_id in scene_ids:

        try:

            process_scene(
                scene_id
            )

            successful += 1

        except Exception as e:

            print(
                "\nERROR processing:",
                scene_id
            )

            print(
                str(e)
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "\nSuccessfully processed:",
        successful,
        "/",
        len(scene_ids)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()'''

import os
from pathlib import Path

import numpy as np
import rasterio


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

RAW_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Landsat Collection 2 Level-2 Surface Reflectance scaling
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

# Landsat Collection 2 Level-2 Surface Temperature scaling
ST_SCALE = 0.00341802
ST_OFFSET = 149.0

# ------------------------------------------------------------
# RGB ENHANCEMENT
# ------------------------------------------------------------
#
# Landsat surface reflectance usually occupies a relatively
# small part of the theoretical 0-1 range.
#
# Stretching 0.00 -> 0.30 to 0 -> 1 makes the imagery
# significantly more visible and useful as a Pix2Pix target.
# ------------------------------------------------------------

RGB_MIN = 0.00
RGB_MAX = 0.40
GAMMA = 1.00


# ------------------------------------------------------------
# THERMAL NORMALIZATION
# ------------------------------------------------------------

THERMAL_MIN_C = -20.0
THERMAL_MAX_C = 60.0


# ============================================================
# FIND TIFF
# ============================================================

def find_tif(
    scene_dir,
    keyword
):

    matches = [
        path
        for path in scene_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in [".tif", ".tiff"]
            and keyword.lower()
            in path.name.lower()
        )
    ]

    if len(matches) == 0:

        raise FileNotFoundError(
            f"Could not find {keyword} "
            f"in {scene_dir}"
        )

    if len(matches) > 1:

        print(
            f"WARNING: multiple files found "
            f"for {keyword}"
        )

        for path in matches:
            print(
                "  ",
                path.name
            )

        print(
            "Using:",
            matches[0].name
        )

    return matches[0]


# ============================================================
# READ RASTER
# ============================================================

def read_raster(
    path
):

    with rasterio.open(
        path
    ) as src:

        data = src.read(
            1
        )

    return data


# ============================================================
# SCALE SURFACE REFLECTANCE
# ============================================================

def scale_reflectance(
    dn
):

    dn = dn.astype(
        np.float32
    )

    reflectance = (
        dn * SR_SCALE
        + SR_OFFSET
    )

    return reflectance.astype(
        np.float32
    )


# ============================================================
# SCALE TEMPERATURE
# ============================================================

def scale_temperature(
    dn
):

    dn = dn.astype(
        np.float32
    )

    kelvin = (
        dn * ST_SCALE
        + ST_OFFSET
    )

    celsius = (
        kelvin
        - 273.15
    )

    return celsius.astype(
        np.float32
    )


# ============================================================
# CREATE VALID PIXEL MASK
# ============================================================

def create_valid_mask(
    qa
):

    qa = qa.astype(
        np.uint16
    )

    # Landsat QA_PIXEL bit definitions

    fill = (
        qa & (1 << 0)
    ) != 0

    dilated_cloud = (
        qa & (1 << 1)
    ) != 0

    cirrus = (
        qa & (1 << 2)
    ) != 0

    cloud = (
        qa & (1 << 3)
    ) != 0

    cloud_shadow = (
        qa & (1 << 4)
    ) != 0

    snow = (
        qa & (1 << 5)
    ) != 0

    invalid = (
        fill
        | dilated_cloud
        | cirrus
        | cloud
        | cloud_shadow
        | snow
    )

    valid = ~invalid

    return valid


# ============================================================
# ENHANCE + NORMALIZE RGB
# ============================================================

def normalize_rgb(
    red,
    green,
    blue,
    valid_mask
):

    # --------------------------------------------------------
    # STACK TRUE-COLOR RGB
    # --------------------------------------------------------

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # FIXED REFLECTANCE STRETCH
    # --------------------------------------------------------
    #
    # 0.00 reflectance -> 0
    # 0.30 reflectance -> 1
    #
    # Fixed across ALL scenes.
    # This is preferable to per-scene percentile stretching
    # for supervised training because the target mapping stays
    # consistent.
    # --------------------------------------------------------

    rgb = (
        rgb - RGB_MIN
    ) / (
        RGB_MAX - RGB_MIN
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # GAMMA ENHANCEMENT
    # --------------------------------------------------------
    #
    # Gamma = 0.8 gives a mild brightness boost.
    # --------------------------------------------------------

    rgb = np.power(
        rgb,
        GAMMA
    )

    # --------------------------------------------------------
    # INVALID PIXELS
    # --------------------------------------------------------

    rgb[
        ~valid_mask
    ] = 0.0

    # --------------------------------------------------------
    # NUMERICAL SAFETY
    # --------------------------------------------------------

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
    thermal_celsius,
    valid_mask
):

    thermal = (
        thermal_celsius
        - THERMAL_MIN_C
    ) / (
        THERMAL_MAX_C
        - THERMAL_MIN_C
    )

    thermal = np.clip(
        thermal,
        0.0,
        1.0
    )

    thermal[
        ~valid_mask
    ] = 0.0

    thermal = np.nan_to_num(
        thermal,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return thermal.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE SCENE
# ============================================================

def process_scene(
    scene_dir
):

    scene_id = (
        scene_dir.name
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PROCESSING SCENE"
    )

    print(
        "=" * 60
    )

    print(
        "\nScene:",
        scene_id
    )

    # ========================================================
    # FIND FILES
    # ========================================================

    print(
        "\nFinding bands..."
    )

    b2_path = find_tif(
        scene_dir,
        "SR_B2"
    )

    b3_path = find_tif(
        scene_dir,
        "SR_B3"
    )

    b4_path = find_tif(
        scene_dir,
        "SR_B4"
    )

    thermal_path = find_tif(
        scene_dir,
        "ST_B10"
    )

    qa_path = find_tif(
        scene_dir,
        "QA_PIXEL"
    )

    # ========================================================
    # READ BANDS
    # ========================================================

    print(
        "\nReading bands..."
    )

    blue_dn = read_raster(
        b2_path
    )

    print(
        "✓ B2"
    )

    green_dn = read_raster(
        b3_path
    )

    print(
        "✓ B3"
    )

    red_dn = read_raster(
        b4_path
    )

    print(
        "✓ B4"
    )

    thermal_dn = read_raster(
        thermal_path
    )

    print(
        "✓ ST_B10"
    )

    qa = read_raster(
        qa_path
    )

    print(
        "✓ QA_PIXEL"
    )

    # ========================================================
    # SHAPE CHECK
    # ========================================================

    shapes = [
        blue_dn.shape,
        green_dn.shape,
        red_dn.shape,
        thermal_dn.shape,
        qa.shape
    ]

    if len(
        set(shapes)
    ) != 1:

        raise RuntimeError(
            f"Raster shape mismatch "
            f"in {scene_id}:\n"
            f"{shapes}"
        )

    # ========================================================
    # SCALE
    # ========================================================

    blue = scale_reflectance(
        blue_dn
    )

    green = scale_reflectance(
        green_dn
    )

    red = scale_reflectance(
        red_dn
    )

    thermal_celsius = (
        scale_temperature(
            thermal_dn
        )
    )

    # ========================================================
    # CREATE QA MASK
    # ========================================================

    valid_mask = (
        create_valid_mask(
            qa
        )
    )

    # ========================================================
    # REMOVE INVALID SENSOR VALUES
    # ========================================================

    valid_mask &= np.isfinite(
        blue
    )

    valid_mask &= np.isfinite(
        green
    )

    valid_mask &= np.isfinite(
        red
    )

    valid_mask &= np.isfinite(
        thermal_celsius
    )

    # --------------------------------------------------------
    # Remove obvious invalid reflectance values
    # --------------------------------------------------------

    valid_mask &= (
        blue > -0.1
    )

    valid_mask &= (
        green > -0.1
    )

    valid_mask &= (
        red > -0.1
    )

    # --------------------------------------------------------
    # Remove obviously impossible thermal values
    # --------------------------------------------------------

    valid_mask &= (
        thermal_celsius
        > -80.0
    )

    valid_mask &= (
        thermal_celsius
        < 100.0
    )

    # ========================================================
    # VALID PIXEL STATISTICS
    # ========================================================

    valid_pixels = int(
        valid_mask.sum()
    )

    total_pixels = int(
        valid_mask.size
    )

    valid_percentage = (
        valid_pixels
        / total_pixels
        * 100.0
    )

    print(
        f"\nValid pixels: "
        f"{valid_pixels} / "
        f"{total_pixels}"
    )

    print(
        f"Valid percentage: "
        f"{valid_percentage:.2f}%"
    )

    # ========================================================
    # MASK RAW/SCALED DATA
    # ========================================================

    blue = blue.astype(
        np.float32
    )

    green = green.astype(
        np.float32
    )

    red = red.astype(
        np.float32
    )

    thermal_celsius = (
        thermal_celsius.astype(
            np.float32
        )
    )

    blue[
        ~valid_mask
    ] = np.nan

    green[
        ~valid_mask
    ] = np.nan

    red[
        ~valid_mask
    ] = np.nan

    thermal_celsius[
        ~valid_mask
    ] = np.nan

    # ========================================================
    # RGB NORMALIZATION / ENHANCEMENT
    # ========================================================

    print(
        "\nEnhancing RGB..."
    )

    rgb_normalized = (
        normalize_rgb(
            red,
            green,
            blue,
            valid_mask
        )
    )

    print(
        "✓ RGB enhanced"
    )

    # ========================================================
    # THERMAL NORMALIZATION
    # ========================================================

    print(
        "\nNormalizing thermal..."
    )

    thermal_normalized = (
        normalize_thermal(
            thermal_celsius,
            valid_mask
        )
    )

    print(
        "✓ Thermal normalized"
    )

    # ========================================================
    # DEBUG STATISTICS
    # ========================================================

    if valid_pixels > 0:

        valid_rgb = (
            rgb_normalized[
                valid_mask
            ]
        )

        valid_thermal = (
            thermal_normalized[
                valid_mask
            ]
        )

        print(
            "\nNormalized RGB statistics:"
        )

        print(
            f"  Min:  "
            f"{valid_rgb.min():.4f}"
        )

        print(
            f"  Max:  "
            f"{valid_rgb.max():.4f}"
        )

        print(
            f"  Mean: "
            f"{valid_rgb.mean():.4f}"
        )

        print(
            "\nNormalized thermal statistics:"
        )

        print(
            f"  Min:  "
            f"{valid_thermal.min():.4f}"
        )

        print(
            f"  Max:  "
            f"{valid_thermal.max():.4f}"
        )

        print(
            f"  Mean: "
            f"{valid_thermal.mean():.4f}"
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        PROCESSED_DIR
        / scene_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    print(
        "\nSaving..."
    )

    np.save(
        output_dir
        / "blue.npy",
        blue
    )

    np.save(
        output_dir
        / "green.npy",
        green
    )

    np.save(
        output_dir
        / "red.npy",
        red
    )

    np.save(
        output_dir
        / "thermal_celsius.npy",
        thermal_celsius
    )

    np.save(
        output_dir
        / "rgb_normalized.npy",
        rgb_normalized
    )

    np.save(
        output_dir
        / "thermal_normalized.npy",
        thermal_normalized
    )

    np.save(
        output_dir
        / "valid_mask.npy",
        valid_mask.astype(
            np.bool_
        )
    )

    print(
        f"\n✓ Processed: "
        f"{scene_id}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "LANDSAT MULTI-SCENE PREPROCESSING"
    )

    print(
        "=" * 60
    )

    print(
        "\nRGB enhancement:"
    )

    print(
        f"  Reflectance range: "
        f"{RGB_MIN:.2f} → {RGB_MAX:.2f}"
    )

    print(
        f"  Gamma: "
        f"{GAMMA:.2f}"
    )

    print(
        "\nThermal range:"
    )

    print(
        f"  {THERMAL_MIN_C:.1f}°C "
        f"→ {THERMAL_MAX_C:.1f}°C"
    )

    # ========================================================
    # FIND SCENES
    # ========================================================

    if not RAW_DIR.exists():

        raise RuntimeError(
            f"Raw directory does not exist:\n"
            f"{RAW_DIR}"
        )

    scenes = sorted(
        [
            path
            for path in RAW_DIR.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith(".")
                and "_L2SP_" in path.name
            )
        ]
    )

    print(
        f"\nL2SP scenes found: "
        f"{len(scenes)}"
    )

    if len(
        scenes
    ) == 0:

        raise RuntimeError(
            "No Landsat L2SP scenes found."
        )

    # ========================================================
    # PROCESS
    # ========================================================

    successful = 0

    failed = []

    for scene_dir in scenes:

        try:

            process_scene(
                scene_dir
            )

            successful += 1

        except KeyboardInterrupt:

            print(
                "\n\nPreprocessing interrupted."
            )

            raise

        except Exception as e:

            print(
                "\n✗ FAILED:"
            )

            print(
                scene_dir.name
            )

            print(
                "Error:",
                e
            )

            failed.append(
                scene_dir.name
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nSuccessfully processed: "
        f"{successful} / "
        f"{len(scenes)}"
    )

    if failed:

        print(
            "\nFailed scenes:"
        )

        for scene in failed:

            print(
                f"  - {scene}"
            )

    else:

        print(
            "\n✓ All scenes processed successfully."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()