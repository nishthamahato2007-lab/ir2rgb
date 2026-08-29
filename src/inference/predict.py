import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from PIL import Image
from rasterio.windows import Window


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# MODEL IMPORT
# ============================================================

from src.models.pix2pix import Pix2Pix


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "baseline_v1.pth"
)

PATCH_SIZE = 256
PATCH_SEARCH_STRIDE = 128

LEARNING_RATE = 0.0002
LAMBDA_L1 = 100.0


# ============================================================
# LANDSAT COLLECTION 2 LEVEL-2 ST_B10
# ============================================================

ST_SCALE = 0.00341802
ST_OFFSET = 149.0


# ============================================================
# LANDSAT COLLECTION 2 SURFACE REFLECTANCE
# ============================================================

SR_SCALE = 0.0000275
SR_OFFSET = -0.2


# ============================================================
# RGB PREPROCESSING
# MUST MATCH TRAINING
# ============================================================

RGB_MIN = 0.02
RGB_MAX = 0.35
RGB_GAMMA = 1.0


# ============================================================
# THERMAL NORMALIZATION
# ============================================================

THERMAL_MIN_C = -20.0
THERMAL_MAX_C = 60.0


# ============================================================
# PREVIEW
# ============================================================

PREVIEW_MAX_SIZE = 900


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Inference device:",
    DEVICE
)


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global _model

    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print(
        "\nLoading IRIS model..."
    )

    model = Pix2Pix(
        learning_rate=LEARNING_RATE,
        lambda_l1=LAMBDA_L1,
        device=DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and
        "generator" in checkpoint
    ):
        generator_state = checkpoint[
            "generator"
        ]

    elif (
        isinstance(checkpoint, dict)
        and
        "generator_state_dict" in checkpoint
    ):
        generator_state = checkpoint[
            "generator_state_dict"
        ]

    else:
        generator_state = checkpoint

    model.generator.load_state_dict(
        generator_state
    )

    model.generator.to(
        DEVICE
    )

    model.generator.eval()

    _model = model

    print(
        "✓ Model loaded"
    )

    if (
        isinstance(checkpoint, dict)
        and
        "epoch" in checkpoint
    ):
        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if (
        isinstance(checkpoint, dict)
        and
        "val_l1" in checkpoint
    ):
        print(
            "Saved validation L1:",
            checkpoint["val_l1"]
        )

    return _model


# ============================================================
# READ FULL RASTER
# ============================================================

def read_raster(
    raster_path
):

    raster_path = Path(
        raster_path
    )

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    with rasterio.open(
        raster_path
    ) as src:

        data = src.read(
            1
        )

        return {
            "data": data,
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "bounds": src.bounds
        }


# ============================================================
# READ 256x256 WINDOW
# ============================================================

def read_raster_window(
    raster_path,
    x,
    y,
    size
):

    raster_path = Path(
        raster_path
    )

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    with rasterio.open(
        raster_path
    ) as src:

        window = Window(
            x,
            y,
            size,
            size
        )

        data = src.read(
            1,
            window=window
        )

        return {
            "data": data,
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "bounds": src.bounds
        }


# ============================================================
# VERIFY BAND ALIGNMENT
# ============================================================

def verify_alignment(
    reference_info,
    other_info
):

    if (
        reference_info["width"]
        !=
        other_info["width"]
        or
        reference_info["height"]
        !=
        other_info["height"]
    ):
        raise ValueError(
            "Uploaded Landsat bands do not "
            "have matching dimensions."
        )

    if (
        reference_info["crs"]
        !=
        other_info["crs"]
    ):
        raise ValueError(
            "Uploaded Landsat bands do not "
            "have matching CRS."
        )

    if (
        reference_info["transform"]
        !=
        other_info["transform"]
    ):
        raise ValueError(
            "Uploaded Landsat bands are "
            "not spatially aligned."
        )


# ============================================================
# B10 VALID MASK
# ============================================================

def create_b10_valid_mask(
    b10_dn,
    nodata=None
):

    valid = np.isfinite(
        b10_dn
    )

    valid &= (
        b10_dn > 0
    )

    if nodata is not None:
        valid &= (
            b10_dn != nodata
        )

    return valid


# ============================================================
# SURFACE REFLECTANCE VALID MASK
# ============================================================

def create_sr_valid_mask(
    data,
    nodata=None
):

    valid = np.isfinite(
        data
    )

    valid &= (
        data > 0
    )

    if nodata is not None:
        valid &= (
            data != nodata
        )

    return valid


# ============================================================
# QA_PIXEL MASK
# ============================================================

def create_qa_valid_mask(
    qa_array
):

    qa = qa_array.astype(
        np.uint16
    )

    fill = (
        qa
        &
        (1 << 0)
    ) != 0

    dilated_cloud = (
        qa
        &
        (1 << 1)
    ) != 0

    cirrus = (
        qa
        &
        (1 << 2)
    ) != 0

    cloud = (
        qa
        &
        (1 << 3)
    ) != 0

    cloud_shadow = (
        qa
        &
        (1 << 4)
    ) != 0

    snow = (
        qa
        &
        (1 << 5)
    ) != 0

    invalid = (
        fill
        |
        dilated_cloud
        |
        cirrus
        |
        cloud
        |
        cloud_shadow
        |
        snow
    )

    return ~invalid


# ============================================================
# B10 DN -> CELSIUS
# ============================================================

def dn_to_celsius(
    b10_dn
):

    kelvin = (
        b10_dn.astype(
            np.float32
        )
        *
        ST_SCALE
        +
        ST_OFFSET
    )

    return (
        kelvin
        -
        273.15
    )


# ============================================================
# SR DN -> REFLECTANCE
# ============================================================

def dn_to_reflectance(
    dn
):

    return (
        dn.astype(
            np.float32
        )
        *
        SR_SCALE
        +
        SR_OFFSET
    )


# ============================================================
# RGB NORMALIZATION
# ============================================================

def normalize_rgb_band(
    reflectance
):

    normalized = (
        reflectance
        -
        RGB_MIN
    ) / (
        RGB_MAX
        -
        RGB_MIN
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    if RGB_GAMMA != 1.0:
        normalized = np.power(
            normalized,
            1.0 / RGB_GAMMA
        )

    return normalized.astype(
        np.float32
    )


# ============================================================
# THERMAL NORMALIZATION
# ============================================================

def normalize_thermal(
    thermal_celsius,
    valid_mask
):

    normalized = (
        thermal_celsius
        -
        THERMAL_MIN_C
    ) / (
        THERMAL_MAX_C
        -
        THERMAL_MIN_C
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    normalized = normalized.astype(
        np.float32
    )

    normalized[
        ~valid_mask
    ] = 0.0

    return normalized


# ============================================================
# THERMAL PREVIEW
# ============================================================

def thermal_preview_image(
    thermal_array,
    valid_mask
):

    thermal_array = thermal_array.astype(
        np.float32
    )

    valid_values = thermal_array[
        valid_mask
    ]

    preview = np.zeros_like(
        thermal_array,
        dtype=np.float32
    )

    if valid_values.size > 0:

        low = np.percentile(
            valid_values,
            2
        )

        high = np.percentile(
            valid_values,
            98
        )

        if high > low:

            preview = (
                thermal_array
                -
                low
            ) / (
                high
                -
                low
            )

            preview = np.clip(
                preview,
                0.0,
                1.0
            )

        else:
            preview = thermal_array.copy()

    preview[
        ~valid_mask
    ] = 0.0

    preview_uint8 = (
        preview
        *
        255.0
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        preview_uint8,
        mode="L"
    )

    width, height = image.size

    longest_side = max(
        width,
        height
    )

    if longest_side > PREVIEW_MAX_SIZE:

        scale = (
            PREVIEW_MAX_SIZE
            /
            longest_side
        )

        new_width = max(
            1,
            int(
                width * scale
            )
        )

        new_height = max(
            1,
            int(
                height * scale
            )
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.BILINEAR
        )

    return image


# ============================================================
# FIND CLEANEST 256x256 PATCH
# ============================================================

def find_best_patch(
    valid_mask
):

    height, width = valid_mask.shape

    if (
        height < PATCH_SIZE
        or
        width < PATCH_SIZE
    ):
        raise ValueError(
            f"Scene is smaller than "
            f"{PATCH_SIZE}x{PATCH_SIZE}."
        )

    best_ratio = -1.0
    best_y = 0
    best_x = 0

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        PATCH_SEARCH_STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            PATCH_SEARCH_STRIDE
        ):

            patch_mask = valid_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            ratio = float(
                patch_mask.mean()
            )

            if ratio > best_ratio:

                best_ratio = ratio
                best_y = y
                best_x = x

            if ratio >= 0.9999:
                return (
                    y,
                    x,
                    ratio
                )

    if best_ratio <= 0.0:
        raise ValueError(
            "No valid thermal patch could be found."
        )

    return (
        best_y,
        best_x,
        best_ratio
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_patch(
    thermal_patch
):

    model = load_model()

    if thermal_patch.shape != (
        PATCH_SIZE,
        PATCH_SIZE
    ):
        raise ValueError(
            "Thermal patch must be "
            f"{PATCH_SIZE}x{PATCH_SIZE}."
        )

    model_input = (
        thermal_patch
        *
        2.0
        -
        1.0
    )

    tensor = torch.from_numpy(
        model_input.astype(
            np.float32
        )
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    with torch.no_grad():

        generated = model.generate(
            tensor
        )

    generated = (
        generated
        .detach()
        .cpu()
        .squeeze(0)
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    generated = (
        generated + 1.0
    ) / 2.0

    generated = np.clip(
        generated,
        0.0,
        1.0
    )

    return generated.astype(
        np.float32
    )


# ============================================================
# RGB -> PIL
# ============================================================

def rgb_to_image(
    rgb_array
):

    rgb_uint8 = (
        np.clip(
            rgb_array,
            0.0,
            1.0
        )
        *
        255.0
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        rgb_uint8,
        mode="RGB"
    )


# ============================================================
# BUILD TRUE RGB TARGET
# B4 = RED
# B3 = GREEN
# B2 = BLUE
# ============================================================

def build_ground_truth_rgb(
    b10_info,
    b2_path,
    b3_path,
    b4_path,
    x,
    y,
    patch_mask
):

    b2_info = read_raster_window(
        b2_path,
        x,
        y,
        PATCH_SIZE
    )

    b3_info = read_raster_window(
        b3_path,
        x,
        y,
        PATCH_SIZE
    )

    b4_info = read_raster_window(
        b4_path,
        x,
        y,
        PATCH_SIZE
    )

    verify_alignment(
        b10_info,
        b2_info
    )

    verify_alignment(
        b10_info,
        b3_info
    )

    verify_alignment(
        b10_info,
        b4_info
    )

    b2 = b2_info["data"]
    b3 = b3_info["data"]
    b4 = b4_info["data"]

    if (
        b2.shape != (PATCH_SIZE, PATCH_SIZE)
        or
        b3.shape != (PATCH_SIZE, PATCH_SIZE)
        or
        b4.shape != (PATCH_SIZE, PATCH_SIZE)
    ):
        raise ValueError(
            "Visible-band validation patch "
            "is not 256x256."
        )

    rgb_valid = patch_mask.copy()

    rgb_valid &= create_sr_valid_mask(
        b2,
        b2_info["nodata"]
    )

    rgb_valid &= create_sr_valid_mask(
        b3,
        b3_info["nodata"]
    )

    rgb_valid &= create_sr_valid_mask(
        b4,
        b4_info["nodata"]
    )

    blue = normalize_rgb_band(
        dn_to_reflectance(
            b2
        )
    )

    green = normalize_rgb_band(
        dn_to_reflectance(
            b3
        )
    )

    red = normalize_rgb_band(
        dn_to_reflectance(
            b4
        )
    )

    target_rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    target_rgb[
        ~rgb_valid
    ] = 0.0

    return (
        target_rgb.astype(
            np.float32
        ),
        rgb_valid
    )


# ============================================================
# L1
# ============================================================

def calculate_l1(
    generated,
    target,
    valid_mask
):

    if valid_mask.sum() == 0:
        return None

    mask3 = np.repeat(
        valid_mask[
            ...,
            None
        ],
        3,
        axis=2
    )

    difference = np.abs(
        generated
        -
        target
    )

    return float(
        difference[
            mask3
        ].mean()
    )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    generated,
    target,
    valid_mask
):

    if valid_mask.sum() == 0:
        return None

    mask3 = np.repeat(
        valid_mask[
            ...,
            None
        ],
        3,
        axis=2
    )

    squared_error = (
        generated
        -
        target
    ) ** 2

    mse = float(
        squared_error[
            mask3
        ].mean()
    )

    if mse <= 1e-12:
        return 100.0

    psnr = (
        10.0
        *
        np.log10(
            1.0 / mse
        )
    )

    return float(
        psnr
    )


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    generated,
    target,
    valid_mask
):

    if valid_mask.sum() == 0:
        return None

    generated_copy = generated.copy()
    target_copy = target.copy()

    generated_copy[
        ~valid_mask
    ] = 0.0

    target_copy[
        ~valid_mask
    ] = 0.0

    generated_tensor = (
        torch.from_numpy(
            generated_copy.transpose(
                2,
                0,
                1
            )
        )
        .unsqueeze(0)
        .float()
    )

    target_tensor = (
        torch.from_numpy(
            target_copy.transpose(
                2,
                0,
                1
            )
        )
        .unsqueeze(0)
        .float()
    )

    window_size = 11
    sigma = 1.5

    coords = torch.arange(
        window_size,
        dtype=torch.float32
    )

    coords -= (
        window_size - 1
    ) / 2.0

    gaussian = torch.exp(
        -(
            coords ** 2
        )
        /
        (
            2.0
            *
            sigma ** 2
        )
    )

    gaussian /= gaussian.sum()

    kernel_2d = (
        gaussian[:, None]
        *
        gaussian[None, :]
    )

    kernel = (
        kernel_2d
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(
            3,
            1,
            1,
            1
        )
    )

    padding = (
        window_size // 2
    )

    mu_x = F.conv2d(
        generated_tensor,
        kernel,
        padding=padding,
        groups=3
    )

    mu_y = F.conv2d(
        target_tensor,
        kernel,
        padding=padding,
        groups=3
    )

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2

    mu_xy = (
        mu_x
        *
        mu_y
    )

    sigma_x_sq = (
        F.conv2d(
            generated_tensor ** 2,
            kernel,
            padding=padding,
            groups=3
        )
        -
        mu_x_sq
    )

    sigma_y_sq = (
        F.conv2d(
            target_tensor ** 2,
            kernel,
            padding=padding,
            groups=3
        )
        -
        mu_y_sq
    )

    sigma_xy = (
        F.conv2d(
            generated_tensor
            *
            target_tensor,
            kernel,
            padding=padding,
            groups=3
        )
        -
        mu_xy
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (
        (
            2.0
            *
            mu_xy
            +
            c1
        )
        *
        (
            2.0
            *
            sigma_xy
            +
            c2
        )
    )

    denominator = (
        (
            mu_x_sq
            +
            mu_y_sq
            +
            c1
        )
        *
        (
            sigma_x_sq
            +
            sigma_y_sq
            +
            c2
        )
    )

    ssim_map = (
        numerator
        /
        (
            denominator
            +
            1e-12
        )
    )

    mask_tensor = torch.from_numpy(
        valid_mask.astype(
            np.float32
        )
    ).unsqueeze(
        0
    ).unsqueeze(
        0
    )

    mask_tensor = mask_tensor.repeat(
        1,
        3,
        1,
        1
    )

    valid_sum = mask_tensor.sum()

    if valid_sum.item() <= 0:
        return None

    score = (
        (
            ssim_map
            *
            mask_tensor
        ).sum()
        /
        valid_sum
    )

    score = float(
        score.item()
    )

    return max(
        -1.0,
        min(
            1.0,
            score
        )
    )


# ============================================================
# VALIDATION METRICS
# ============================================================

def calculate_validation_metrics(
    generated_rgb,
    target_rgb,
    valid_mask
):

    valid_percentage = float(
        valid_mask.mean()
        *
        100.0
    )

    if valid_mask.sum() == 0:

        return {
            "available": False,
            "reason": (
                "No valid RGB pixels were "
                "available for comparison."
            ),
            "valid_percentage": 0.0
        }

    l1 = calculate_l1(
        generated_rgb,
        target_rgb,
        valid_mask
    )

    psnr = calculate_psnr(
        generated_rgb,
        target_rgb,
        valid_mask
    )

    ssim = calculate_ssim(
        generated_rgb,
        target_rgb,
        valid_mask
    )

    return {
        "available": True,

        "l1": (
            round(
                l1,
                4
            )
            if l1 is not None
            else None
        ),

        "psnr_db": (
            round(
                psnr,
                2
            )
            if psnr is not None
            else None
        ),

        "ssim": (
            round(
                ssim,
                4
            )
            if ssim is not None
            else None
        ),

        "valid_percentage": round(
            valid_percentage,
            2
        )
    }


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence_score(
    validation,
    scene_valid_percentage,
    patch_valid_percentage,
    cloud_mask_used
):

    # ========================================================
    # CASE 1:
    # GROUND-TRUTH VALIDATION AVAILABLE
    # ========================================================

    if (
        validation
        and
        validation.get(
            "available",
            False
        )
    ):

        l1 = validation.get(
            "l1"
        )

        psnr = validation.get(
            "psnr_db"
        )

        ssim = validation.get(
            "ssim"
        )

        valid_percentage = validation.get(
            "valid_percentage",
            0.0
        )


        # ----------------------------------------------------
        # L1 SCORE
        #
        # Lower is better.
        #
        # 0.00 -> 1.0
        # 0.30 -> 0.0
        # ----------------------------------------------------

        if l1 is not None:

            l1_score = max(
                0.0,
                min(
                    1.0,
                    1.0
                    -
                    (
                        l1
                        /
                        0.30
                    )
                )
            )

        else:

            l1_score = 0.0


        # ----------------------------------------------------
        # PSNR SCORE
        #
        # <= 10 dB -> 0
        # >= 25 dB -> 1
        # ----------------------------------------------------

        if psnr is not None:

            psnr_score = max(
                0.0,
                min(
                    1.0,
                    (
                        psnr
                        -
                        10.0
                    )
                    /
                    15.0
                )
            )

        else:

            psnr_score = 0.0


        # ----------------------------------------------------
        # SSIM SCORE
        #
        # Clamp to 0..1.
        # ----------------------------------------------------

        if ssim is not None:

            ssim_score = max(
                0.0,
                min(
                    1.0,
                    ssim
                )
            )

        else:

            ssim_score = 0.0


        # ----------------------------------------------------
        # VALID PIXEL SCORE
        # ----------------------------------------------------

        valid_score = max(
            0.0,
            min(
                1.0,
                valid_percentage
                /
                100.0
            )
        )


        # ----------------------------------------------------
        # RECALIBRATED WEIGHTS
        #
        # L1 gets more weight because it matches the model's
        # training objective more directly.
        #
        # SSIM is still important but is reduced because
        # thermal-to-RGB reconstruction is underdetermined.
        # ----------------------------------------------------

        confidence = (
            0.35
            *
            l1_score

            +
            0.25
            *
            psnr_score

            +
            0.25
            *
            ssim_score

            +
            0.15
            *
            valid_score
        )


        confidence_percentage = (
            confidence
            *
            100.0
        )


        # ----------------------------------------------------
        # RECALIBRATED LEVELS
        # ----------------------------------------------------

        if confidence_percentage >= 75:

            level = "High"

        elif confidence_percentage >= 50:

            level = "Moderate"

        elif confidence_percentage >= 35:

            level = "Low"

        else:

            level = "Very Low"


        return {

            "score":
                round(
                    confidence_percentage,
                    2
                ),

            "level":
                level,

            "type":
                "Ground-truth validated",

            "description":
                (
                    "Confidence is a heuristic reconstruction "
                    "quality score derived from L1, PSNR, SSIM "
                    "and valid-pixel coverage. It is not a "
                    "probabilistic model confidence."
                )
        }


    # ========================================================
    # CASE 2:
    # NO GROUND TRUTH
    #
    # INPUT QUALITY ONLY
    # ========================================================

    scene_score = max(
        0.0,
        min(
            1.0,
            scene_valid_percentage
            /
            100.0
        )
    )


    patch_score = max(
        0.0,
        min(
            1.0,
            patch_valid_percentage
            /
            100.0
        )
    )


    qa_score = (
        1.0
        if cloud_mask_used
        else 0.65
    )


    confidence = (
        0.55
        *
        patch_score

        +
        0.30
        *
        scene_score

        +
        0.15
        *
        qa_score
    )


    confidence_percentage = (
        confidence
        *
        100.0
    )


    if confidence_percentage >= 85:

        level = "High"

    elif confidence_percentage >= 70:

        level = "Moderate"

    else:

        level = "Low"


    return {

        "score":
            round(
                confidence_percentage,
                2
            ),

        "level":
            level,

        "type":
            "Input-quality estimate",

        "description":
            (
                "Ground-truth RGB was not supplied. "
                "This score reflects input quality and "
                "preprocessing reliability only, not "
                "verified reconstruction accuracy."
            )
    }


# ============================================================
# SCENE DIAGNOSTICS
# ============================================================

def generate_scene_diagnostics(
    thermal_celsius,
    valid_mask,
    thermal_patch_celsius,
    patch_mask
):

    scene_values = thermal_celsius[
        valid_mask
    ]

    if scene_values.size == 0:

        return {
            "scene_character": "Unknown",
            "thermal_condition": "Unknown",
            "thermal_std_c": None,
            "hot_surface_percentage": None,
            "cool_surface_percentage": None,
            "invalid_percentage": None,
            "patch_mean_temperature_c": None,
            "patch_temperature_std_c": None,
            "patch_character": "Unknown",
            "interpretation": (
                "Insufficient valid thermal "
                "data for scene diagnostics."
            )
        }

    scene_mean = float(
        np.mean(
            scene_values
        )
    )

    scene_std = float(
        np.std(
            scene_values
        )
    )

    hot_percentage = float(
        np.mean(
            scene_values >= 40.0
        )
        *
        100.0
    )

    cool_percentage = float(
        np.mean(
            scene_values <= 20.0
        )
        *
        100.0
    )

    invalid_percentage = float(
        (
            1.0
            -
            valid_mask.mean()
        )
        *
        100.0
    )

    if scene_mean < 10:
        thermal_condition = "Very Cold"

    elif scene_mean < 20:
        thermal_condition = "Cool"

    elif scene_mean < 30:
        thermal_condition = "Moderate"

    elif scene_mean < 40:
        thermal_condition = "Warm"

    else:
        thermal_condition = "Hot"

    if scene_std < 3:
        variability = (
            "thermally uniform"
        )

    elif scene_std < 7:
        variability = (
            "moderately varied"
        )

    else:
        variability = (
            "thermally heterogeneous"
        )

    scene_character = (
        f"{thermal_condition}, "
        f"{variability} surface"
    )

    patch_values = thermal_patch_celsius[
        patch_mask
    ]

    if patch_values.size > 0:

        patch_mean = float(
            np.mean(
                patch_values
            )
        )

        patch_std = float(
            np.std(
                patch_values
            )
        )

        if patch_std < 3:
            patch_variability = (
                "thermally uniform"
            )

        elif patch_std < 7:
            patch_variability = (
                "moderately varied"
            )

        else:
            patch_variability = (
                "highly varied"
            )

        patch_character = (
            f"{patch_variability} region "
            f"with mean temperature "
            f"{patch_mean:.2f} °C"
        )

    else:

        patch_mean = None
        patch_std = None

        patch_character = (
            "No valid thermal pixels "
            "in selected patch."
        )

    if (
        scene_mean >= 40
        and
        scene_std >= 7
    ):

        interpretation = (
            "The scene is dominated by "
            "high-temperature surfaces with strong "
            "thermal variation. This pattern is "
            "consistent with warm terrestrial surfaces "
            "such as dry exposed soil, sparsely "
            "vegetated terrain, or built-up areas. "
            "Thermal imagery alone cannot uniquely "
            "identify the exact land-cover type."
        )

    elif scene_mean >= 40:

        interpretation = (
            "The scene contains predominantly hot "
            "surface temperatures with relatively "
            "consistent thermal conditions. This may "
            "correspond to dry terrain, bare surfaces, "
            "sparse vegetation, or developed areas."
        )

    elif (
        scene_mean >= 30
        and
        scene_std >= 5
    ):

        interpretation = (
            "The scene contains generally warm "
            "surfaces with noticeable temperature "
            "variation. This suggests a mixture of "
            "different surface materials or land "
            "conditions."
        )

    elif cool_percentage >= 40:

        interpretation = (
            "A large portion of the valid scene has "
            "relatively low surface temperatures. "
            "This may be associated with cooler "
            "surfaces such as vegetation, water, "
            "shaded terrain, or cooler environmental "
            "conditions."
        )

    elif scene_std < 3:

        interpretation = (
            "The scene is relatively thermally "
            "uniform, with small temperature "
            "differences across valid pixels."
        )

    else:

        interpretation = (
            "The scene contains a mixture of surface "
            "temperatures and is thermally "
            "heterogeneous. Multiple surface types "
            "or environmental conditions may be "
            "present. Additional spectral information "
            "is required for reliable land-cover "
            "classification."
        )

    return {
        "scene_character":
            scene_character,

        "thermal_condition":
            thermal_condition,

        "thermal_std_c":
            round(
                scene_std,
                2
            ),

        "hot_surface_percentage":
            round(
                hot_percentage,
                2
            ),

        "cool_surface_percentage":
            round(
                cool_percentage,
                2
            ),

        "invalid_percentage":
            round(
                invalid_percentage,
                2
            ),

        "patch_mean_temperature_c":
            (
                round(
                    patch_mean,
                    2
                )
                if patch_mean is not None
                else None
            ),

        "patch_temperature_std_c":
            (
                round(
                    patch_std,
                    2
                )
                if patch_std is not None
                else None
            ),

        "patch_character":
            patch_character,

        "interpretation":
            interpretation
    }


# ============================================================
# MAIN LANDSAT INFERENCE
# ============================================================

def predict_landsat(
    b10_path,
    qa_path=None,
    b2_path=None,
    b3_path=None,
    b4_path=None
):

    b10_info = read_raster(
        b10_path
    )

    b10_dn = b10_info[
        "data"
    ]

    valid_mask = create_b10_valid_mask(
        b10_dn,
        b10_info["nodata"]
    )

    cloud_mask_used = False

    # ========================================================
    # QA
    # ========================================================

    if qa_path is not None:

        qa_info = read_raster(
            qa_path
        )

        verify_alignment(
            b10_info,
            qa_info
        )

        qa_valid = create_qa_valid_mask(
            qa_info["data"]
        )

        valid_mask &= qa_valid

        cloud_mask_used = True

    # ========================================================
    # TEMPERATURE
    # ========================================================

    thermal_celsius = dn_to_celsius(
        b10_dn
    )

    valid_mask &= np.isfinite(
        thermal_celsius
    )

    valid_mask &= (
        thermal_celsius > -80.0
    )

    valid_mask &= (
        thermal_celsius < 100.0
    )

    if valid_mask.sum() == 0:
        raise ValueError(
            "No valid thermal pixels "
            "remain after masking."
        )

    thermal_normalized = normalize_thermal(
        thermal_celsius,
        valid_mask
    )

    full_preview = thermal_preview_image(
        thermal_normalized,
        valid_mask
    )

    (
        y,
        x,
        patch_valid_ratio
    ) = find_best_patch(
        valid_mask
    )

    thermal_patch = thermal_normalized[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE
    ]

    patch_mask = valid_mask[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE
    ]

    thermal_patch_celsius = thermal_celsius[
        y:y + PATCH_SIZE,
        x:x + PATCH_SIZE
    ]

    # ========================================================
    # MODEL
    # ========================================================

    generated_rgb = predict_patch(
        thermal_patch
    )

    thermal_patch_preview = thermal_preview_image(
        thermal_patch,
        patch_mask
    )

    generated_image = rgb_to_image(
        generated_rgb
    )

    # ========================================================
    # TEMPERATURE STATISTICS
    # ========================================================

    valid_temperature_values = thermal_celsius[
        valid_mask
    ]

    min_temp = float(
        valid_temperature_values.min()
    )

    max_temp = float(
        valid_temperature_values.max()
    )

    mean_temp = float(
        valid_temperature_values.mean()
    )

    scene_valid_percentage = round(
        float(
            valid_mask.mean()
        )
        *
        100.0,
        2
    )

    patch_valid_percentage = round(
        patch_valid_ratio
        *
        100.0,
        2
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics = generate_scene_diagnostics(
        thermal_celsius=
            thermal_celsius,

        valid_mask=
            valid_mask,

        thermal_patch_celsius=
            thermal_patch_celsius,

        patch_mask=
            patch_mask
    )

    # ========================================================
    # OPTIONAL VALIDATION
    # ========================================================

    validation_requested = (
        b2_path is not None
        and
        b3_path is not None
        and
        b4_path is not None
    )

    if validation_requested:

        print(
            "Running ground-truth validation..."
        )

        (
            target_rgb,
            validation_mask
        ) = build_ground_truth_rgb(
            b10_info=
                b10_info,

            b2_path=
                b2_path,

            b3_path=
                b3_path,

            b4_path=
                b4_path,

            x=
                x,

            y=
                y,

            patch_mask=
                patch_mask
        )

        validation = calculate_validation_metrics(
            generated_rgb,
            target_rgb,
            validation_mask
        )

        print(
            "✓ Ground-truth validation complete"
        )

    else:

        validation = {
            "available": False,
            "reason": (
                "Matching B2, B3 and B4 "
                "bands were not supplied."
            )
        }

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = calculate_confidence_score(
        validation=
            validation,

        scene_valid_percentage=
            scene_valid_percentage,

        patch_valid_percentage=
            patch_valid_percentage,

        cloud_mask_used=
            cloud_mask_used
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = {
        "scene_width":
            int(
                b10_info["width"]
            ),

        "scene_height":
            int(
                b10_info["height"]
            ),

        "cloud_mask_used":
            cloud_mask_used,

        "scene_valid_percentage":
            scene_valid_percentage,

        "selected_patch_x":
            int(x),

        "selected_patch_y":
            int(y),

        "patch_size":
            PATCH_SIZE,

        "selected_patch_valid_percentage":
            patch_valid_percentage,

        "minimum_temperature_c":
            round(
                min_temp,
                2
            ),

        "maximum_temperature_c":
            round(
                max_temp,
                2
            ),

        "mean_temperature_c":
            round(
                mean_temp,
                2
            ),

        "diagnostics":
            diagnostics,

        "validation":
            validation,

        "confidence":
            confidence
    }

    return (
        full_preview,
        thermal_patch_preview,
        generated_image,
        metrics
    )