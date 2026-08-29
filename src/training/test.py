import os
import sys
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORTS
# ============================================================

from src.data.dataset import (
    LandsatDataset,
    VAL_INPUT_DIR,
    VAL_TARGET_DIR
)

from src.models.pix2pix import Pix2Pix


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best.pth"
)

TEST_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "test"
)

os.makedirs(
    TEST_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 2

NUM_WORKERS = 0

NUM_VISUAL_SAMPLES = 20

# Thermal display contrast stretch
THERMAL_PERCENTILE_LOW = 2

THERMAL_PERCENTILE_HIGH = 98


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    mse = F.mse_loss(
        prediction,
        target,
        reduction="mean"
    ).item()

    if mse <= 1e-12:

        return float(
            "inf"
        )

    return 10.0 * math.log10(
        1.0 / mse
    )


# ============================================================
# CONVERT [-1, 1] -> [0, 1]
# ============================================================

def denormalize(
    tensor
):

    tensor = (
        tensor + 1.0
    ) / 2.0

    return torch.clamp(
        tensor,
        0.0,
        1.0
    )


# ============================================================
# THERMAL DISPLAY RANGE
# ============================================================

def get_thermal_display_range(
    thermal
):

    # Ignore non-finite values
    valid_values = thermal[
        np.isfinite(
            thermal
        )
    ]

    if valid_values.size == 0:

        return 0.0, 1.0


    vmin = np.percentile(
        valid_values,
        THERMAL_PERCENTILE_LOW
    )

    vmax = np.percentile(
        valid_values,
        THERMAL_PERCENTILE_HIGH
    )


    # Safety in case the image is almost completely uniform
    if vmax <= vmin:

        vmin = float(
            valid_values.min()
        )

        vmax = float(
            valid_values.max()
        )


    if vmax <= vmin:

        vmin = 0.0
        vmax = 1.0


    return (
        float(vmin),
        float(vmax)
    )


# ============================================================
# SAVE COMPARISON IMAGE
# ============================================================

def save_comparison(
    thermal,
    generated,
    real,
    index,
    sample_l1,
    sample_psnr
):

    # --------------------------------------------------------
    # TENSOR -> NUMPY
    # --------------------------------------------------------

    thermal = (
        thermal
        .detach()
        .cpu()
        .squeeze(0)
        .numpy()
    )

    generated = (
        generated
        .detach()
        .cpu()
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    real = (
        real
        .detach()
        .cpu()
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    thermal = np.clip(
        thermal,
        0.0,
        1.0
    )

    generated = np.clip(
        generated,
        0.0,
        1.0
    )

    real = np.clip(
        real,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # THERMAL VISUAL CONTRAST
    # --------------------------------------------------------

    vmin, vmax = (
        get_thermal_display_range(
            thermal
        )
    )


    # ========================================================
    # FIGURE
    # ========================================================

    plt.figure(
        figsize=(13, 4)
    )


    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        thermal,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax
    )

    plt.title(
        "Thermal Input"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # GENERATED
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        generated
    )

    plt.title(
        f"Generated RGB\n"
        f"L1: {sample_l1:.4f} | "
        f"PSNR: {sample_psnr:.2f} dB"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # REAL RGB
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        real
    )

    plt.title(
        "Real RGB Target"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # ========================================================
    # SAVE
    # ========================================================

    output_path = os.path.join(
        TEST_OUTPUT_DIR,
        f"comparison_{index:04d}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IR2RGB PIX2PIX TESTING")
    print("=" * 60)


    # ========================================================
    # DEVICE
    # ========================================================

    print(
        "\nDevice:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # CHECK CHECKPOINT
    # ========================================================

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


    # ========================================================
    # DATASET
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING VALIDATION DATASET"
    )

    print(
        "=" * 60
    )


    val_dataset = LandsatDataset(
        VAL_INPUT_DIR,
        VAL_TARGET_DIR
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )


    print(
        "\nValidation samples:",
        len(val_dataset)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )


    if len(
        val_dataset
    ) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING MODEL"
    )

    print(
        "=" * 60
    )


    model = Pix2Pix(
        learning_rate=0.0002,
        lambda_l1=100.0,
        device=DEVICE
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    model.generator.load_state_dict(
        checkpoint[
            "generator_state_dict"
        ]
    )


    model.generator.eval()


    print(
        "\n✓ Best generator loaded"
    )


    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )


    if "validation_l1" in checkpoint:

        print(
            "Saved validation L1:",
            checkpoint[
                "validation_l1"
            ]
        )


    if "l1_loss" in checkpoint:

        print(
            "Saved training L1:",
            checkpoint[
                "l1_loss"
            ]
        )


    if "gan_loss" in checkpoint:

        print(
            "Saved GAN loss:",
            checkpoint[
                "gan_loss"
            ]
        )


    # ========================================================
    # TEST LOOP
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "RUNNING TEST"
    )

    print(
        "=" * 60
    )


    # Exact L1 accumulation
    total_absolute_error = 0.0

    total_elements = 0


    # PSNR accumulation
    total_psnr = 0.0

    finite_psnr_samples = 0


    total_samples = 0

    saved_visuals = 0


    with torch.no_grad():

        for batch_idx, batch in enumerate(
            val_loader
        ):


            # =================================================
            # MOVE TO DEVICE
            # =================================================

            thermal = batch[
                "thermal"
            ].to(
                DEVICE,
                non_blocking=True
            )


            real_rgb = batch[
                "rgb"
            ].to(
                DEVICE,
                non_blocking=True
            )


            # =================================================
            # GENERATE RGB
            # =================================================

            fake_rgb = model.generate(
                thermal
            )


            # =================================================
            # EXACT L1 IN [-1, 1]
            # =================================================

            batch_l1_sum = F.l1_loss(
                fake_rgb,
                real_rgb,
                reduction="sum"
            )


            total_absolute_error += (
                batch_l1_sum.item()
            )


            total_elements += (
                real_rgb.numel()
            )


            # =================================================
            # CONVERT TO [0,1]
            # =================================================

            fake_01 = denormalize(
                fake_rgb
            )

            real_01 = denormalize(
                real_rgb
            )

            thermal_01 = denormalize(
                thermal
            )


            batch_size = (
                thermal.shape[0]
            )


            # =================================================
            # PER-SAMPLE METRICS
            # =================================================

            for i in range(
                batch_size
            ):


                sample_l1 = F.l1_loss(
                    fake_rgb[i],
                    real_rgb[i],
                    reduction="mean"
                ).item()


                sample_psnr = calculate_psnr(
                    fake_01[i],
                    real_01[i]
                )


                if math.isfinite(
                    sample_psnr
                ):

                    total_psnr += (
                        sample_psnr
                    )

                    finite_psnr_samples += 1


                total_samples += 1


                # =================================================
                # SAVE VISUALS
                # =================================================

                if (
                    saved_visuals
                    < NUM_VISUAL_SAMPLES
                ):

                    save_comparison(
                        thermal_01[i],
                        fake_01[i],
                        real_01[i],
                        saved_visuals,
                        sample_l1,
                        sample_psnr
                    )

                    saved_visuals += 1


            # =================================================
            # PROGRESS
            # =================================================

            if (
                batch_idx == 0
                or
                (batch_idx + 1) % 50 == 0
                or
                (batch_idx + 1)
                == len(val_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(val_loader)}"
                )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    if total_samples == 0:

        raise RuntimeError(
            "No validation samples processed."
        )


    if total_elements == 0:

        raise RuntimeError(
            "No validation pixels processed."
        )


    avg_l1 = (
        total_absolute_error
        / total_elements
    )


    if finite_psnr_samples > 0:

        avg_psnr = (
            total_psnr
            / finite_psnr_samples
        )

    else:

        avg_psnr = float(
            "inf"
        )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 60
    )


    print(
        f"\nSamples tested: "
        f"{total_samples}"
    )


    print(
        f"Average L1: "
        f"{avg_l1:.6f}"
    )


    if math.isfinite(
        avg_psnr
    ):

        print(
            f"Average PSNR: "
            f"{avg_psnr:.2f} dB"
        )

    else:

        print(
            "Average PSNR: inf"
        )


    print(
        f"\nVisual samples saved: "
        f"{saved_visuals}"
    )


    print(
        "\nThermal display:"
    )

    print(
        f"  Colormap: inferno"
    )

    print(
        f"  Contrast stretch: "
        f"{THERMAL_PERCENTILE_LOW}th → "
        f"{THERMAL_PERCENTILE_HIGH}th percentile"
    )


    print(
        "\nOutput directory:"
    )

    print(
        TEST_OUTPUT_DIR
    )


    print(
        "\n✓ Testing complete"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()  