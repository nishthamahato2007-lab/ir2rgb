'''import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    LandsatDataset,
    TRAIN_INPUT_DIR,
    TRAIN_TARGET_DIR,
    VAL_INPUT_DIR,
    VAL_TARGET_DIR
)

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
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from src.data.dataset import LandsatDataset
from src.models.pix2pix import Pix2Pix


# ============================================================
# PATHS
# ============================================================

TRAIN_THERMAL = os.path.join(
    PROJECT_ROOT,
    "data",
    "train",
    "input_thermal"
)

TRAIN_RGB = os.path.join(
    PROJECT_ROOT,
    "data",
    "train",
    "target_rgb"
)

VAL_THERMAL = os.path.join(
    PROJECT_ROOT,
    "data",
    "val",
    "input_thermal"
)

VAL_RGB = os.path.join(
    PROJECT_ROOT,
    "data",
    "val",
    "target_rgb"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)


# Create directories if they don't exist

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 2

NUM_EPOCHS = 20

LEARNING_RATE = 0.0002

LAMBDA_L1 = 100.0

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("IR2RGB PIX2PIX TRAINING")
    print("=" * 60)

    print("\nDevice:")
    print(DEVICE)

    print("\nConfiguration:")
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Lambda L1:", LAMBDA_L1)

    # ========================================================
    # DATASET
    # ========================================================

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    train_dataset = LandsatDataset(
        TRAIN_INPUT_DIR,
        TRAIN_TARGET_DIR
    )

    val_dataset = LandsatDataset(
    VAL_INPUT_DIR,
    VAL_TARGET_DIR
)

    print("\nTraining samples:")
    print(len(train_dataset))

    print("\nValidation samples:")
    print(len(val_dataset))


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\nTraining batches:")
    print(len(train_loader))

    print("\nValidation batches:")
    print(len(val_loader))


    # ========================================================
    # CREATE PIX2PIX MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("CREATING PIX2PIX MODEL")
    print("=" * 60)

    model = Pix2Pix(
        learning_rate=LEARNING_RATE,
        lambda_l1=LAMBDA_L1,
        device=DEVICE
    )

    print("\n✓ Pix2Pix model created")


    # ========================================================
    # BEST VALIDATION LOSS
    # ========================================================

    best_val_loss = float("inf")


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)


    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()


        # ----------------------------------------------------
        # TRAIN MODE
        # ----------------------------------------------------

        model.generator.train()

        model.discriminator.train()


        # ----------------------------------------------------
        # LOSS ACCUMULATORS
        # ----------------------------------------------------

        total_g_loss = 0.0

        total_gan_loss = 0.0

        total_l1_loss = 0.0

        total_d_loss = 0.0

        total_real_loss = 0.0

        total_fake_loss = 0.0


        # ====================================================
        # TRAINING BATCHES
        # ====================================================

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        print("-" * 60)


        for batch_idx, batch in enumerate(
        train_loader):
            thermal, real_rgb = batch

            thermal = thermal.to(DEVICE)
            real_rgb = real_rgb.to(DEVICE)


            # =================================================
            # GENERATOR
            # =================================================

            generator_result = (
                model.train_generator(
                    thermal,
                    real_rgb
                )
            )


            g_loss = generator_result[
                "loss_G"
            ]

            gan_loss = generator_result[
                "gan_loss"
            ]

            l1_loss = generator_result[
                "l1_loss"
            ]

            fake_rgb = generator_result[
                "fake_rgb"
            ]


            # =================================================
            # DISCRIMINATOR
            # =================================================

            discriminator_result = (
                model.train_discriminator(
                    thermal,
                    real_rgb,
                    fake_rgb
                )
            )


            d_loss = discriminator_result[
                "loss_D"
            ]

            real_loss = discriminator_result[
                "real_loss"
            ]

            fake_loss = discriminator_result[
                "fake_loss"
            ]


            # =================================================
            # ACCUMULATE
            # =================================================

            total_g_loss += g_loss

            total_gan_loss += gan_loss

            total_l1_loss += l1_loss

            total_d_loss += d_loss

            total_real_loss += real_loss

            total_fake_loss += fake_loss


            # =================================================
            # PROGRESS
            # =================================================

            if (
                batch_idx == 0
                or (batch_idx + 1) % 20 == 0
                or (batch_idx + 1)
                == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} | "
                    f"G: {g_loss:.4f} | "
                    f"D: {d_loss:.4f}"
                )


        # ====================================================
        # AVERAGE TRAINING LOSSES
        # ====================================================

        num_batches = len(train_loader)


        avg_g_loss = (
            total_g_loss /
            num_batches
        )

        avg_gan_loss = (
            total_gan_loss /
            num_batches
        )

        avg_l1_loss = (
            total_l1_loss /
            num_batches
        )

        avg_d_loss = (
            total_d_loss /
            num_batches
        )

        avg_real_loss = (
            total_real_loss /
            num_batches
        )

        avg_fake_loss = (
            total_fake_loss /
            num_batches
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.generator.eval()

        validation_l1 = 0.0

        validation_batches = 0


        with torch.no_grad():

            for batch in val_loader:

                thermal = batch[
                    "thermal"
                ].to(DEVICE)

                real_rgb = batch[
                    "rgb"
                ].to(DEVICE)


                fake_rgb = model.generate(
                    thermal
                )


                loss = torch.nn.functional.l1_loss(
                    fake_rgb,
                    real_rgb
                )


                validation_l1 += loss.item()

                validation_batches += 1


        avg_val_l1 = (
            validation_l1 /
            validation_batches
        )


        # ====================================================
        # TIME
        # ====================================================

        epoch_time = (
            time.time()
            - epoch_start
        )


        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print("\n" + "=" * 60)
        print(
            f"EPOCH {epoch} SUMMARY"
        )
        print("=" * 60)

        print(
            f"Generator Loss:      "
            f"{avg_g_loss:.6f}"
        )

        print(
            f"GAN Loss:            "
            f"{avg_gan_loss:.6f}"
        )

        print(
            f"L1 Loss:             "
            f"{avg_l1_loss:.6f}"
        )

        print(
            f"Discriminator Loss:  "
            f"{avg_d_loss:.6f}"
        )

        print(
            f"Real Loss:           "
            f"{avg_real_loss:.6f}"
        )

        print(
            f"Fake Loss:           "
            f"{avg_fake_loss:.6f}"
        )

        print(
            f"Validation L1:       "
            f"{avg_val_l1:.6f}"
        )

        print(
            f"Time:                "
            f"{epoch_time:.2f} seconds"
        )


        # ====================================================
        # SAVE LATEST CHECKPOINT
        # ====================================================

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "latest.pth"
        )


        torch.save(
            {
                "epoch": epoch,

                "generator_state_dict":
                    model.generator.state_dict(),

                "discriminator_state_dict":
                    model.discriminator.state_dict(),

                "optimizer_G_state_dict":
                    model.optimizer_G.state_dict(),

                "optimizer_D_state_dict":
                    model.optimizer_D.state_dict(),

                "generator_loss":
                    avg_g_loss,

                "discriminator_loss":
                    avg_d_loss,

                "validation_l1":
                    avg_val_l1
            },
            latest_path
        )


        print(
            "\n✓ Latest checkpoint saved"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if avg_val_l1 < best_val_loss:

            best_val_loss = avg_val_l1


            best_path = os.path.join(
                CHECKPOINT_DIR,
                "best.pth"
            )


            torch.save(
                {
                    "epoch": epoch,

                    "generator_state_dict":
                        model.generator.state_dict(),

                    "discriminator_state_dict":
                        model.discriminator.state_dict(),

                    "optimizer_G_state_dict":
                        model.optimizer_G.state_dict(),

                    "optimizer_D_state_dict":
                        model.optimizer_D.state_dict(),

                    "generator_loss":
                        avg_g_loss,

                    "discriminator_loss":
                        avg_d_loss,

                    "validation_l1":
                        avg_val_l1
                },
                best_path
            )


            print(
                "✓ New BEST model saved!"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nBest validation L1:"
    )

    print(best_val_loss)

    print("\nCheckpoints saved in:")

    print(CHECKPOINT_DIR)

    print("\nFiles:")

    print("  best.pth")
    print("  latest.pth")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()'''

import os
import sys
import time

import torch
import torch.nn.functional as F
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
    TRAIN_INPUT_DIR,
    TRAIN_TARGET_DIR,
    VAL_INPUT_DIR,
    VAL_TARGET_DIR
)

from src.models.pix2pix import Pix2Pix


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 2

NUM_EPOCHS = 20

LEARNING_RATE = 0.0002

LAMBDA_L1 = 100.0

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HELPER: CONVERT LOSS TO PYTHON FLOAT
# ============================================================

def to_float(value):

    if torch.is_tensor(value):

        return float(
            value.detach().item()
        )

    return float(value)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("IR2RGB PIX2PIX TRAINING")
    print("=" * 60)

    print("\nDevice:")
    print(DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nConfiguration:")

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Lambda L1:",
        LAMBDA_L1
    )


    # ========================================================
    # DATASETS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "LOADING DATASET"
    )

    print(
        "=" * 60
    )

    train_dataset = LandsatDataset(
        TRAIN_INPUT_DIR,
        TRAIN_TARGET_DIR
    )

    val_dataset = LandsatDataset(
        VAL_INPUT_DIR,
        VAL_TARGET_DIR
    )

    print(
        "\nTraining samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # drop_last=True is used for training because the final
    # batch may contain only 1 sample.
    #
    # Your U-Net bottleneck becomes 1x1 and BatchNorm cannot
    # train with only one value per channel.
    #
    # Example:
    #
    # [1, 512, 1, 1]
    #
    # causes:
    #
    # Expected more than 1 value per channel...
    #
    # So we discard only the final incomplete training batch.
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


    # --------------------------------------------------------
    # Validation does NOT need drop_last=True because
    # generator is in eval mode and BatchNorm uses stored
    # running statistics.
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CREATING PIX2PIX MODEL"
    )

    print(
        "=" * 60
    )

    model = Pix2Pix(
        learning_rate=LEARNING_RATE,
        lambda_l1=LAMBDA_L1,
        device=DEVICE
    )

    print(
        "\n✓ Pix2Pix model created"
    )


    # ========================================================
    # BEST VALIDATION LOSS
    # ========================================================

    best_val_loss = float(
        "inf"
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 60
    )


    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()


        # ----------------------------------------------------
        # TRAIN MODE
        # ----------------------------------------------------

        model.generator.train()

        model.discriminator.train()


        # ----------------------------------------------------
        # LOSS ACCUMULATORS
        # ----------------------------------------------------

        total_g_loss = 0.0

        total_gan_loss = 0.0

        total_l1_loss = 0.0

        total_d_loss = 0.0

        total_real_loss = 0.0

        total_fake_loss = 0.0


        print(
            f"\nEpoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 60
        )


        # ====================================================
        # TRAINING BATCHES
        # ====================================================

        for batch_idx, batch in enumerate(
            train_loader
        ):


            # ------------------------------------------------
            # DATASET OUTPUT
            # ------------------------------------------------
            #
            # dataset.py returns:
            #
            # {
            #     "thermal": tensor,
            #     "rgb": tensor
            # }
            # ------------------------------------------------

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
            # TRAIN GENERATOR
            # =================================================

            generator_result = (
                model.train_generator(
                    thermal,
                    real_rgb
                )
            )


            g_loss = generator_result[
                "loss_G"
            ]


            gan_loss = generator_result[
                "gan_loss"
            ]


            l1_loss = generator_result[
                "l1_loss"
            ]


            fake_rgb = generator_result[
                "fake_rgb"
            ]


            # =================================================
            # TRAIN DISCRIMINATOR
            # =================================================

            discriminator_result = (
                model.train_discriminator(
                    thermal,
                    real_rgb,
                    fake_rgb
                )
            )


            d_loss = discriminator_result[
                "loss_D"
            ]


            real_loss = discriminator_result[
                "real_loss"
            ]


            fake_loss = discriminator_result[
                "fake_loss"
            ]


            # =================================================
            # CONVERT LOSSES TO FLOATS
            # =================================================

            g_loss_value = to_float(
                g_loss
            )


            gan_loss_value = to_float(
                gan_loss
            )


            l1_loss_value = to_float(
                l1_loss
            )


            d_loss_value = to_float(
                d_loss
            )


            real_loss_value = to_float(
                real_loss
            )


            fake_loss_value = to_float(
                fake_loss
            )


            # =================================================
            # ACCUMULATE
            # =================================================

            total_g_loss += (
                g_loss_value
            )


            total_gan_loss += (
                gan_loss_value
            )


            total_l1_loss += (
                l1_loss_value
            )


            total_d_loss += (
                d_loss_value
            )


            total_real_loss += (
                real_loss_value
            )


            total_fake_loss += (
                fake_loss_value
            )


            # =================================================
            # PROGRESS
            # =================================================

            if (
                batch_idx == 0
                or
                (batch_idx + 1) % 20 == 0
                or
                (batch_idx + 1)
                == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} | "
                    f"G: "
                    f"{g_loss_value:.4f} | "
                    f"D: "
                    f"{d_loss_value:.4f} | "
                    f"L1: "
                    f"{l1_loss_value:.4f}"
                )


        # ====================================================
        # AVERAGE TRAINING LOSSES
        # ====================================================

        num_batches = len(
            train_loader
        )


        if num_batches == 0:

            raise RuntimeError(
                "Training loader contains "
                "zero batches."
            )


        avg_g_loss = (
            total_g_loss
            / num_batches
        )


        avg_gan_loss = (
            total_gan_loss
            / num_batches
        )


        avg_l1_loss = (
            total_l1_loss
            / num_batches
        )


        avg_d_loss = (
            total_d_loss
            / num_batches
        )


        avg_real_loss = (
            total_real_loss
            / num_batches
        )


        avg_fake_loss = (
            total_fake_loss
            / num_batches
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.generator.eval()


        # ----------------------------------------------------
        # Instead of averaging each batch's mean L1 equally,
        # accumulate absolute error over all pixels.
        #
        # This is more accurate when the final validation batch
        # contains fewer samples.
        # ----------------------------------------------------

        validation_absolute_error = 0.0

        validation_elements = 0


        with torch.no_grad():

            for batch in val_loader:


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


                fake_rgb = model.generate(
                    thermal
                )


                # ------------------------------------------------
                # Sum absolute error over every RGB pixel
                # ------------------------------------------------

                loss_sum = F.l1_loss(
                    fake_rgb,
                    real_rgb,
                    reduction="sum"
                )


                validation_absolute_error += (
                    to_float(
                        loss_sum
                    )
                )


                validation_elements += (
                    real_rgb.numel()
                )


        if validation_elements == 0:

            raise RuntimeError(
                "Validation loader contains "
                "zero elements."
            )


        avg_val_l1 = (
            validation_absolute_error
            / validation_elements
        )


        # ====================================================
        # EPOCH TIME
        # ====================================================

        epoch_time = (
            time.time()
            - epoch_start
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            f"EPOCH {epoch} SUMMARY"
        )

        print(
            "=" * 60
        )


        print(
            f"Generator Loss:      "
            f"{avg_g_loss:.6f}"
        )


        print(
            f"GAN Loss:            "
            f"{avg_gan_loss:.6f}"
        )


        print(
            f"L1 Loss:             "
            f"{avg_l1_loss:.6f}"
        )


        print(
            f"Discriminator Loss:  "
            f"{avg_d_loss:.6f}"
        )


        print(
            f"Real Loss:           "
            f"{avg_real_loss:.6f}"
        )


        print(
            f"Fake Loss:           "
            f"{avg_fake_loss:.6f}"
        )


        print(
            f"Validation L1:       "
            f"{avg_val_l1:.6f}"
        )


        print(
            f"Time:                "
            f"{epoch_time:.2f} seconds"
        )


        # ====================================================
        # SAVE LATEST CHECKPOINT
        # ====================================================

        latest_path = os.path.join(
            CHECKPOINT_DIR,
            "latest.pth"
        )


        torch.save(
            {
                "epoch":
                    epoch,

                "generator_state_dict":
                    model.generator.state_dict(),

                "discriminator_state_dict":
                    model.discriminator.state_dict(),

                "optimizer_G_state_dict":
                    model.optimizer_G.state_dict(),

                "optimizer_D_state_dict":
                    model.optimizer_D.state_dict(),

                "generator_loss":
                    avg_g_loss,

                "gan_loss":
                    avg_gan_loss,

                "l1_loss":
                    avg_l1_loss,

                "discriminator_loss":
                    avg_d_loss,

                "real_loss":
                    avg_real_loss,

                "fake_loss":
                    avg_fake_loss,

                "validation_l1":
                    avg_val_l1
            },
            latest_path
        )


        print(
            "\n✓ Latest checkpoint saved"
        )


        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if avg_val_l1 < best_val_loss:


            best_val_loss = (
                avg_val_l1
            )


            best_path = os.path.join(
                CHECKPOINT_DIR,
                "best.pth"
            )


            torch.save(
                {
                    "epoch":
                        epoch,

                    "generator_state_dict":
                        model.generator.state_dict(),

                    "discriminator_state_dict":
                        model.discriminator.state_dict(),

                    "optimizer_G_state_dict":
                        model.optimizer_G.state_dict(),

                    "optimizer_D_state_dict":
                        model.optimizer_D.state_dict(),

                    "generator_loss":
                        avg_g_loss,

                    "gan_loss":
                        avg_gan_loss,

                    "l1_loss":
                        avg_l1_loss,

                    "discriminator_loss":
                        avg_d_loss,

                    "real_loss":
                        avg_real_loss,

                    "fake_loss":
                        avg_fake_loss,

                    "validation_l1":
                        avg_val_l1
                },
                best_path
            )


            print(
                "✓ New BEST model saved!"
            )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )


    print(
        "\nBest validation L1:"
    )

    print(
        best_val_loss
    )


    print(
        "\nCheckpoints saved in:"
    )

    print(
        CHECKPOINT_DIR
    )


    print(
        "\nFiles:"
    )

    print(
        "  best.pth"
    )

    print(
        "  latest.pth"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()