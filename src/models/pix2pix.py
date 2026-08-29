import torch
import torch.nn as nn

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator


# ============================================================
# PIX2PIX MODEL
#
# Thermal Image -> RGB Image
#
# Generator:
#     Thermal -> Fake RGB
#
# Discriminator:
#     Thermal + RGB -> Real/Fake
# ============================================================


class Pix2Pix(nn.Module):

    def __init__(
        self,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999,
        lambda_l1=100.0,
        device=None
    ):

        super().__init__()

        # ====================================================
        # DEVICE
        # ====================================================

        if device is None:

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        print("\nDevice:")
        print(self.device)

        # ====================================================
        # MODELS
        # ====================================================

        self.generator = UNetGenerator().to(self.device)

        self.discriminator = (
            PatchGANDiscriminator().to(self.device)
        )

        # ====================================================
        # LOSSES
        # ====================================================

        # GAN loss
        #
        # PatchGAN does not have a sigmoid at the output.
        # Therefore BCEWithLogitsLoss is used.

        self.gan_loss = nn.BCEWithLogitsLoss()

        # Reconstruction loss

        self.l1_loss = nn.L1Loss()

        # Weight of reconstruction loss

        self.lambda_l1 = lambda_l1

        # ====================================================
        # OPTIMIZERS
        # ====================================================

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )

        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )

    # ========================================================
    # GENERATOR FORWARD PASS
    # ========================================================

    def generate(self, thermal):

        return self.generator(thermal)

    # ========================================================
    # DISCRIMINATOR FORWARD PASS
    # ========================================================

    def discriminate(
        self,
        thermal,
        rgb
    ):

        return self.discriminator(
            thermal,
            rgb
        )

    # ========================================================
    # GENERATOR LOSS
    # ========================================================

    def generator_loss(
        self,
        thermal,
        real_rgb
    ):

        # Generate fake RGB

        fake_rgb = self.generator(thermal)

        # Ask discriminator whether fake RGB
        # looks real when conditioned on thermal

        fake_prediction = self.discriminator(
            thermal,
            fake_rgb
        )

        # Generator wants discriminator to classify
        # fake RGB as REAL.

        real_labels = torch.ones_like(
            fake_prediction
        )

        adversarial_loss = self.gan_loss(
            fake_prediction,
            real_labels
        )

        # Reconstruction loss

        reconstruction_loss = self.l1_loss(
            fake_rgb,
            real_rgb
        )

        # Total generator loss

        total_loss = (
            adversarial_loss
            + self.lambda_l1 * reconstruction_loss
        )

        return (
            total_loss,
            adversarial_loss,
            reconstruction_loss,
            fake_rgb
        )

    # ========================================================
    # DISCRIMINATOR LOSS
    # ========================================================

    def discriminator_loss(
        self,
        thermal,
        real_rgb,
        fake_rgb
    ):

        # ----------------------------------------------------
        # REAL IMAGE
        # ----------------------------------------------------

        real_prediction = self.discriminator(
            thermal,
            real_rgb
        )

        real_labels = torch.ones_like(
            real_prediction
        )

        real_loss = self.gan_loss(
            real_prediction,
            real_labels
        )

        # ----------------------------------------------------
        # FAKE IMAGE
        # ----------------------------------------------------

        fake_prediction = self.discriminator(
            thermal,
            fake_rgb.detach()
        )

        fake_labels = torch.zeros_like(
            fake_prediction
        )

        fake_loss = self.gan_loss(
            fake_prediction,
            fake_labels
        )

        # ----------------------------------------------------
        # TOTAL DISCRIMINATOR LOSS
        # ----------------------------------------------------

        total_loss = (
            real_loss
            + fake_loss
        ) * 0.5

        return (
            total_loss,
            real_loss,
            fake_loss
        )

    # ========================================================
    # TRAIN DISCRIMINATOR
    # ========================================================

    def train_discriminator(
        self,
        thermal,
        real_rgb,
        fake_rgb
    ):

        self.optimizer_D.zero_grad()

        (
            loss_D,
            real_loss,
            fake_loss
        ) = self.discriminator_loss(
            thermal,
            real_rgb,
            fake_rgb
        )

        loss_D.backward()

        self.optimizer_D.step()

        return {
            "loss_D": loss_D.item(),
            "real_loss": real_loss.item(),
            "fake_loss": fake_loss.item()
        }

    # ========================================================
    # TRAIN GENERATOR
    # ========================================================

    def train_generator(
        self,
        thermal,
        real_rgb
    ):

        self.optimizer_G.zero_grad()

        (
            loss_G,
            adversarial_loss,
            reconstruction_loss,
            fake_rgb
        ) = self.generator_loss(
            thermal,
            real_rgb
        )

        loss_G.backward()

        self.optimizer_G.step()

        return {
            "loss_G": loss_G.item(),
            "gan_loss": adversarial_loss.item(),
            "l1_loss": reconstruction_loss.item(),
            "fake_rgb": fake_rgb.detach()
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING PIX2PIX MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUsing device:")
    print(device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = Pix2Pix(
        learning_rate=0.0002,
        lambda_l1=100.0,
        device=device
    )

    # --------------------------------------------------------
    # Dummy data
    # --------------------------------------------------------

    thermal = torch.randn(
        2,
        1,
        256,
        256,
        device=device
    )

    real_rgb = torch.randn(
        2,
        3,
        256,
        256,
        device=device
    )

    print("\nThermal shape:")
    print(thermal.shape)

    print("\nReal RGB shape:")
    print(real_rgb.shape)

    # --------------------------------------------------------
    # Generate fake RGB
    # --------------------------------------------------------

    with torch.no_grad():

        fake_rgb = model.generate(
            thermal
        )

    print("\nFake RGB shape:")
    print(fake_rgb.shape)

    # --------------------------------------------------------
    # Test discriminator
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model.discriminate(
            thermal,
            fake_rgb
        )

    print("\nDiscriminator prediction shape:")
    print(prediction.shape)

    # --------------------------------------------------------
    # Test losses
    # --------------------------------------------------------

    (
        g_loss,
        gan_loss,
        l1_loss,
        fake_rgb
    ) = model.generator_loss(
        thermal,
        real_rgb
    )

    (
        d_loss,
        real_loss,
        fake_loss
    ) = model.discriminator_loss(
        thermal,
        real_rgb,
        fake_rgb
    )

    print("\nGenerator losses:")
    print("Total:", g_loss.item())
    print("GAN:", gan_loss.item())
    print("L1:", l1_loss.item())

    print("\nDiscriminator losses:")
    print("Total:", d_loss.item())
    print("Real:", real_loss.item())
    print("Fake:", fake_loss.item())

    # --------------------------------------------------------
    # Verify shapes
    # --------------------------------------------------------

    assert fake_rgb.shape == (
        2,
        3,
        256,
        256
    )

    assert prediction.shape[0] == 2
    assert prediction.shape[1] == 1

    print("\n✓ Pix2Pix forward pass passed")
    print("✓ Generator loss passed")
    print("✓ Discriminator loss passed")

    print("\n============================================================")
    print("PIX2PIX MODEL TEST PASSED")
    print("============================================================")