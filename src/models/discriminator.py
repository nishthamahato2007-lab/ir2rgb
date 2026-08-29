import torch
import torch.nn as nn


# ============================================================
# PATCHGAN DISCRIMINATOR
# Thermal + RGB -> Real/Fake Patch Map
# ============================================================


class DiscriminatorBlock(nn.Module):
    """
    Single PatchGAN discriminator block.

    Conv2D -> BatchNorm -> LeakyReLU
    """

    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        ]

        if normalize:
            layers.append(
                nn.BatchNorm2d(out_channels)
            )

        layers.append(
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator.

    Input:
        Thermal image : 1 channel
        RGB image     : 3 channels

    Combined input:
        4 channels

    Input shape:
        [B, 4, 256, 256]

    Output:
        Patch-level real/fake prediction map.
    """

    def __init__(self):
        super().__init__()

        # ====================================================
        # DISCRIMINATOR
        # ====================================================

        self.model = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------
            # 256 -> 128

            DiscriminatorBlock(
                4,
                64,
                normalize=False
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------
            # 128 -> 64

            DiscriminatorBlock(
                64,
                128
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------
            # 64 -> 32

            DiscriminatorBlock(
                128,
                256
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------
            # 32 -> 16

            DiscriminatorBlock(
                256,
                512
            ),

            # ------------------------------------------------
            # Final PatchGAN layer
            # ------------------------------------------------

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, thermal, rgb):

        # ====================================================
        # CONCATENATE THERMAL + RGB
        # ====================================================

        x = torch.cat(
            [thermal, rgb],
            dim=1
        )

        # ====================================================
        # PATCHGAN PREDICTION
        # ====================================================

        output = self.model(x)

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING PATCHGAN DISCRIMINATOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Create dummy thermal input
    # --------------------------------------------------------

    thermal = torch.randn(
        4,
        1,
        256,
        256
    )

    # --------------------------------------------------------
    # Create dummy RGB input
    # --------------------------------------------------------

    rgb = torch.randn(
        4,
        3,
        256,
        256
    )

    print("\nThermal shape:")
    print(thermal.shape)

    print("\nRGB shape:")
    print(rgb.shape)

    # --------------------------------------------------------
    # Create discriminator
    # --------------------------------------------------------

    discriminator = PatchGANDiscriminator()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = discriminator(
            thermal,
            rgb
        )

    print("\nDiscriminator output shape:")
    print(output.shape)

    print("\nDiscriminator output range:")

    print(
        output.min().item(),
        "to",
        output.max().item()
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    assert output.shape[0] == 4
    assert output.shape[1] == 1

    print("\n✓ Discriminator test passed")