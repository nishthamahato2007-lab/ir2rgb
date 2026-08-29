import torch
import torch.nn as nn


# ============================================================
# U-NET GENERATOR
# Input  : Thermal image (1 channel)
# Output : RGB image (3 channels)
#
# IMPORTANT:
# This architecture matches baseline_v1.pth
# ============================================================


class DownBlock(nn.Module):
    """
    Encoder block:
    Conv -> BatchNorm -> LeakyReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True
    ):
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
                nn.BatchNorm2d(
                    out_channels
                )
            )

        layers.append(
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        self.block = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x
    ):
        return self.block(
            x
        )


# ============================================================
# DECODER BLOCK
# ============================================================

class UpBlock(nn.Module):
    """
    Decoder block:

    ConvTranspose2d
        ->
    BatchNorm
        ->
    ReLU
        ->
    optional Dropout

    This is the architecture used by baseline_v1.pth.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False
    ):
        super().__init__()

        layers = [

            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        ]


        if dropout:

            layers.append(
                nn.Dropout(
                    0.5
                )
            )


        self.block = nn.Sequential(
            *layers
        )


    def forward(
        self,
        x
    ):

        return self.block(
            x
        )


# ============================================================
# U-NET GENERATOR
# ============================================================

class UNetGenerator(nn.Module):
    """
    U-Net Generator for Thermal -> RGB.

    Input:
        [B, 1, 256, 256]

    Output:
        [B, 3, 256, 256]

    Output range:
        [-1, 1]
    """

    def __init__(
        self
    ):
        super().__init__()


        # ====================================================
        # ENCODER
        # ====================================================

        self.down1 = DownBlock(
            1,
            64,
            normalize=False
        )

        self.down2 = DownBlock(
            64,
            128
        )

        self.down3 = DownBlock(
            128,
            256
        )

        self.down4 = DownBlock(
            256,
            512
        )

        self.down5 = DownBlock(
            512,
            512
        )

        self.down6 = DownBlock(
            512,
            512
        )

        self.down7 = DownBlock(
            512,
            512
        )

        self.down8 = DownBlock(
            512,
            512
        )


        # ====================================================
        # DECODER
        # ====================================================

        self.up1 = UpBlock(
            512,
            512,
            dropout=True
        )

        self.up2 = UpBlock(
            1024,
            512,
            dropout=True
        )

        self.up3 = UpBlock(
            1024,
            512,
            dropout=True
        )

        self.up4 = UpBlock(
            1024,
            512
        )

        self.up5 = UpBlock(
            1024,
            256
        )

        self.up6 = UpBlock(
            512,
            128
        )

        self.up7 = UpBlock(
            256,
            64
        )


        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        self.final = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()

        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # ====================================================
        # ENCODER
        # ====================================================

        d1 = self.down1(
            x
        )

        d2 = self.down2(
            d1
        )

        d3 = self.down3(
            d2
        )

        d4 = self.down4(
            d3
        )

        d5 = self.down5(
            d4
        )

        d6 = self.down6(
            d5
        )

        d7 = self.down7(
            d6
        )

        d8 = self.down8(
            d7
        )


        # ====================================================
        # DECODER + SKIP CONNECTIONS
        # ====================================================

        u1 = self.up1(
            d8
        )

        u1 = torch.cat(
            [
                u1,
                d7
            ],
            dim=1
        )


        u2 = self.up2(
            u1
        )

        u2 = torch.cat(
            [
                u2,
                d6
            ],
            dim=1
        )


        u3 = self.up3(
            u2
        )

        u3 = torch.cat(
            [
                u3,
                d5
            ],
            dim=1
        )


        u4 = self.up4(
            u3
        )

        u4 = torch.cat(
            [
                u4,
                d4
            ],
            dim=1
        )


        u5 = self.up5(
            u4
        )

        u5 = torch.cat(
            [
                u5,
                d3
            ],
            dim=1
        )


        u6 = self.up6(
            u5
        )

        u6 = torch.cat(
            [
                u6,
                d2
            ],
            dim=1
        )


        u7 = self.up7(
            u6
        )

        u7 = torch.cat(
            [
                u7,
                d1
            ],
            dim=1
        )


        # ====================================================
        # RGB OUTPUT
        # ====================================================

        output = self.final(
            u7
        )


        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "TESTING U-NET GENERATOR"
    )

    print(
        "=" * 60
    )


    generator = UNetGenerator()


    x = torch.randn(
        2,
        1,
        256,
        256
    )


    print(
        "\nInput shape:"
    )

    print(
        x.shape
    )


    with torch.no_grad():

        output = generator(
            x
        )


    print(
        "\nOutput shape:"
    )

    print(
        output.shape
    )


    print(
        "\nOutput range:"
    )

    print(
        output.min().item(),
        "to",
        output.max().item()
    )


    assert output.shape == (
        2,
        3,
        256,
        256
    )


    print(
        "\n✓ Generator test passed"
    )