'''import os
import requests

import pystac_client
import planetary_computer


# ============================================================
# SCENES TO DOWNLOAD
# ============================================================
#
# Add as many Landsat scenes as you want here.
#
# Example:
#
# SCENE_IDS = [
#     "LC09_L2SP_107080_20251215_02_T1",
#     "LC09_L2SP_108080_20251222_02_T1",
#     "LC09_L2SP_107081_20260101_02_T1",
# ]
#
# Start with around 5 scenes.
#

SCENE_IDS = [

    "LC09_L2SP_107080_20251215_02_T1",
    "LC09_L2SR_107120_20251231_20260102_02_T2"

    # Add more scenes here
    # "LC09_L2SP_108080_20251222_02_T1",
    # "LC09_L2SP_107081_20260101_02_T1",
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)


# ============================================================
# BANDS WE NEED
# ============================================================

BANDS = [
    "blue",
    "green",
    "red",
    "lwir11",
    "qa_pixel"
]


# ============================================================
# CONNECT TO PLANETARY COMPUTER
# ============================================================

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)


# ============================================================
# DOWNLOAD ONE SCENE
# ============================================================

def download_scene(scene_id):

    print("\n" + "=" * 60)
    print("DOWNLOADING SCENE")
    print("=" * 60)

    print("\nScene:")
    print(scene_id)

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = catalog.search(
        collections=["landsat-c2-l2"],
        ids=[scene_id]
    )

    items = search.item_collection()

    print("\nScenes found:", len(items))

    if len(items) == 0:

        print(
            "\nERROR: Scene not found."
        )

        print(
            "Check the SCENE_ID:"
        )

        print(scene_id)

        return False

    item = items[0]

    print("\nScene found!")

    print(
        "Date:",
        item.datetime
    )

    # --------------------------------------------------------
    # Scene-specific directory
    # --------------------------------------------------------

    scene_dir = os.path.join(
        RAW_DIR,
        scene_id
    )

    os.makedirs(
        scene_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download bands
    # --------------------------------------------------------

    for band in BANDS:

        print("\n" + "-" * 40)

        print(
            "Checking:",
            band
        )

        if band not in item.assets:

            print(
                "ERROR: Asset not found:",
                band
            )

            continue

        asset = item.assets[band]

        signed_url = planetary_computer.sign(
            asset.href
        )

        filename = os.path.basename(
            signed_url.split("?")[0]
        )

        output_path = os.path.join(
            scene_dir,
            filename
        )

        # ----------------------------------------------------
        # Skip already downloaded file
        # ----------------------------------------------------

        if os.path.exists(output_path):

            print(
                "Already exists:"
            )

            print(output_path)

            continue

        print(
            "Downloading:"
        )

        print(
            filename
        )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        response = requests.get(
            signed_url,
            stream=True,
            timeout=120
        )

        response.raise_for_status()

        with open(
            output_path,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

        print(
            "✓ Download complete"
        )

    print("\n✓ Scene download complete")

    print(
        "Saved to:",
        scene_dir
    )

    return True


# ============================================================
# DOWNLOAD ALL SCENES
# ============================================================

def main():

    print("=" * 60)
    print("LANDSAT MULTI-SCENE DOWNLOADER")
    print("=" * 60)

    print(
        "\nNumber of scenes:",
        len(SCENE_IDS)
    )

    successful = 0

    for scene_id in SCENE_IDS:

        try:

            success = download_scene(
                scene_id
            )

            if success:
                successful += 1

        except Exception as e:

            print(
                "\nERROR while downloading:",
                scene_id
            )

            print(
                str(e)
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(
        "\nSuccessful scenes:",
        successful,
        "/",
        len(SCENE_IDS)
    )

    print(
        "\nRaw data directory:"
    )

    print(RAW_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()'''

import os
import time
import requests

import rasterio
import pystac_client
import planetary_computer


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SCENES = 15

MAX_CLOUD_COVER = 70

BBOX = [
    70.0,
    5.0,
    95.0,
    30.0
]

START_DATE = "2022-01-01"
END_DATE = "2026-08-24"


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)


# ============================================================
# REQUIRED LANDSAT ASSETS
# ============================================================

BANDS = [
    "blue",
    "green",
    "red",
    "lwir11",
    "qa_pixel"
]


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

MAX_RETRIES = 5

CONNECT_TIMEOUT = 60

READ_TIMEOUT = 1800

CHUNK_SIZE = 8 * 1024 * 1024

RETRY_DELAY = 10


# ============================================================
# CONNECT TO PLANETARY COMPUTER
# ============================================================

print("=" * 70)
print("CONNECTING TO MICROSOFT PLANETARY COMPUTER")
print("=" * 70)

try:

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    print("✓ Connected to Planetary Computer")

except Exception as e:

    print("\n✗ Could not connect to Planetary Computer")
    print("Error:", e)

    raise


# ============================================================
# VERIFY GEOTIFF
# ============================================================

def verify_tif(file_path):

    try:

        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(
            file_path
        )

        # Reject only obviously broken/empty files.
        if file_size < 10 * 1024:
            return False

        with rasterio.open(file_path) as src:

            if src.width <= 0:
                return False

            if src.height <= 0:
                return False

            if src.count < 1:
                return False

            # Read a tiny area to ensure the TIFF is actually
            # readable and not merely a file with a TIFF name.
            window = rasterio.windows.Window(
                0,
                0,
                min(
                    32,
                    src.width
                ),
                min(
                    32,
                    src.height
                )
            )

            sample = src.read(
                1,
                window=window
            )

            if sample.size == 0:
                return False

        return True

    except Exception as e:

        print(
            "GeoTIFF verification error:",
            e
        )

        return False


# ============================================================
# CHECK SCENE
# ============================================================

def scene_is_valid(item):

    # --------------------------------------------------------
    # Landsat 9 only
    # --------------------------------------------------------

    if not item.id.startswith(
        "LC09_"
    ):

        return False

    # --------------------------------------------------------
    # L2SP only
    # --------------------------------------------------------

    if "_L2SP_" not in item.id:

        return False

    # --------------------------------------------------------
    # Cloud filtering
    # --------------------------------------------------------

    cloud = item.properties.get(
        "eo:cloud_cover",
        100.0
    )

    if cloud is None:
        cloud = 100.0

    try:

        cloud = float(
            cloud
        )

    except Exception:

        return False

    if cloud > MAX_CLOUD_COVER:

        return False

    # --------------------------------------------------------
    # Required assets
    # --------------------------------------------------------

    for band in BANDS:

        if band not in item.assets:

            return False

    return True


# ============================================================
# FIND GOOD SCENES
# ============================================================

def find_scenes():

    print("\n" + "=" * 70)
    print("SEARCHING LANDSAT 9 L2SP SCENES")
    print("=" * 70)

    print("Date range:")
    print(
        f"  {START_DATE} → {END_DATE}"
    )

    print("\nBBOX:")
    print(
        f"  {BBOX}"
    )

    print("\nCloud limit:")
    print(
        f"  {MAX_CLOUD_COVER}%"
    )

    # --------------------------------------------------------
    # STAC SEARCH
    # --------------------------------------------------------

    print(
        "\nStarting STAC search..."
    )

    try:

        search = catalog.search(
            collections=[
                "landsat-c2-l2"
            ],
            bbox=BBOX,
            datetime=(
                f"{START_DATE}/"
                f"{END_DATE}"
            ),
            max_items=200
        )

        print(
            "✓ STAC search created"
        )

    except Exception as e:

        print(
            "\n✗ STAC SEARCH FAILED"
        )

        print(
            "Error:",
            e
        )

        raise

    # --------------------------------------------------------
    # FETCH ITEMS
    # --------------------------------------------------------

    print(
        "\nFetching STAC items..."
    )

    print(
        "This may take some time..."
    )

    try:

        items = search.item_collection()

    except Exception as e:

        print(
            "\n✗ FAILED TO FETCH STAC ITEMS"
        )

        print(
            "Error:",
            e
        )

        raise

    print(
        f"\n✓ STAC items fetched: "
        f"{len(items)}"
    )

    if len(items) == 0:

        raise RuntimeError(
            "\nPlanetary Computer returned ZERO "
            "STAC items.\n\n"
            "The problem is with the STAC search itself."
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    print(
        "\nFiltering scenes..."
    )

    candidates = []

    required_assets = {
        "blue",
        "green",
        "red",
        "lwir11",
        "qa_pixel"
    }

    total_landsat9 = 0
    total_l2sp = 0
    total_assets = 0
    total_cloud = 0

    for item in items:

        # ----------------------------------------------------
        # Landsat 9
        # ----------------------------------------------------

        if not item.id.startswith(
            "LC09_"
        ):

            continue

        total_landsat9 += 1

        # ----------------------------------------------------
        # L2SP
        # ----------------------------------------------------

        if "_L2SP_" not in item.id:

            continue

        total_l2sp += 1

        # ----------------------------------------------------
        # Required assets
        # ----------------------------------------------------

        available = set(
            item.assets.keys()
        )

        if not required_assets.issubset(
            available
        ):

            continue

        total_assets += 1

        # ----------------------------------------------------
        # Cloud cover
        # ----------------------------------------------------

        cloud = item.properties.get(
            "eo:cloud_cover",
            100.0
        )

        if cloud is None:

            cloud = 100.0

        try:

            cloud = float(
                cloud
            )

        except Exception:

            continue

        if cloud > MAX_CLOUD_COVER:

            continue

        total_cloud += 1

        candidates.append(
            item
        )

    # --------------------------------------------------------
    # FILTERING STATISTICS
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "FILTERING RESULTS"
    )

    print(
        "-" * 70
    )

    print(
        f"Total STAC items:              "
        f"{len(items)}"
    )

    print(
        f"Landsat 9 items:               "
        f"{total_landsat9}"
    )

    print(
        f"Landsat 9 L2SP items:          "
        f"{total_l2sp}"
    )

    print(
        f"Items with required assets:     "
        f"{total_assets}"
    )

    print(
        f"Items under cloud threshold:    "
        f"{total_cloud}"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: float(
            item.properties.get(
                "eo:cloud_cover",
                100.0
            )
        )
    )

    print(
        "\nSuitable Landsat 9 L2SP scenes:",
        len(candidates)
    )

    # --------------------------------------------------------
    # SHOW CANDIDATES
    # --------------------------------------------------------

    if candidates:

        print(
            "\nBest available candidates:"
        )

        print(
            "-" * 70
        )

        for i, item in enumerate(
            candidates[:30],
            start=1
        ):

            cloud = float(
                item.properties.get(
                    "eo:cloud_cover",
                    100.0
                )
            )

            print(
                f"{i:02d}. "
                f"{item.id} | "
                f"Cloud: {cloud:.2f}% | "
                f"Date: {item.datetime}"
            )

    # --------------------------------------------------------
    # CHECK COUNT
    # --------------------------------------------------------

    if len(candidates) < NUM_SCENES:

        raise RuntimeError(
            f"\nOnly {len(candidates)} suitable "
            f"Landsat 9 L2SP scenes found.\n\n"
            f"Required: {NUM_SCENES}\n\n"
            "Try one or more of the following:\n"
            "1. Increase MAX_CLOUD_COVER\n"
            "2. Expand the date range\n"
            "3. Expand the BBOX"
        )

    # --------------------------------------------------------
    # SELECT BEST
    # --------------------------------------------------------

    selected = candidates[
        :NUM_SCENES
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED SCENES"
    )

    print(
        "=" * 70
    )

    for i, item in enumerate(
        selected,
        start=1
    ):

        cloud = float(
            item.properties.get(
                "eo:cloud_cover",
                100.0
            )
        )

        print(
            f"{i:02d}. "
            f"{item.id} | "
            f"Cloud: {cloud:.2f}% | "
            f"Date: {item.datetime}"
        )

    return selected


# ============================================================
# DOWNLOAD ONE ASSET
# ============================================================

def download_asset(
    item,
    band,
    scene_dir
):

    asset = item.assets.get(
        band
    )

    if asset is None:

        print(
            f"✗ Missing asset: {band}"
        )

        return False

    # --------------------------------------------------------
    # GET INITIAL URL TO DETERMINE FILE NAME
    # --------------------------------------------------------

    try:

        initial_url = planetary_computer.sign(
            asset.href
        )

    except Exception as e:

        print(
            f"✗ Failed to sign asset URL: {e}"
        )

        return False

    filename = os.path.basename(
        initial_url.split("?")[0]
    )

    output_path = os.path.join(
        scene_dir,
        filename
    )

    temporary_path = (
        output_path
        + ".part"
    )

    # ========================================================
    # ALREADY DOWNLOADED
    # ========================================================

    if os.path.exists(
        output_path
    ):

        print(
            "Existing file found. Verifying..."
        )

        if verify_tif(
            output_path
        ):

            size_mb = (
                os.path.getsize(
                    output_path
                )
                / (1024 ** 2)
            )

            print(
                f"✓ Already exists: "
                f"{filename} "
                f"({size_mb:.2f} MB)"
            )

            return True

        print(
            "✗ Existing file failed verification."
        )

        print(
            "Removing:",
            filename
        )

        try:

            os.remove(
                output_path
            )

        except Exception:

            pass

    # ========================================================
    # REMOVE OLD PART FILE
    # ========================================================

    if os.path.exists(
        temporary_path
    ):

        print(
            "Removing previous incomplete download..."
        )

        try:

            os.remove(
                temporary_path
            )

        except Exception:

            pass

    # ========================================================
    # RETRIES
    # ========================================================

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            f"\nDownloading: {filename}"
        )

        print(
            f"Attempt {attempt}/{MAX_RETRIES}"
        )

        try:

            # ------------------------------------------------
            # GET FRESH SIGNED URL
            # ------------------------------------------------

            signed_url = planetary_computer.sign(
                asset.href
            )

            # ------------------------------------------------
            # HTTP REQUEST
            # ------------------------------------------------

            with requests.get(
                signed_url,
                stream=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT
                )
            ) as response:

                response.raise_for_status()

                # --------------------------------------------
                # FILE SIZE
                # --------------------------------------------

                total = int(
                    response.headers.get(
                        "content-length",
                        0
                    )
                )

                if total:

                    total_mb = (
                        total
                        / (1024 ** 2)
                    )

                    print(
                        f"File size: "
                        f"{total_mb:.2f} MB"
                    )

                else:

                    print(
                        "File size: unknown"
                    )

                # --------------------------------------------
                # DOWNLOAD
                # --------------------------------------------

                downloaded = 0

                start_time = time.time()

                with open(
                    temporary_path,
                    "wb"
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if not chunk:

                            continue

                        f.write(
                            chunk
                        )

                        downloaded += len(
                            chunk
                        )

                        if total:

                            percent = (
                                downloaded
                                / total
                                * 100
                            )

                            elapsed = (
                                time.time()
                                - start_time
                            )

                            if elapsed > 0:

                                speed_mb = (
                                    downloaded
                                    / (1024 ** 2)
                                    / elapsed
                                )

                            else:

                                speed_mb = 0.0

                            print(
                                f"\r  "
                                f"{percent:6.2f}% | "
                                f"{speed_mb:6.2f} MB/s",
                                end="",
                                flush=True
                            )

                        else:

                            downloaded_mb = (
                                downloaded
                                / (1024 ** 2)
                            )

                            print(
                                f"\r  "
                                f"{downloaded_mb:.2f} MB",
                                end="",
                                flush=True
                            )

            print()

            # =================================================
            # CHECK TEMP FILE
            # =================================================

            if not os.path.exists(
                temporary_path
            ):

                raise RuntimeError(
                    "Temporary file was not created."
                )

            temporary_size = os.path.getsize(
                temporary_path
            )

            if temporary_size == 0:

                raise RuntimeError(
                    "Downloaded file is empty."
                )

            # =================================================
            # MOVE INTO FINAL LOCATION
            # =================================================

            os.replace(
                temporary_path,
                output_path
            )

            # =================================================
            # REAL TIFF VERIFICATION
            # =================================================

            if not verify_tif(
                output_path
            ):

                raise RuntimeError(
                    "Downloaded file failed "
                    "GeoTIFF validation."
                )

            final_size_mb = (
                os.path.getsize(
                    output_path
                )
                / (1024 ** 2)
            )

            print(
                f"✓ Download complete: "
                f"{filename} "
                f"({final_size_mb:.2f} MB)"
            )

            return True

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        except requests.exceptions.Timeout as e:

            print(
                "\n✗ REQUEST TIMEOUT"
            )

            print(
                f"  {e}"
            )

        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError as e:

            print(
                "\n✗ CONNECTION ERROR"
            )

            print(
                f"  {e}"
            )

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        except requests.exceptions.HTTPError as e:

            print(
                "\n✗ HTTP ERROR"
            )

            print(
                f"  {e}"
            )

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as e:

            print(
                "\n✗ DOWNLOAD ERROR"
            )

            print(
                f"  {e}"
            )

        # ====================================================
        # CLEAN PARTIAL FILES
        # ====================================================

        if os.path.exists(
            temporary_path
        ):

            try:

                partial_size = (
                    os.path.getsize(
                        temporary_path
                    )
                    / (1024 ** 2)
                )

                print(
                    f"Partial file: "
                    f"{partial_size:.2f} MB"
                )

                os.remove(
                    temporary_path
                )

            except Exception:

                pass

        # If an invalid final file was created,
        # remove it before retrying.

        if os.path.exists(
            output_path
        ):

            if not verify_tif(
                output_path
            ):

                try:

                    os.remove(
                        output_path
                    )

                except Exception:

                    pass

        # ====================================================
        # RETRY
        # ====================================================

        if attempt < MAX_RETRIES:

            print(
                f"Retrying in "
                f"{RETRY_DELAY} seconds..."
            )

            time.sleep(
                RETRY_DELAY
            )

        else:

            print(
                "\n✗ Maximum retries reached."
            )

    return False


# ============================================================
# DOWNLOAD ONE SCENE
# ============================================================

def download_scene(item):

    scene_id = item.id

    print(
        "\n" + "=" * 70
    )

    print(
        "SCENE"
    )

    print(
        "=" * 70
    )

    print(
        "Scene ID:",
        scene_id
    )

    print(
        "Date:",
        item.datetime
    )

    print(
        "Cloud:",
        item.properties.get(
            "eo:cloud_cover",
            "unknown"
        )
    )

    scene_dir = os.path.join(
        RAW_DIR,
        scene_id
    )

    os.makedirs(
        scene_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DOWNLOAD ASSETS
    # --------------------------------------------------------

    success_count = 0

    for band in BANDS:

        print(
            "\n" + "-" * 60
        )

        print(
            f"Asset: {band}"
        )

        print(
            "-" * 60
        )

        success = download_asset(
            item,
            band,
            scene_dir
        )

        if success:

            success_count += 1

        time.sleep(
            1
        )

    # --------------------------------------------------------
    # VALIDATE SCENE
    # --------------------------------------------------------

    print(
        "\nScene validation..."
    )

    if success_count == len(
        BANDS
    ):

        print(
            "\n✓ COMPLETE SCENE:",
            scene_id
        )

        return True

    print(
        "\n✗ INCOMPLETE SCENE:",
        scene_id
    )

    print(
        f"Downloaded "
        f"{success_count}/{len(BANDS)} assets"
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "LANDSAT TRAINING DATA DOWNLOADER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTarget scenes: "
        f"{NUM_SCENES}"
    )

    print(
        f"Maximum cloud cover: "
        f"{MAX_CLOUD_COVER}%"
    )

    print(
        f"Date range: "
        f"{START_DATE} → {END_DATE}"
    )

    print(
        f"Raw directory:\n"
        f"{RAW_DIR}"
    )

    # --------------------------------------------------------
    # FIND SCENES
    # --------------------------------------------------------

    scenes = find_scenes()

    if len(scenes) < NUM_SCENES:

        raise RuntimeError(
            f"\nOnly {len(scenes)} suitable scenes "
            f"were found, but {NUM_SCENES} are required."
        )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    selected = scenes[
        :NUM_SCENES
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL DOWNLOAD LIST"
    )

    print(
        "=" * 70
    )

    for index, item in enumerate(
        selected,
        start=1
    ):

        print(
            f"{index:02d}. "
            f"{item.id}"
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    successful = 0

    failed = []

    for index, item in enumerate(
        selected,
        start=1
    ):

        print(
            "\n\n"
            + "#" * 70
        )

        print(
            f"SCENE "
            f"{index}/{NUM_SCENES}"
        )

        print(
            "#" * 70
        )

        try:

            success = download_scene(
                item
            )

            if success:

                successful += 1

            else:

                failed.append(
                    item.id
                )

        except KeyboardInterrupt:

            print(
                "\n\nDownload interrupted by user."
            )

            break

        except Exception as e:

            print(
                "\n✗ Unexpected scene error:"
            )

            print(
                e
            )

            failed.append(
                item.id
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "DOWNLOAD SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSuccessful scenes: "
        f"{successful}/{NUM_SCENES}"
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
            "\n✓ No failed scenes."
        )

    print(
        "\nRaw data directory:"
    )

    print(
        RAW_DIR
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if successful < NUM_SCENES:

        print(
            "\n" + "!" * 70
        )

        print(
            "WARNING"
        )

        print(
            "Some scenes are incomplete."
        )

        print(
            "Run the downloader again."
        )

        print(
            "Already verified files will be skipped, "
            "so only missing/failed assets will be downloaded."
        )

        print(
            "!" * 70
        )

    else:

        print(
            "\n" + "=" * 70
        )

        print(
            "✓ ALL SCENES DOWNLOADED SUCCESSFULLY"
        )

        print(
            "You can now run preprocessing."
        )

        print(
            "=" * 70
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()