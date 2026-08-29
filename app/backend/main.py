import json
import shutil
import sys
import time

from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)

from fastapi.responses import (
    FileResponse
)

from fastapi.staticfiles import (
    StaticFiles
)


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
# INFERENCE
# ============================================================

from src.inference.predict import (
    CHECKPOINT_PATH,
    predict_landsat
)


# ============================================================
# DIRECTORIES
# ============================================================

TEMP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "temp"
)

FRONTEND_DIR = (
    PROJECT_ROOT
    / "app"
    / "frontend"
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="IRIS API",
    description=(
        "Landsat thermal-to-RGB reconstruction "
        "with diagnostics and optional quantitative validation."
    ),
    version="1.2.0"
)


# ============================================================
# TEMP OUTPUT FILES
# ============================================================

app.mount(
    "/temp",
    StaticFiles(
        directory=str(
            TEMP_DIR
        )
    ),
    name="temp"
)


# ============================================================
# TEMP CLEANUP
# ============================================================

def clear_temp_directory():

    if TEMP_DIR.exists():

        for item in TEMP_DIR.iterdir():

            try:

                if item.is_file():
                    item.unlink()

                elif item.is_dir():
                    shutil.rmtree(
                        item
                    )

            except Exception as error:

                print(
                    "Could not delete:",
                    item,
                    repr(error)
                )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_upload(
    upload,
    destination
):

    contents = await upload.read()

    if not contents:
        raise ValueError(
            f"{upload.filename} is empty."
        )

    with open(
        destination,
        "wb"
    ) as file:

        file.write(
            contents
        )


# ============================================================
# TIFF VALIDATION
# ============================================================

def validate_tiff(
    upload,
    label
):

    if upload is None:
        return

    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail=f"{label} has no filename."
        )

    suffix = (
        Path(upload.filename)
        .suffix
        .lower()
    )

    if suffix not in {
        ".tif",
        ".tiff"
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} must be a "
                ".tif or .tiff file."
            )
        )


# ============================================================
# BAND NAME VALIDATION
# ============================================================

def validate_band_name(
    upload,
    expected_text,
    label
):

    if upload is None:
        return

    if (
        expected_text.upper()
        not in
        upload.filename.upper()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} does not appear to "
                f"be a {expected_text} Landsat file."
            )
        )


# ============================================================
# LANDSAT SCENE ID
# ============================================================

def get_scene_id(
    filename
):

    stem = (
        Path(filename)
        .stem
        .upper()
    )

    suffixes = [
        "_ST_B10",
        "_QA_PIXEL",
        "_SR_B2",
        "_SR_B3",
        "_SR_B4"
    ]

    for suffix in suffixes:

        if stem.endswith(
            suffix
        ):
            return stem[
                :-len(suffix)
            ]

    return stem


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.get("/")
def serve_index():

    return FileResponse(
        str(
            FRONTEND_DIR
            /
            "index.html"
        ),
        media_type="text/html"
    )


@app.get("/style.css")
def serve_css():

    return FileResponse(
        str(
            FRONTEND_DIR
            /
            "style.css"
        ),
        media_type="text/css"
    )


@app.get("/script.js")
def serve_js():

    return FileResponse(
        str(
            FRONTEND_DIR
            /
            "script.js"
        ),
        media_type="application/javascript"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "api": "online",

        "model": (
            "ready"
            if CHECKPOINT_PATH.exists()
            else "not_available"
        ),

        "checkpoint": (
            CHECKPOINT_PATH.name
            if CHECKPOINT_PATH.exists()
            else None
        )
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(

    b10_file:
        UploadFile
        =
        File(...),

    qa_file:
        UploadFile | None
        =
        File(None),

    b2_file:
        UploadFile | None
        =
        File(None),

    b3_file:
        UploadFile | None
        =
        File(None),

    b4_file:
        UploadFile | None
        =
        File(None)
):

    # ========================================================
    # CHECKPOINT
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                "Model checkpoint is not available."
            )
        )

    # ========================================================
    # FILE TYPE
    # ========================================================

    validate_tiff(
        b10_file,
        "ST_B10"
    )

    validate_tiff(
        qa_file,
        "QA_PIXEL"
    )

    validate_tiff(
        b2_file,
        "B2"
    )

    validate_tiff(
        b3_file,
        "B3"
    )

    validate_tiff(
        b4_file,
        "B4"
    )

    # ========================================================
    # FILE NAMES
    # ========================================================

    validate_band_name(
        b10_file,
        "ST_B10",
        "Thermal input"
    )

    if qa_file is not None:

        validate_band_name(
            qa_file,
            "QA_PIXEL",
            "QA input"
        )

    visible_supplied = [
        b2_file is not None,
        b3_file is not None,
        b4_file is not None
    ]

    if (
        any(
            visible_supplied
        )
        and
        not all(
            visible_supplied
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Ground-truth validation requires "
                "B2, B3 and B4 together."
            )
        )

    validation_mode = all(
        visible_supplied
    )

    if validation_mode:

        validate_band_name(
            b2_file,
            "B2",
            "B2 input"
        )

        validate_band_name(
            b3_file,
            "B3",
            "B3 input"
        )

        validate_band_name(
            b4_file,
            "B4",
            "B4 input"
        )

    # ========================================================
    # SAME SCENE CHECK
    # ========================================================

    reference_scene = get_scene_id(
        b10_file.filename
    )

    compare_files = []

    if qa_file is not None:
        compare_files.append(
            qa_file
        )

    if validation_mode:

        compare_files.extend(
            [
                b2_file,
                b3_file,
                b4_file
            ]
        )

    for upload in compare_files:

        if (
            get_scene_id(
                upload.filename
            )
            !=
            reference_scene
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded Landsat files "
                    "do not belong to the same scene."
                )
            )

    try:

        clear_temp_directory()

        # ====================================================
        # PATHS
        # ====================================================

        b10_path = (
            TEMP_DIR
            /
            "current_ST_B10.tif"
        )

        qa_path = (
            TEMP_DIR
            /
            "current_QA_PIXEL.tif"
        )

        b2_path = (
            TEMP_DIR
            /
            "current_SR_B2.tif"
        )

        b3_path = (
            TEMP_DIR
            /
            "current_SR_B3.tif"
        )

        b4_path = (
            TEMP_DIR
            /
            "current_SR_B4.tif"
        )

        # ====================================================
        # SAVE
        # ====================================================

        print(
            "\nSaving ST_B10..."
        )

        await save_upload(
            b10_file,
            b10_path
        )

        actual_qa_path = None

        if qa_file is not None:

            print(
                "Saving QA_PIXEL..."
            )

            await save_upload(
                qa_file,
                qa_path
            )

            actual_qa_path = (
                qa_path
            )

        actual_b2_path = None
        actual_b3_path = None
        actual_b4_path = None

        if validation_mode:

            print(
                "Saving B2/B3/B4..."
            )

            await save_upload(
                b2_file,
                b2_path
            )

            await save_upload(
                b3_file,
                b3_path
            )

            await save_upload(
                b4_file,
                b4_path
            )

            actual_b2_path = (
                b2_path
            )

            actual_b3_path = (
                b3_path
            )

            actual_b4_path = (
                b4_path
            )

        # ====================================================
        # INFERENCE
        # ====================================================

        start_time = time.time()

        (
            full_thermal_preview,
            selected_thermal_patch,
            generated_rgb,
            metrics
        ) = predict_landsat(

            b10_path=
                b10_path,

            qa_path=
                actual_qa_path,

            b2_path=
                actual_b2_path,

            b3_path=
                actual_b3_path,

            b4_path=
                actual_b4_path
        )

        metrics[
            "processing_time_seconds"
        ] = round(
            time.time()
            -
            start_time,
            2
        )

        # ====================================================
        # OUTPUT PATHS
        # ====================================================

        full_preview_path = (
            TEMP_DIR
            /
            "thermal_scene_preview.png"
        )

        patch_preview_path = (
            TEMP_DIR
            /
            "selected_thermal_patch.png"
        )

        generated_path = (
            TEMP_DIR
            /
            "generated_rgb.png"
        )

        metrics_path = (
            TEMP_DIR
            /
            "metrics.json"
        )

        # ====================================================
        # SAVE OUTPUTS
        # ====================================================

        full_thermal_preview.save(
            full_preview_path
        )

        selected_thermal_patch.save(
            patch_preview_path
        )

        generated_rgb.save(
            generated_path
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        # ====================================================
        # WARNING
        # ====================================================

        warning = None

        if qa_file is None:

            warning = (
                "QA_PIXEL was not supplied. "
                "Cloud/shadow/snow quality masking "
                "was not applied."
            )

        timestamp = int(
            time.time()
            *
            1000
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return {
            "success": True,

            "model":
                CHECKPOINT_PATH.name,

            "b10_filename":
                b10_file.filename,

            "qa_filename": (
                qa_file.filename
                if qa_file
                else None
            ),

            "validation_mode":
                validation_mode,

            "b2_filename": (
                b2_file.filename
                if validation_mode
                else None
            ),

            "b3_filename": (
                b3_file.filename
                if validation_mode
                else None
            ),

            "b4_filename": (
                b4_file.filename
                if validation_mode
                else None
            ),

            "cloud_mask_used":
                actual_qa_path is not None,

            "warning":
                warning,

            "full_thermal_preview":
                (
                    "/temp/"
                    "thermal_scene_preview.png"
                    f"?v={timestamp}"
                ),

            "selected_thermal_patch":
                (
                    "/temp/"
                    "selected_thermal_patch.png"
                    f"?v={timestamp}"
                ),

            "generated_rgb":
                (
                    "/temp/"
                    "generated_rgb.png"
                    f"?v={timestamp}"
                ),

            "metrics":
                metrics
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "\nPrediction failed:"
        )

        print(
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            )
        )