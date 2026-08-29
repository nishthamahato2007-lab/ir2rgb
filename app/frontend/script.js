// ============================================================
// IRIS FRONTEND
// ============================================================

const API_URL = "/predict";

console.log(
    "IRIS frontend initialized."
);


// ============================================================
// INPUTS
// ============================================================

const b10Input =
    document.getElementById("b10Input");

const qaInput =
    document.getElementById("qaInput");

const b2Input =
    document.getElementById("b2Input");

const b3Input =
    document.getElementById("b3Input");

const b4Input =
    document.getElementById("b4Input");


// ============================================================
// UPLOAD ZONES
// ============================================================

const b10UploadZone =
    document.getElementById("b10UploadZone");

const qaUploadZone =
    document.getElementById("qaUploadZone");

const b2UploadZone =
    document.getElementById("b2UploadZone");

const b3UploadZone =
    document.getElementById("b3UploadZone");

const b4UploadZone =
    document.getElementById("b4UploadZone");


// ============================================================
// LABELS
// ============================================================

const b10FileLabel =
    document.getElementById("b10FileLabel");

const qaFileLabel =
    document.getElementById("qaFileLabel");

const b2FileLabel =
    document.getElementById("b2FileLabel");

const b3FileLabel =
    document.getElementById("b3FileLabel");

const b4FileLabel =
    document.getElementById("b4FileLabel");


// ============================================================
// PREVIEWS / STATUS
// ============================================================

const inputStatus =
    document.getElementById("inputStatus");

const inputPreview =
    document.getElementById("inputPreview");

const inputPlaceholder =
    document.getElementById("inputPlaceholder");

const patchPreview =
    document.getElementById("patchPreview");

const patchPlaceholder =
    document.getElementById("patchPlaceholder");

const outputPreview =
    document.getElementById("outputPreview");

const outputPlaceholder =
    document.getElementById("outputPlaceholder");

const generateBtn =
    document.getElementById("generateBtn");

const clearBtn =
    document.getElementById("clearBtn");

const inferStatus =
    document.getElementById("inferStatus");

const outputStatus =
    document.getElementById("outputStatus");

const fileName =
    document.getElementById("fileName");

const maskBadge =
    document.getElementById("maskBadge");

const warningBanner =
    document.getElementById("warningBanner");

const scanOverlay =
    document.getElementById("scanOverlay");

const logBox =
    document.getElementById("logBox");


// ============================================================
// TELEMETRY
// ============================================================

const metricsStatus =
    document.getElementById("metricsStatus");

const metricSceneSize =
    document.getElementById("metricSceneSize");

const metricSceneValid =
    document.getElementById("metricSceneValid");

const metricPatchValid =
    document.getElementById("metricPatchValid");

const metricCloudMask =
    document.getElementById("metricCloudMask");

const metricMinTemp =
    document.getElementById("metricMinTemp");

const metricMeanTemp =
    document.getElementById("metricMeanTemp");

const metricMaxTemp =
    document.getElementById("metricMaxTemp");

const metricTime =
    document.getElementById("metricTime");


// ============================================================
// DIAGNOSTICS
// ============================================================

const diagnosticsStatus =
    document.getElementById("diagnosticsStatus");

const diagSceneCharacter =
    document.getElementById("diagSceneCharacter");

const diagThermalCondition =
    document.getElementById("diagThermalCondition");

const diagThermalStd =
    document.getElementById("diagThermalStd");

const diagHotSurface =
    document.getElementById("diagHotSurface");

const diagCoolSurface =
    document.getElementById("diagCoolSurface");

const diagCloud =
    document.getElementById("diagCloud");

const diagPatch =
    document.getElementById("diagPatch");

const diagInterpretation =
    document.getElementById("diagInterpretation");


// ============================================================
// VALIDATION
// ============================================================

const validationStatus =
    document.getElementById("validationStatus");

const metricValidationL1 =
    document.getElementById("metricValidationL1");

const metricValidationPSNR =
    document.getElementById("metricValidationPSNR");

const metricValidationSSIM =
    document.getElementById("metricValidationSSIM");

const metricValidationValid =
    document.getElementById("metricValidationValid");

const metricConfidence =
    document.getElementById("metricConfidence");

const validationMessage =
    document.getElementById("validationMessage");

const confidenceDescription =
    document.getElementById("confidenceDescription");


// ============================================================
// STATE
// ============================================================

let b10File = null;

let qaFile = null;

let b2File = null;

let b3File = null;

let b4File = null;


// ============================================================
// LOG
// ============================================================

function addLog(message) {

    console.log(
        "[IRIS]",
        message
    );

    if (!logBox) {
        return;
    }

    const p =
        document.createElement("p");

    p.textContent =
        `> ${message}`;

    logBox.appendChild(
        p
    );

    logBox.scrollTop =
        logBox.scrollHeight;
}


// ============================================================
// TIFF CHECK
// ============================================================

function isTiff(file) {

    if (!file) {
        return false;
    }

    const name =
        file.name.toLowerCase();

    return (
        name.endsWith(".tif")
        ||
        name.endsWith(".tiff")
    );
}


// ============================================================
// VALIDATION BAND STATE
// ============================================================

function validationBandsComplete() {

    return (
        b2File !== null
        &&
        b3File !== null
        &&
        b4File !== null
    );
}


function validationBandsPartial() {

    const states = [
        b2File !== null,
        b3File !== null,
        b4File !== null
    ];

    return (
        states.some(Boolean)
        &&
        !states.every(Boolean)
    );
}


// ============================================================
// FORMAT TEMPERATURE
// ============================================================

function formatTemperature(value) {

    if (
        value === null
        ||
        value === undefined
    ) {
        return "—";
    }

    return `${value} °C`;
}


// ============================================================
// LOAD IMAGE
// ============================================================

function loadImage(
    imageElement,
    placeholderElement,
    imagePath,
    label
) {

    return new Promise(
        (
            resolve,
            reject
        ) => {

            if (!imageElement) {

                reject(
                    new Error(
                        `${label}: image element missing.`
                    )
                );

                return;
            }

            if (!imagePath) {

                reject(
                    new Error(
                        `${label}: image path missing.`
                    )
                );

                return;
            }

            imageElement.onload =
                function () {

                    imageElement.style.display =
                        "block";

                    if (placeholderElement) {

                        placeholderElement.style.display =
                            "none";
                    }

                    resolve(
                        imagePath
                    );
                };

            imageElement.onerror =
                function () {

                    reject(
                        new Error(
                            `${label} failed to load.`
                        )
                    );
                };

            imageElement.src =
                imagePath;
        }
    );
}


// ============================================================
// RESET OUTPUT RESULTS
// ============================================================

function resetResults() {

    const previews = [
        [
            inputPreview,
            inputPlaceholder
        ],

        [
            patchPreview,
            patchPlaceholder
        ],

        [
            outputPreview,
            outputPlaceholder
        ]
    ];

    previews.forEach(
        (
            [
                image,
                placeholder
            ]
        ) => {

            if (image) {

                image.onload =
                    null;

                image.onerror =
                    null;

                image.removeAttribute(
                    "src"
                );

                image.style.display =
                    "none";
            }

            if (placeholder) {

                placeholder.style.display =
                    "block";
            }
        }
    );


    // ========================================================
    // GENERAL STATUS
    // ========================================================

    if (inferStatus) {

        inferStatus.textContent =
            "IDLE";

        inferStatus.classList.remove(
            "running",
            "error"
        );
    }

    if (outputStatus) {

        outputStatus.textContent =
            "Standby";
    }

    if (metricsStatus) {

        metricsStatus.textContent =
            "NO DATA";

        metricsStatus.classList.remove(
            "running",
            "error"
        );
    }


    // ========================================================
    // TELEMETRY
    // ========================================================

    [
        metricSceneSize,
        metricSceneValid,
        metricPatchValid,
        metricCloudMask,
        metricMinTemp,
        metricMeanTemp,
        metricMaxTemp,
        metricTime
    ].forEach(
        element => {

            if (element) {

                element.textContent =
                    "—";
            }
        }
    );


    // ========================================================
    // QA BADGE
    // ========================================================

    if (maskBadge) {

        maskBadge.textContent =
            "QA UNKNOWN";

        maskBadge.classList.remove(
            "enabled",
            "disabled"
        );
    }


    // ========================================================
    // WARNING
    // ========================================================

    if (warningBanner) {

        warningBanner.textContent =
            "";

        warningBanner.classList.remove(
            "visible"
        );
    }


    // ========================================================
    // DIAGNOSTICS
    // ========================================================

    if (diagnosticsStatus) {

        diagnosticsStatus.textContent =
            "AWAITING ANALYSIS";

        diagnosticsStatus.classList.remove(
            "running",
            "error"
        );
    }

    [
        diagSceneCharacter,
        diagThermalCondition,
        diagThermalStd,
        diagHotSurface,
        diagCoolSurface,
        diagCloud,
        diagPatch
    ].forEach(
        element => {

            if (element) {
                element.textContent =
                    "—";
            }
        }
    );

    if (diagInterpretation) {

        diagInterpretation.textContent =
            (
                "Scene interpretation will appear "
                +
                "after reconstruction."
            );
    }


    // ========================================================
    // VALIDATION
    // ========================================================

    if (validationStatus) {

        validationStatus.textContent =
            validationBandsComplete()
                ? "READY"
                : "NOT AVAILABLE";

        validationStatus.classList.remove(
            "running",
            "error"
        );
    }

    [
        metricValidationL1,
        metricValidationPSNR,
        metricValidationSSIM,
        metricValidationValid,
        metricConfidence
    ].forEach(
        element => {

            if (element) {

                element.textContent =
                    "—";
            }
        }
    );

    if (validationMessage) {

        validationMessage.innerHTML =
            `
            <span>INFO</span>
            Upload matching B2 + B3 + B4 together to enable
            quantitative validation.
            `;
    }

    if (confidenceDescription) {

        confidenceDescription.innerHTML =
            `
            <span>CONFIDENCE</span>
            <p>Awaiting reconstruction.</p>
            `;
    }

    if (scanOverlay) {

        scanOverlay.classList.remove(
            "active"
        );
    }
}


// ============================================================
// UPDATE GENERATE BUTTON
// ============================================================

function updateGenerateButton() {

    if (generateBtn) {

        generateBtn.disabled =
            b10File === null;
    }

    if (!inputStatus) {
        return;
    }

    if (!b10File) {

        inputStatus.textContent =
            "AWAITING";
    }

    else if (
        validationBandsPartial()
    ) {

        inputStatus.textContent =
            "VALIDATION INCOMPLETE";
    }

    else if (
        validationBandsComplete()
    ) {

        inputStatus.textContent =
            "VALIDATION READY";
    }

    else if (qaFile) {

        inputStatus.textContent =
            "B10 + QA READY";
    }

    else {

        inputStatus.textContent =
            "B10 READY";
    }
}


// ============================================================
// OPTIONAL BAND HANDLER
// ============================================================

function setupOptionalInput(
    input,
    zone,
    label,
    displayName,
    setter
) {

    if (!input) {
        return;
    }

    input.addEventListener(
        "change",
        function (event) {

            const file =
                event.target.files[0];

            if (!file) {

                setter(
                    null
                );

                if (zone) {

                    zone.classList.remove(
                        "file-selected"
                    );
                }

                updateGenerateButton();

                return;
            }

            if (!isTiff(file)) {

                alert(
                    `${displayName} must be a TIFF file.`
                );

                input.value =
                    "";

                setter(
                    null
                );

                updateGenerateButton();

                return;
            }

            setter(
                file
            );

            resetResults();

            if (zone) {

                zone.classList.add(
                    "file-selected"
                );
            }

            if (label) {

                label.textContent =
                    file.name;
            }

            addLog(
                `${displayName} selected: ${file.name}`
            );

            updateGenerateButton();
        }
    );
}


// ============================================================
// B10
// ============================================================

if (b10Input) {

    b10Input.addEventListener(
        "change",
        function (event) {

            const file =
                event.target.files[0];

            if (!file) {

                b10File =
                    null;

                updateGenerateButton();

                return;
            }

            if (!isTiff(file)) {

                alert(
                    "Please select a valid ST_B10 TIFF."
                );

                b10Input.value =
                    "";

                b10File =
                    null;

                updateGenerateButton();

                return;
            }

            b10File =
                file;

            resetResults();

            if (b10UploadZone) {

                b10UploadZone.classList.add(
                    "file-selected"
                );
            }

            if (b10FileLabel) {

                b10FileLabel.textContent =
                    file.name;
            }

            if (fileName) {

                fileName.textContent =
                    file.name;
            }

            addLog(
                `ST_B10 selected: ${file.name}`
            );

            updateGenerateButton();
        }
    );
}


// ============================================================
// QA
// ============================================================

setupOptionalInput(
    qaInput,
    qaUploadZone,
    qaFileLabel,
    "QA_PIXEL",
    file => {
        qaFile = file;
    }
);


// ============================================================
// B2
// ============================================================

setupOptionalInput(
    b2Input,
    b2UploadZone,
    b2FileLabel,
    "SR_B2",
    file => {
        b2File = file;
    }
);


// ============================================================
// B3
// ============================================================

setupOptionalInput(
    b3Input,
    b3UploadZone,
    b3FileLabel,
    "SR_B3",
    file => {
        b3File = file;
    }
);


// ============================================================
// B4
// ============================================================

setupOptionalInput(
    b4Input,
    b4UploadZone,
    b4FileLabel,
    "SR_B4",
    file => {
        b4File = file;
    }
);


// ============================================================
// GENERATE
// ============================================================

if (generateBtn) {

    generateBtn.addEventListener(
        "click",
        async function (event) {

            event.preventDefault();
            event.stopPropagation();

            if (!b10File) {

                alert(
                    "Please select ST_B10 first."
                );

                return;
            }

            if (
                validationBandsPartial()
            ) {

                alert(
                    "For validation, B2, B3 and B4 must "
                    +
                    "all be uploaded together."
                );

                return;
            }

            const formData =
                new FormData();

            formData.append(
                "b10_file",
                b10File
            );

            if (qaFile) {

                formData.append(
                    "qa_file",
                    qaFile
                );
            }

            if (
                validationBandsComplete()
            ) {

                formData.append(
                    "b2_file",
                    b2File
                );

                formData.append(
                    "b3_file",
                    b3File
                );

                formData.append(
                    "b4_file",
                    b4File
                );
            }


            // =================================================
            // RUNNING STATE
            // =================================================

            generateBtn.disabled =
                true;

            if (clearBtn) {

                clearBtn.disabled =
                    true;
            }

            if (inferStatus) {

                inferStatus.textContent =
                    "RUNNING";

                inferStatus.classList.remove(
                    "error"
                );

                inferStatus.classList.add(
                    "running"
                );
            }

            if (metricsStatus) {

                metricsStatus.textContent =
                    "PROCESSING";

                metricsStatus.classList.add(
                    "running"
                );
            }

            if (diagnosticsStatus) {

                diagnosticsStatus.textContent =
                    "ANALYZING";

                diagnosticsStatus.classList.add(
                    "running"
                );
            }

            if (validationStatus) {

                if (
                    validationBandsComplete()
                ) {

                    validationStatus.textContent =
                        "VALIDATING";

                    validationStatus.classList.add(
                        "running"
                    );
                }

                else {

                    validationStatus.textContent =
                        "NOT AVAILABLE";
                }
            }

            if (outputStatus) {

                outputStatus.textContent =
                    "Processing Landsat scene";
            }

            if (scanOverlay) {

                scanOverlay.classList.add(
                    "active"
                );
            }

            if (warningBanner) {

                warningBanner.textContent =
                    "";

                warningBanner.classList.remove(
                    "visible"
                );
            }

            addLog(
                "Uploading Landsat scene to IRIS..."
            );

            if (qaFile) {

                addLog(
                    "QA_PIXEL quality masking enabled."
                );
            }

            if (
                validationBandsComplete()
            ) {

                addLog(
                    "Ground-truth validation enabled."
                );
            }


            try {

                const response =
                    await fetch(
                        API_URL,
                        {
                            method:
                                "POST",

                            body:
                                formData
                        }
                    );

                const rawText =
                    await response.text();

                if (!rawText) {

                    throw new Error(
                        `Backend returned empty response `
                        +
                        `(HTTP ${response.status}).`
                    );
                }

                let data;

                try {

                    data =
                        JSON.parse(
                            rawText
                        );
                }

                catch (error) {

                    throw new Error(
                        "Backend response was not valid JSON."
                    );
                }

                if (!response.ok) {

                    throw new Error(
                        data.detail
                        ||
                        `Backend returned HTTP ${response.status}`
                    );
                }

                if (!data.success) {

                    throw new Error(
                        "Backend returned success=false."
                    );
                }

                console.log(
                    "IRIS response:",
                    data
                );

                addLog(
                    "Backend reconstruction complete."
                );


                // =================================================
                // IMAGES
                // =================================================

                const imageResults =
                    await Promise.allSettled(
                        [
                            loadImage(
                                inputPreview,
                                inputPlaceholder,
                                data.full_thermal_preview,
                                "Thermal preview"
                            ),

                            loadImage(
                                patchPreview,
                                patchPlaceholder,
                                data.selected_thermal_patch,
                                "Selected patch"
                            ),

                            loadImage(
                                outputPreview,
                                outputPlaceholder,
                                data.generated_rgb,
                                "Generated RGB"
                            )
                        ]
                    );

                const failures =
                    imageResults.filter(
                        result =>
                            result.status === "rejected"
                    );

                if (failures.length > 0) {

                    addLog(
                        `${failures.length} image(s) failed to load.`
                    );
                }


                // =================================================
                // METRICS
                // =================================================

                const metrics =
                    data.metrics || {};

                if (metricSceneSize) {

                    metricSceneSize.textContent =
                        (
                            metrics.scene_width !== undefined
                            &&
                            metrics.scene_height !== undefined
                        )
                            ? `${metrics.scene_width} × ${metrics.scene_height}`
                            : "—";
                }

                if (metricSceneValid) {

                    metricSceneValid.textContent =
                        metrics.scene_valid_percentage !== undefined
                            ? `${metrics.scene_valid_percentage}%`
                            : "—";
                }

                if (metricPatchValid) {

                    metricPatchValid.textContent =
                        metrics.selected_patch_valid_percentage !== undefined
                            ? `${metrics.selected_patch_valid_percentage}%`
                            : "—";
                }

                if (metricCloudMask) {

                    metricCloudMask.textContent =
                        data.cloud_mask_used
                            ? "Enabled"
                            : "Disabled";
                }

                if (metricMinTemp) {

                    metricMinTemp.textContent =
                        formatTemperature(
                            metrics.minimum_temperature_c
                        );
                }

                if (metricMeanTemp) {

                    metricMeanTemp.textContent =
                        formatTemperature(
                            metrics.mean_temperature_c
                        );
                }

                if (metricMaxTemp) {

                    metricMaxTemp.textContent =
                        formatTemperature(
                            metrics.maximum_temperature_c
                        );
                }

                if (metricTime) {

                    metricTime.textContent =
                        metrics.processing_time_seconds !== undefined
                            ? `${metrics.processing_time_seconds} s`
                            : "—";
                }


                // =================================================
                // DIAGNOSTICS
                // =================================================

                const diagnostics =
                    metrics.diagnostics || {};

                if (diagSceneCharacter) {

                    diagSceneCharacter.textContent =
                        diagnostics.scene_character
                        ||
                        "—";
                }

                if (diagThermalCondition) {

                    diagThermalCondition.textContent =
                        diagnostics.thermal_condition
                        ||
                        "—";
                }

                if (diagThermalStd) {

                    diagThermalStd.textContent =
                        diagnostics.thermal_std_c != null
                            ? `${diagnostics.thermal_std_c} °C`
                            : "—";
                }

                if (diagHotSurface) {

                    diagHotSurface.textContent =
                        diagnostics.hot_surface_percentage != null
                            ? `${diagnostics.hot_surface_percentage}%`
                            : "—";
                }

                if (diagCoolSurface) {

                    diagCoolSurface.textContent =
                        diagnostics.cool_surface_percentage != null
                            ? `${diagnostics.cool_surface_percentage}%`
                            : "—";
                }

                if (diagCloud) {

                    diagCloud.textContent =
                        diagnostics.invalid_percentage != null
                            ? `${diagnostics.invalid_percentage}%`
                            : "—";
                }

                if (diagPatch) {

                    diagPatch.textContent =
                        diagnostics.patch_character
                        ||
                        "—";
                }

                if (diagInterpretation) {

                    diagInterpretation.textContent =
                        diagnostics.interpretation
                        ||
                        "No interpretation available.";
                }

                if (diagnosticsStatus) {

                    diagnosticsStatus.textContent =
                        "ANALYSIS COMPLETE";

                    diagnosticsStatus.classList.remove(
                        "running",
                        "error"
                    );
                }


                // =================================================
                // VALIDATION
                // =================================================

                const validation =
                    metrics.validation || {};

                if (
                    validation.available
                ) {

                    if (metricValidationL1) {

                        metricValidationL1.textContent =
                            validation.l1 != null
                                ? validation.l1
                                : "—";
                    }

                    if (metricValidationPSNR) {

                        metricValidationPSNR.textContent =
                            validation.psnr_db != null
                                ? `${validation.psnr_db} dB`
                                : "—";
                    }

                    if (metricValidationSSIM) {

                        metricValidationSSIM.textContent =
                            validation.ssim != null
                                ? validation.ssim
                                : "—";
                    }

                    if (metricValidationValid) {

                        metricValidationValid.textContent =
                            validation.valid_percentage != null
                                ? `${validation.valid_percentage}%`
                                : "—";
                    }

                    if (validationStatus) {

                        validationStatus.textContent =
                            "VALIDATED";

                        validationStatus.classList.remove(
                            "running",
                            "error"
                        );
                    }

                    if (validationMessage) {

                        validationMessage.innerHTML =
                            `
                            <span>SUCCESS</span>
                            Generated RGB was compared with
                            matching Landsat visible-band
                            ground truth.
                            `;
                    }

                    addLog(
                        `Validation — L1: ${validation.l1}, `
                        +
                        `PSNR: ${validation.psnr_db} dB, `
                        +
                        `SSIM: ${validation.ssim}`
                    );
                }

                else {

                    if (validationStatus) {

                        validationStatus.textContent =
                            "NOT AVAILABLE";

                        validationStatus.classList.remove(
                            "running",
                            "error"
                        );
                    }

                    if (validationMessage) {

                        validationMessage.innerHTML =
                            `
                            <span>INFO</span>
                            ${
                                validation.reason
                                ||
                                "Ground-truth bands not supplied."
                            }
                            `;
                    }
                }


                // =================================================
                // CONFIDENCE
                // =================================================

                const confidence =
                    metrics.confidence || {};

                if (metricConfidence) {

                    if (
                        confidence.score !== null
                        &&
                        confidence.score !== undefined
                    ) {

                        metricConfidence.textContent =
                            `${confidence.score}% • ${
                                confidence.level || ""
                            }`;
                    }

                    else {

                        metricConfidence.textContent =
                            "—";
                    }
                }

                if (confidenceDescription) {

                    confidenceDescription.innerHTML =
                        `
                        <span>
                            ${confidence.type || "CONFIDENCE"}
                        </span>

                        <p>
                            ${
                                confidence.description
                                ||
                                "Confidence information unavailable."
                            }
                        </p>
                        `;
                }

                if (
                    confidence.score !== undefined
                ) {

                    addLog(
                        `Confidence: ${confidence.score}% `
                        +
                        `(${confidence.level}).`
                    );
                }


                // =================================================
                // QA BADGE
                // =================================================

                if (maskBadge) {

                    maskBadge.classList.remove(
                        "enabled",
                        "disabled"
                    );

                    if (
                        data.cloud_mask_used
                    ) {

                        maskBadge.textContent =
                            "QA MASKED";

                        maskBadge.classList.add(
                            "enabled"
                        );
                    }

                    else {

                        maskBadge.textContent =
                            "QA NOT USED";

                        maskBadge.classList.add(
                            "disabled"
                        );
                    }
                }


                // =================================================
                // WARNING
                // =================================================

                if (
                    warningBanner
                    &&
                    data.warning
                ) {

                    warningBanner.textContent =
                        data.warning;

                    warningBanner.classList.add(
                        "visible"
                    );
                }


                // =================================================
                // COMPLETE
                // =================================================

                if (inferStatus) {

                    inferStatus.textContent =
                        "COMPLETE";

                    inferStatus.classList.remove(
                        "running",
                        "error"
                    );
                }

                if (metricsStatus) {

                    metricsStatus.textContent =
                        "COMPLETE";

                    metricsStatus.classList.remove(
                        "running",
                        "error"
                    );
                }

                if (outputStatus) {

                    outputStatus.textContent =
                        "Reconstruction complete";
                }

                addLog(
                    "Thermal preprocessing complete."
                );

                addLog(
                    `Selected patch validity: ${
                        metrics.selected_patch_valid_percentage ?? "—"
                    }%`
                );

                addLog(
                    `Processing time: ${
                        metrics.processing_time_seconds ?? "—"
                    } seconds`
                );

                addLog(
                    `RGB reconstruction generated using ${
                        data.model || "IRIS model"
                    }.`
                );
            }

            catch (error) {

                console.error(
                    "IRIS frontend error:",
                    error
                );

                if (inferStatus) {

                    inferStatus.textContent =
                        "ERROR";

                    inferStatus.classList.remove(
                        "running"
                    );

                    inferStatus.classList.add(
                        "error"
                    );
                }

                if (metricsStatus) {

                    metricsStatus.textContent =
                        "FAILED";

                    metricsStatus.classList.remove(
                        "running"
                    );

                    metricsStatus.classList.add(
                        "error"
                    );
                }

                if (diagnosticsStatus) {

                    diagnosticsStatus.textContent =
                        "FAILED";

                    diagnosticsStatus.classList.remove(
                        "running"
                    );

                    diagnosticsStatus.classList.add(
                        "error"
                    );
                }

                if (validationStatus) {

                    validationStatus.textContent =
                        "FAILED";

                    validationStatus.classList.remove(
                        "running"
                    );

                    validationStatus.classList.add(
                        "error"
                    );
                }

                if (outputStatus) {

                    outputStatus.textContent =
                        "Inference failed";
                }

                if (warningBanner) {

                    warningBanner.textContent =
                        error.message;

                    warningBanner.classList.add(
                        "visible"
                    );
                }

                addLog(
                    `ERROR: ${error.message}`
                );
            }

            finally {

                if (scanOverlay) {

                    scanOverlay.classList.remove(
                        "active"
                    );
                }

                if (clearBtn) {

                    clearBtn.disabled =
                        false;
                }

                updateGenerateButton();
            }
        }
    );
}


// ============================================================
// RESET
// ============================================================

if (clearBtn) {

    clearBtn.addEventListener(
        "click",
        function (event) {

            event.preventDefault();
            event.stopPropagation();

            b10File = null;
            qaFile = null;

            b2File = null;
            b3File = null;
            b4File = null;

            [
                b10Input,
                qaInput,
                b2Input,
                b3Input,
                b4Input
            ].forEach(
                input => {

                    if (input) {
                        input.value =
                            "";
                    }
                }
            );

            [
                b10UploadZone,
                qaUploadZone,
                b2UploadZone,
                b3UploadZone,
                b4UploadZone
            ].forEach(
                zone => {

                    if (zone) {

                        zone.classList.remove(
                            "file-selected"
                        );
                    }
                }
            );

            if (b10FileLabel) {

                b10FileLabel.textContent =
                    "Required • .TIF / .TIFF";
            }

            if (qaFileLabel) {

                qaFileLabel.textContent =
                    "Optional • recommended";
            }

            if (b2FileLabel) {

                b2FileLabel.textContent =
                    "Optional • SR_B2";
            }

            if (b3FileLabel) {

                b3FileLabel.textContent =
                    "Optional • SR_B3";
            }

            if (b4FileLabel) {

                b4FileLabel.textContent =
                    "Optional • SR_B4";
            }

            if (fileName) {

                fileName.textContent =
                    "—";
            }

            resetResults();

            updateGenerateButton();

            if (logBox) {

                logBox.innerHTML =
                    `
                    <p>&gt; System reset.</p>
                    <p>&gt; Awaiting Landsat ST_B10 imagery.</p>
                    `;
            }
        }
    );
}


// ============================================================
// REVEAL ANIMATION
// ============================================================

const revealElements =
    document.querySelectorAll(
        ".reveal"
    );

if (
    "IntersectionObserver"
    in window
) {

    const revealObserver =
        new IntersectionObserver(
            entries => {

                entries.forEach(
                    entry => {

                        if (
                            entry.isIntersecting
                        ) {

                            entry.target
                                .classList
                                .add(
                                    "visible"
                                );
                        }
                    }
                );
            },
            {
                threshold:
                    0.12
            }
        );

    revealElements.forEach(
        element => {

            revealObserver.observe(
                element
            );
        }
    );
}


// ============================================================
// NAVBAR
// ============================================================

window.addEventListener(
    "scroll",
    function () {

        const navbar =
            document.querySelector(
                ".navbar"
            );

        if (!navbar) {
            return;
        }

        navbar.style.background =
            window.scrollY > 80
                ? "rgba(5, 10, 25, 0.88)"
                : "rgba(8, 14, 32, 0.66)";
    }
);


// ============================================================
// INITIAL STATE
// ============================================================

resetResults();

updateGenerateButton();

console.log(
    "IRIS frontend ready."
);