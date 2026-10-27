# Lauren Cohan 3D face reconstruction

This project creates a textured, three-dimensional head from the supplied
single photograph and exports a self-contained glTF 2.0 binary. The supplied
workspace asset is a 1600 x 2000 JPEG whose filename ends in `.glb`; it is not
a GLB model. It has already been copied byte-for-byte to
`input/assigned_face.jpg`. Keep the original unchanged.

## Method and limitations

The pipeline uses OpenCV's frontal-face detector to prepare a crop, then maps
that exact crop onto the front half of a parametric head surface. It adds
procedural nose/cheek depth, ears, and a neck. The inspected environment is
WSL2/Linux with Python 3.13.2. No NVIDIA/CUDA tools were detected; the pipeline
runs on CPU and does not download model weights or upload the photograph.

This is a practical fallback, not a learned 3D face reconstruction. The source
contains only one view, so the profile and rear geometry are approximations;
the original photo texture is visible mainly from the front. The inspection
renders show a narrow texture sliver at profile angles and simplified ears,
neck, and hair. Facial landmarks and hair geometry are not independently
reconstructed. DECA is a face-specific single-image method, but its official
implementation requires PyTorch/PyTorch3D and downloaded model data, and its
code and model are licensed for non-commercial scientific research. The
original Microsoft Deep3DFaceReconstruction repository says it is no longer
maintained and requires separately obtained Basel Face Model data. Neither was
installed or run here. See the [DECA repository](https://github.com/yfeng95/DECA)
and [Deep3DFaceReconstruction repository](https://github.com/microsoft/Deep3DFaceReconstruction)
for their current instructions and terms before choosing that route.

Blender is not required by the scripts. The Khronos glTF Validator was run;
the local `trimesh` parser is also used as a separate structural check.

## Requirements

- Linux, macOS, or Windows
- Python 3.10 or newer
- About 1 GB of free disk space for the environment and outputs
- No GPU required
- Node.js and npm are needed only for the optional Khronos Validator step

The listed packages are pinned in `requirements.txt`. From this folder, create
an isolated environment and install them:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run each stage

1. The supplied photo is already at `input/assigned_face.jpg`. To use a new
   reference, place it there without changing the original workspace file.
2. Inspect dimensions and produce a face crop:

   ```bash
   python scripts/inspect_image.py
   ```

   Review `reports/face_crop.jpg` and `reports/image_inspection.json`. The
   script stops with an explicit error if it cannot detect a face.
3. Generate the parametric surface and texture atlas:

   ```bash
   python scripts/reconstruct_face.py
   ```

   This writes intermediate assets under `models/`.
4. Export the GLB:

   ```bash
   python scripts/export_glb.py
   ```

5. Run Khronos' official validator (optional Node.js prerequisite):

   ```bash
   npm install --prefix .validator --no-save gltf-validator@2.0.0-dev.3.10
   node scripts/validate_khronos.cjs
   ```

   This writes `reports/khronos_validation.json`. The ignored `.validator/`
   directory is local tooling, not part of the deliverable.
6. Validate its container, embedded texture, mesh/material counts, parser
   result, and 2,000,000-byte limit:

   ```bash
   python scripts/validate_glb.py
   ```

   The JSON report is written to `reports/validation.json` and includes the
   Khronos report summary if that step was run.

7. Render the five required viewpoints for a quick software inspection:

   ```bash
   python scripts/render_inspection.py
   ```

   The front, profile, and three-quarter images are in `reports/views/`; the
   contact sheet is `reports/multi_view_inspection.png`. These are inspection
   renders, not evidence that the side geometry matches an unavailable side
   reference.

The final local file is `output/lauren_cohan.glb`. The crop, reports, and five
generated angle views are in `reports/`. See
[`reports/implementation_status.md`](reports/implementation_status.md) for the
latest executed results and the remaining hosting step. Inspect the GLB in a
glTF-compatible viewer before submission.

## Public hosting

Publishing is intentionally manual. Do not upload the input photo, temporary
models, or this project to a public repository unless you explicitly choose to.
For a GitHub option, create a public repository, upload only
`output/lauren_cohan.glb`, then copy its actual raw-file HTTPS URL. Do not
guess the owner, repository, branch, or URL. Check the host supports CORS.

Test the actual URL (not a viewer-page URL) with:

```bash
python scripts/test_public_url.py "https://your-actual-host/path/model.glb"
```

The script requires HTTPS, a successful response, a GLB binary response, a
permissive CORS header, and valid GLB structure. It makes no request until an
actual URL is supplied.

## Common issues

- **Face not detected:** use the clear, oriented assigned photograph; the
  script will not guess a crop. A different crop/detector must be reviewed
  before using it.
- **Wrong file type:** the supplied workspace asset is JPEG data despite the
  `.glb` suffix. Copy it to `input/assigned_face.jpg`; do not treat it as a
  finished model.
- **GLB exceeds the size limit:** inspect `models/face_atlas.jpg`; the export
  currently uses a compressed JPEG atlas. Reduce crop dimensions or JPEG
  quality only if needed, then export and validate again.
- **CORS test fails:** use a static host whose raw-file response includes
  `Access-Control-Allow-Origin: *` or an allowed origin. A web viewer URL is
  not a direct GLB URL.
