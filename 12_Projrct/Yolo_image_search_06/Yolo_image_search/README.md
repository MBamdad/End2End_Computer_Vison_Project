# Computer Vision–Powered Image Search | YOLO11 + Streamlit

A modular, local-first computer vision application that makes image collections searchable by **what they contain**, rather than by filenames.

The application combines pretrained **Ultralytics YOLO11 object detection**, reusable **JSON metadata**, class-aware **Boolean search**, and a **Streamlit** interface. The main engineering idea is to **run object detection once, persist structured predictions, and perform subsequent searches without rerunning the model**.

## What the application does

- **Index image collections:** run YOLO11 inference over a directory of images.
- **Extract structured detections:** record each object's predicted class, confidence, bounding box, and per-image class counts.
- **Reuse indexed results:** save predictions to `metadata.json`, then reload them in later sessions without inference.
- **Search by content:** find images matching *any* selected class (**OR**) or *all* selected classes (**AND**).
- **Filter by object quantity:** optionally set a maximum count per selected class; a matching class must still appear at least once.
- **Inspect predictions:** browse image cards, show or hide bounding boxes, and highlight matching object classes.
- **Export a subset:** download the matching records as `search_results.json` for further review or processing.

### Example searches

| Search | Expected behavior |
| --- | --- |
| `dog OR cat` | Return images containing at least one dog **or** one cat |
| `dog AND frisbee` | Return images containing **both** objects |
| `person`, max count = 2 | Return images with **one or two** detected people |
| `bus AND person`, bus max = 3 | Return images with both classes and at most three detected buses |

Search results reflect **model predictions**, not verified ground-truth labels. Recognition quality depends on the pretrained model, confidence threshold, and image domain.

## End-to-end architecture

```text
                         IMAGE INDEXING (run as needed)
  Local image directory
          |
          v
  YOLO11 pretrained weights
          |
          v
  Object detection (class + confidence + XYXY bounding box)
          |
          v
  Per-image structured records + class counts
          |
          v
  JSON persistence: metadata.json
          |
          +-----------------------------------------------+
                                                          |
                        INTERACTIVE SEARCH                |
  Streamlit interface <--- Load existing JSON <-----------+
          |
          v
  Discover available classes and observed count options
          |
          v
  Search criteria: selected classes + AND/OR + max counts
          |
          v
  Filter stored detection records in Python (no YOLO call)
          |
          v
  Image gallery + optional bounding boxes + class counts
          |
          v
  Download matching records: search_results.json
```

**Why this separation matters:** inference is typically the expensive stage, whereas searching an already indexed collection is a lightweight operation over structured records. Keeping the two paths separate makes iteration and repeated filtering more efficient.

## Code organization

```text
Yolo_image_search/
├── app.py                 # Streamlit application and search/results UI
├── configs/
│   └── default.yaml       # Confidence threshold and supported extensions
├── src/
│   ├── config.py          # YAML configuration loader
│   ├── inference.py       # YOLO11 model loading and image/directory inference
│   └── utils.py           # Metadata save/load and class/count aggregation
├── data/                  # Image data and generated metadata
├── test/                  # Streamlit experimentation
├── requirements.txt       # Python dependencies
├── instruction.txt        # Environment setup notes
└── yolo11m.pt             # YOLO11 medium pretrained weights
```

### Module responsibilities

| Component | Responsibility |
| --- | --- |
| `YOLOv11Inference` | Load model weights; predict over images; normalize detections into Python dictionaries |
| `process_directory()` | Discover supported image files; process them independently; skip/log individual image failures |
| `save_metadata()` / `load_metadata()` | Persist and restore records as JSON |
| `get_unique_classes_counts()` | Build the available search classes and count options |
| `app.py` | Orchestrate processing/loading, search controls, result cards, and export |
| Streamlit `session_state` | Preserve metadata, search parameters, matches, and display preferences across interface reruns |

## Metadata: the interface between ML and the search application

Each image is represented as a structured record. An illustrative example:

```json
{
  "image_path": "data/raw/images/example.jpg",
  "detections": [
    {
      "class": "dog",
      "confidence": 0.94,
      "bbox": [38.0, 51.0, 218.0, 285.0],
      "count": 2
    },
    {
      "class": "dog",
      "confidence": 0.89,
      "bbox": [242.0, 60.0, 411.0, 294.0],
      "count": 2
    }
  ],
  "total_objects": 2,
  "unique_class": ["dog"],
  "class_counts": {"dog": 2}
}
```

- `bbox` contains pixel-space coordinates in **XYXY** order: `[xmin, ymin, xmax, ymax]`.
- `confidence` is the model's prediction confidence.
- `count` on each detection reflects that class's **total number in the image**, not the index of that individual detection.
- `class_counts` is the convenient per-image summary used by the search logic.
- `image_path` points to the underlying image; the actual image bytes are **not embedded** in the JSON.

> **Important:** moving or sharing `metadata.json` does not automatically transfer the corresponding images. The referenced image paths must remain accessible or be remapped on the recipient's machine.

## Search logic

The application first computes whether each selected class satisfies its count condition:

```python
# Conceptual search logic
n = class_counts.get(selected_class, 0)
class_matches = n >= 1 and (max_count is None or n <= max_count)

# Combine per-class results
matched = any(matches) if mode == "OR" else all(matches)
```

The current UI's count control is a **maximum count**, not an exact-count or minimum-count selector. Classes with zero detections do not match.

This search is performed against saved predictions in Python, **not by querying YOLO11 again**.

## Running locally

### 1. Create an environment

Python 3.11 is a suitable starting point for the provided setup instructions:

```bash
conda create -n yolo_image_search python=3.11 -y
conda activate yolo_image_search
```

### 2. Install dependencies

From the project directory containing `requirements.txt`:

```bash
pip install -r requirements.txt
```

For GPU execution, install a compatible **CUDA-enabled PyTorch build** and ensure an NVIDIA GPU and its driver are available. See `instruction.txt` for the recorded installation commands.

**Current implementation requirement:** `YOLOv11Inference` defaults to `device='cuda'` and moves the model to that device. For a CPU-only environment, change that default to `device='cpu'` in `src/inference.py` before processing new images. Loading existing metadata does not require running the YOLO model.

### 3. Launch Streamlit

Run this command **from the `Yolo_image_search` directory** so that the relative `configs/default.yaml` path resolves:

```bash
streamlit run app.py
```

Streamlit normally serves the application at `http://localhost:8501`. You may choose another port:

```bash
streamlit run app.py --server.port 8080
```

### 4. Index images or reload existing metadata

**Option A — Process new images**

1. Select **Process new images**.
2. Enter the image directory path.
3. Keep `yolo11m.pt` as the weights path or provide another compatible model path.
4. Click **Start Inference**.
5. Reuse the generated `metadata.json` in future sessions.

**Option B — Load existing metadata**

1. Select **Load existing metadata**.
2. Enter the path to an existing `metadata.json`.
3. Click **Load Metadata**.
4. Search and browse images without running detection again.

The project reads the model confidence threshold from `configs/default.yaml` (currently **0.30**) and discovers `.jpg`, `.jpeg`, and `.png` files at the supplied directory level.

### 5. Search and export

Select classes, choose AND/OR, optionally set per-class maximum counts, then click **Search Images**. The results view supports bounding-box display, class highlighting, and an adjustable gallery layout. Use **Download Results (JSON)** to export the matched metadata.

## Technology stack

- **Computer vision:** Ultralytics YOLO11, PyTorch
- **Application/UI:** Streamlit
- **Image annotation display:** Pillow (PIL)
- **Configuration:** YAML / PyYAML
- **Data interchange:** JSON
- **Implementation:** modular Python

## Engineering scope and limitations

This repository demonstrates a **pretrained-inference application**, rather than YOLO model training or fine-tuning. It does not claim improved detection accuracy or production-scale throughput.

Current implementation details to keep in mind:

- Search quality is limited by the detector's predicted objects and its pretrained class vocabulary.
- Detection uses a configured confidence threshold; results below it are filtered out.
- Processing is currently **per image**; it is not a benchmarked batched or distributed indexing system.
- The directory scan uses configured extensions and is not recursively traversing nested folders.
- Metadata stores image paths, so a portable shared dataset requires maintaining or rewriting those paths.
- Streamlit presentation relies on image files still being available on disk.
- The existing `test/streamlit_basics.py` is exploratory Streamlit code, **not** an automated model-quality or integration test suite.
- Exported predictions can support **human review and dataset curation**, but are **pseudo-labels**, not automatically verified annotations.

## Why this project matters

The work connects a computer vision model to the rest of an application workflow:

1. **Inference engineering** — transform raw model outputs into a stable schema.
2. **Data engineering** — save, reload, and reuse predictions as machine-readable metadata.
3. **Application logic** — implement class-aware Boolean and quantity-based search.
4. **Interface integration** — connect Python processing to interactive controls and results visualization.
5. **Reproducibility and maintainability** — organize configuration, model inference, utilities, and UI into distinct components.

The central design principle is **decoupling model computation from user interaction**: run detection when the dataset changes, then search the indexed results repeatedly and efficiently.

---

**Project context:** Part of the [End2End Computer Vision Project](https://github.com/MBamdad/End2End_Computer_Vison_Project), focused on practical, modular computer vision application development.
