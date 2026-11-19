<div align="center">

# 🔎 Visual Image Search Engine

### Search images by what's **inside** them — not by filename.

**YOLO11 · Streamlit · Python · JSON Metadata**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO11](https://img.shields.io/badge/Detection-YOLO11-7059C1?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Metadata](https://img.shields.io/badge/Index-JSON-17A589?style=flat-square)
![Workflow](https://img.shields.io/badge/Workflow-Modular-2274A5?style=flat-square)

**One-time object detection → reusable metadata → fast, interactive search**

</div>

---

## 🎯 The idea in 15 seconds

Imagine a folder with hundreds of images. Instead of opening them one by one, ask the app:

> **“Show images containing a dog AND a frisbee.”**  
> **“Find a bus OR a bicycle.”**  
> **“Show photos with at most two people.”**

The application first builds a searchable index using a **pretrained YOLO11 object detector**. Then it searches **saved predictions**, so changing search criteria does **not** require rerunning the detector.

## 🗺️ Project workflow — at a glance

```mermaid
flowchart LR
    A["📁 Image Folder"] --> B["🧠 YOLO11<br/>Object Detection"]
    B --> C["🧾 Detection Records<br/>Class · Box · Confidence · Count"]
    C --> D[("💾 metadata.json")]
    D --> E["🔎 Search Engine<br/>AND / OR · Max Counts"]
    E --> F["🖼️ Interactive Gallery"]
    F --> G["⬇️ Export Results JSON"]

    classDef input fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef model fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px;
    classDef data fill:#D1FAE5,stroke:#059669,color:#065F46,stroke-width:2px;
    classDef search fill:#FEF3C7,stroke:#D97706,color:#92400E,stroke-width:2px;
    classDef result fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px;
    class A input;
    class B model;
    class C,D data;
    class E search;
    class F,G result;
```

**Key architectural decision:** separate **expensive image inference** from **lightweight search on structured metadata**.

### Two ways to use the app

```mermaid
flowchart TD
    U["🚀 Launch Streamlit App"] --> Q{"Do you already have<br/>metadata.json?"}
    Q -->|"No — first time"| I["📂 Select image directory"]
    I --> M["⚙️ Run YOLO11 inference"]
    M --> S["💾 Save metadata.json"]
    Q -->|"Yes — reuse results"| L["📥 Load metadata.json"]
    S --> X["🔍 Search classes & counts"]
    L --> X
    X --> R["🖼️ Browse, highlight & export"]

    classDef action fill:#E0F2FE,stroke:#0284C7,color:#075985,stroke-width:2px;
    classDef decision fill:#FEF3C7,stroke:#D97706,color:#92400E,stroke-width:2px;
    classDef saved fill:#D1FAE5,stroke:#059669,color:#065F46,stroke-width:2px;
    class U,I,M,L,X,R action;
    class Q decision;
    class S saved;
```

> 🟢 **First run:** detect objects and create the index.  
> 🔵 **Later runs:** load the saved index and search immediately — no new detection run.

---

## ✨ Core capabilities

| | Capability | What it means in practice |
| :---: | --- | --- |
| 🧠 | **Pretrained detection** | YOLO11 predicts object classes, bounding boxes and confidence scores |
| 💾 | **Persistent metadata** | Save image-level detections once and reload them as JSON |
| 🔀 | **Boolean search** | Match **any** selected classes (OR) or **all** selected classes (AND) |
| 🔢 | **Count filters** | Restrict the **maximum number** of a detected class in an image |
| 🖼️ | **Visual inspection** | Display image cards and toggle bounding-box overlays |
| 🎯 | **Match highlighting** | Highlight selected classes or display all predicted boxes |
| 📤 | **Export** | Download matching image records as `search_results.json` |

### 🔎 How search actually works

For every image, Python checks the detected count of each selected class. A class matches only when it is **present at least once**, and its count does not exceed the optional maximum.

```mermaid
flowchart TD
    Q["🔎 Search: dog AND frisbee"] --> A["Image A<br/>dog: 1 · frisbee: 1"]
    Q --> B["Image B<br/>dog: 2 · frisbee: 0"]
    Q --> C["Image C<br/>dog: 0 · frisbee: 1"]
    A --> Y["✅ Match"]
    B --> N["❌ Not a match"]
    C --> N
    classDef query fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px;
    classDef sample fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
    classDef yes fill:#D1FAE5,stroke:#059669,color:#065F46,stroke-width:2px;
    classDef no fill:#FEE2E2,stroke:#DC2626,color:#991B1B;
    class Q query;
    class A,B,C sample;
    class Y yes;
    class N no;
```

| Search | Matches when… |
| --- | --- |
| `dog OR cat` | At least one of the classes is detected |
| `dog AND frisbee` | **Both** classes are detected in the same image |
| `person` + max count `2` | **One or two** people are detected (not zero) |

**Implementation:** `any(class_matches.values())` for OR; `all(class_matches.values())` for AND. This is a **metadata filter**, not another YOLO prediction.

---

## 🧩 Modular software architecture

```mermaid
flowchart TB
    UI["🖥️ app.py<br/>Streamlit UI · Session State · Search · Results"] --> INF["🧠 src/inference.py<br/>YOLOv11Inference"]
    UI --> UT["💾 src/utils.py<br/>JSON Save / Load · Class Counts"]
    INF --> CF["⚙️ src/config.py<br/>YAML Loader"]
    CF --> YML["📋 configs/default.yaml<br/>Confidence · Image Extensions"]
    INF --> MODEL["📦 yolo11m.pt<br/>Pretrained Weights"]
    INF --> UT

    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px;
    classDef service fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef config fill:#FEF3C7,stroke:#D97706,color:#92400E;
    class UI ui;
    class INF,UT service;
    class CF,YML,MODEL config;
```

<details>
<summary><strong>📂 Expand the project folder structure</strong></summary>

```text
Yolo_image_search/
├── app.py                 # Streamlit app, search and visualization
├── src/
│   ├── inference.py       # YOLO model and detection-to-metadata pipeline
│   ├── utils.py           # Save/load JSON and derive search options
│   └── config.py          # Load YAML configuration
├── configs/
│   └── default.yaml       # Confidence threshold and extensions
├── data/                  # Image data / generated metadata
├── test/                  # Streamlit exploration (not automated tests)
├── yolo11m.pt             # Pretrained YOLO11 medium weights
├── requirements.txt
└── instruction.txt
```

</details>

## 🧾 Why JSON is central to the design

The output of YOLO is transformed into a reusable record, rather than being used only to draw a rectangle on one picture.

```mermaid
flowchart LR
    P["📷 One Image"] --> D["🔍 Detected Objects"]
    D --> K["class<br/>confidence<br/>bbox [x1,y1,x2,y2]"]
    K --> J[("💾 JSON Record")]
    J --> Z["🔎 Future Searches"]
    classDef one fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
    classDef two fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95;
    classDef three fill:#D1FAE5,stroke:#059669,color:#065F46;
    class P one;
    class D,K two;
    class J,Z three;
```

<details>
<summary><strong>🔍 View a sample metadata record</strong></summary>

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

*Illustrative record, not a measured output from this repository.* The `count` field on each detection stores the **total count of its class in that image**, while `class_counts` summarizes counts by class. Bounding boxes use pixel-space XYXY coordinates.

</details>

**Important:** the JSON stores **paths to images**, not the image files themselves. Sharing metadata with another machine also requires accessible images and valid paths.

---

## 🚀 Run locally

**1 · Create an environment**

```bash
conda create -n yolo_image_search python=3.11 -y
conda activate yolo_image_search
```

**2 · Install dependencies**

```bash
pip install -r requirements.txt
```

**3 · Start the application**

From the **`Yolo_image_search/`** folder (required by the current relative configuration path):

```bash
streamlit run app.py
```

Open **http://localhost:8501**.

**4 · Choose an entry point**

| 🆕 Process new images | ♻️ Load existing metadata |
| --- | --- |
| Enter image directory and model weights path | Enter existing `metadata.json` path |
| Click **Start Inference** | Click **Load Metadata** |
| Detection runs and metadata is saved | Predictions are reused without inference |

Then select classes, choose OR/AND, optionally set maximum counts, click **Search Images**, adjust display options, and export selected records.

> ⚠️ **Device note:** `src/inference.py` currently defaults to `device='cuda'`. To run new inference on a CPU-only machine, change the default to `device='cpu'`. Reloading existing JSON does not run YOLO. See `instruction.txt` for the original GPU installation guidance.

---

## 🛠️ Technology and implementation details

| Layer | Technology |
| --- | --- |
| Object detection | **Ultralytics YOLO11**, pretrained `yolo11m.pt` |
| Deep learning runtime | **PyTorch** |
| Web application | **Streamlit** |
| Bounding-box visualization | **Pillow (PIL)** |
| Data exchange | **JSON** |
| Configuration | **PyYAML** |
| Implementation | **Python**, separated into UI, inference and utilities |

The current configuration uses **0.30** detection confidence and scans `.jpg`, `.jpeg` and `.png` files in the supplied folder (not recursively).

<details>
<summary><strong>⚙️ Engineering notes and limitations</strong></summary>

- This is a **pretrained inference and search application**, not a YOLO training or fine-tuning project.
- Images are processed individually; no batched/distributed indexing performance is claimed.
- Predicted classes are model-generated labels, **not validated ground truth**.
- The saved `image_path` values must remain resolvable when viewing results.
- Object-count controls are **upper bounds**, not exact counts.
- `test/streamlit_basics.py` is a Streamlit experiment, not an automated integration test suite.
- Exports can support human review and dataset curation, but predicted detections are **pseudo-labels**, not reviewed annotations.

</details>

## 💡 What this project demonstrates

**Model integration** · **structured inference outputs** · **persistent metadata** · **modular Python** · **Boolean retrieval** · **interactive application delivery**

The main takeaway is not only detecting objects. It is building a usable system around model predictions: **index once, search many times, inspect results, and export reusable data**.

---

<div align="center">

Part of [**End2End Computer Vision Project**](https://github.com/MBamdad/End2End_Computer_Vison_Project)

**YOLO11 → JSON → Search → Visualization → Export**

</div>
