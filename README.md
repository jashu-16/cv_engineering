# Computer Vision & Perception Engineering Knowledge Base 🚀

Structured Notion-styled documentation, notes, and implementations for **Machine Learning**, **Deep Learning**, **PyTorch**, **OpenCV**, and **YOLO Object Detection**.

---

## 📚 Curriculum Tracks

1. **Machine Learning Foundations** (12 Levels | 38 Topics)
2. **Deep Learning** (13 Levels | 50 Topics)
3. **PyTorch Mastery** (14 Levels | 54 Topics)
4. **OpenCV Classical Vision** (15 Levels | 55 Topics)
5. **YOLO & Real-Time Perception** (14 Levels | 44 Topics + 5 Milestone Projects)

---

## ⚡ Quick Start

### 1. Set Up the Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Documentation Locally
```bash
mkdocs serve
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Live hot-reloading is enabled!

### 3. Build Static Site
```bash
mkdocs build
```
Builds the production HTML/CSS/JS site into the `site/` directory.

---

## 🛠️ Adding New Notes

1. Place your markdown notes into the relevant track under `docs/`:
   * `docs/01-machine-learning-foundations/`
   * `docs/02-deep-learning/`
   * `docs/03-pytorch/`
   * `docs/04-opencv/`
   * `docs/05-yolo-perception/`
2. Add the path under `nav` in [`mkdocs.yml`](mkdocs.yml).
