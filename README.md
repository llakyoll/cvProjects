# 🎯 Computer Vision Projects

A collection of modular, self-contained computer vision projects — detection, counting, and tracking experiments built with Python, OpenCV, and Ultralytics YOLO.

Each project lives in its own folder with its own README, requirements, and demo. Clone the repo, pick a folder, and run.

> 🚀 Looking for my flagship work? Check out [rtsp-multicam-tracker](https://github.com/llakyoll/rtsp-multicam-tracker) — real-time multi-camera inference on RTSP streams.

---

## 📂 Projects

| Project | Description | Stack | Demo |
|---|---|---|---|
| [vehicle-counting](./vehicle-counting) | Counts vehicles crossing a virtual line in video streams, with per-direction totals | YOLO, OpenCV | — |
| [people-counting](./people-counting) | Real-time people detection and counting with entry/exit zone logic | YOLO, OpenCV | — |

---

## 🗂️ Project Structure

Every project follows the same layout:

```
project-name/
├── README.md          # What it does, how to run it, sample output
├── requirements.txt   # Pinned dependencies
├── main.py            # Entry point
├── src/               # Modules (detection, counting, utils)
└── assets/            # demo.gif, sample frames
```

## ⚙️ Quick Start

```bash
git clone https://github.com/llakyoll/cvProjects.git
cd cvProjects/<project-name>
pip install -r requirements.txt
python main.py --source path/to/video.mp4
```

---

## 🧰 Common Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics_YOLO-111F68?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 📬 Contact

**Ahmed Akyol** — Computer Vision Engineer
[GitHub](https://github.com/llakyoll) · [LinkedIn](https://www.linkedin.com/in/ahmed-akyol-84766622b/) · ahmedakyll@gmail.com

*If a project here is useful to you, a ⭐ helps others find it too.*
