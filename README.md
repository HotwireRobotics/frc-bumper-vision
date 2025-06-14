# FRC Bumper Vision

A lightweight AI-based system to detect FRC robots by identifying their bumpers using YOLOv8 and Limelight vision. Built to support real-time obstacle avoidance with dynamic rerouting via AD\* pathfinding during both autonomous and teleop modes.

## 🔧 Setup

```bash
git clone https://github.com/hotwirerobotics/frc-bumper-vision.git
cd frc-bumper-vision
python -m venv .venv
.venv\Scripts\activate  # For Windows
pip install -r requirements.lock
```

## 🚀 Project Highlights

* Real-time bumper detection via Limelight 2+/4
* YOLOv8 object detection, trained on FRC robot bumpers
* Injects detected robot positions into existing AD\* pathfinder
* Seamless integration with WPILib Java + PathPlanner

## 📌 Status

* [x] Virtual environment + test script ready
* [x] Data collection and labeling
* [ ] YOLO training pipeline
* [ ] Full FRC integration

## 👥 Maintained by

[Hotwire Robotics](https://github.com/hotwirerobotics) - 2025 Offseason Project


randy's nightmare
