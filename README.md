# KoraAi ⚽ AI Video Analysis using OpenCV, PyTorch, and YOLO 🎥📊

![Python](https://img.shields.io/badge/python-3.9.19-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0.84-green.svg)
![YOLOv5](https://img.shields.io/badge/Ultralytics-8.2.48-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-orange.svg)

Use **[OpenCV](https://opencv.org/)** and **[YOLOv5](https://github.com/ultralytics/yolov5)** to analyze football team performance on the pitch, including player speed, team assignments, and ball possession. The YOLOv5 model has been finetuned on a custom dataset from **[Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)**, tailored specifically for recognizing football players, referees, and the ball in various conditions. **[PyTorch](https://pytorch.org/)** was used for fine-tuning the model.

## 🖥️ Project Structure

- **🎥camera_estimator/**: Contains code for estimating and displaying the movement of the camera on the video.
- **🕹️dev_nb/**: A Jupyter notebook used for experimenting with color clustering in player images using K-means to determine the jersey color.
- **🧠models/**: Stores the finetuned YOLOv5 models.
- **⚽player_ball_assigner/**: Responsible for determining which player has the ball.
- **📏speed_and_distance_estimator/**: Estimates the speed and the distance covered by the players.
- **📑stubs/**: Stores various useful constants and thresholds to increase code efficiency.
- **👥team_assigner/**: Assigns players to their respective teams based on jersey color.
- **🔍trackers/**: Tracks objects in the video, assigning IDs to each tracked object.
- **🚀training/**: Contains a YOLO model, yolov5su, and a notebook used for finetuning the model using the RoboFlow dataset.
- **🤖utils/**: Includes utilities to read from and save to video files.
- **🧮view_transformer/**: Manages perspective transformations of the video feed.
- **✅main.py**: The main executable script for the project.
- **🛠️requirements.txt**: Specifies all dependencies required to recreate the project's environment.

## 🛠️ Installation

### 1. Clone this repository:
    git clone https://github.com/Ahmed-Abbas-2077/KoraAi.git
    cd KoraAi/

### 2. Install the required packages:
    pip install -r requirements.txt

## 🚀 usage:

### provide the input and output paths you want inside of main.py and run:
    python main.py
This will start the process of video analysis, where the performance of football teams will be assessed and saved.


## 💡 Contributing

Contributions to this project are welcome. Please create a pull request with your proposed changes or improvements.


## 🔓 License

This project is licensed under the **MIT** License - see the **LICENSE** file for details.


## 🎉 Acknowledgments
- **KoraAi** has been inspired by this engaing 4-hour youtube [video](https://www.youtube.com/watch?v=neBZ6huolkg)
- [Ultralytics](https://ultralytics.com/), for their versatile YOLO models which are foundational to Kora.ai's object detection tasks.
- [Roboflow](https://roboflow.com/), for providing the dataset and tools necessary for model training.
- [OpenCV](https://opencv.org/), for their incredible computer vision library.
- [PyTorch](https://pytorch.org/), for the powerful deep learning platform that enabled fine-tuning of the model.
