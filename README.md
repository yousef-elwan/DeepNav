# DeepNav: Deep Learning-based Inertial Navigation System for UAVs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An innovative, hierarchical deep learning system designed to provide reliable, real-time navigation for Unmanned Aerial Vehicles (UAVs) during GNSS-denied periods, relying solely on low-cost inertial sensor data. This project was developed as a final thesis for an Aerospace Engineering degree.

---

### Table of Contents
1.  [About The Project](#about-the-project)
2.  [System Architecture](#system-architecture)
3.  [Key Features](#key-features)
4.  [Results](#results)
5.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
6.  [Usage](#usage)
    * [Data Preprocessing](#data-preprocessing)
    * [Training the Models](#training-the-models)
    * [Evaluation](#evaluation)
    * [Using the GUI](#using-the-gui)
7.  [License](#license)
8.  [Contact](#contact)
9.  [Acknowledgments](#acknowledgments)

---

### About The Project

UAVs face a significant challenge in maintaining accurate navigation when Global Navigation Satellite System (GNSS) signals are unavailable, such as in urban canyons, indoors, or during intentional jamming. This limitation severely restricts their operational capabilities.

This project addresses this problem by developing a robust alternative navigation system. It leverages deep learning to estimate the UAV's state (attitude, velocity, and position) using only data from its onboard Inertial Measurement Unit (IMU).

The core of this project is an innovative hierarchical cascade architecture of three interconnected Long Short-Term Memory (LSTM) networks. Each network is trained for a specific predictive task, creating a dependency chain that mimics the physical nature of motion: attitude estimation is required for velocity prediction, and both are necessary for accurate position tracking. The models were trained and validated on real-world flight data from quadcopters, sourced from the public "Flight Review" platform.

The system demonstrates promising performance, achieving a **Median Maximum Position Error (MPE) of 28 meters** on validation flight data, showcasing its ability to maintain acceptable drift over extended periods without GNSS.

---

### System Architecture

The navigation system is built on a hierarchical (cascade) model of three distinct LSTM networks:

1.  **Attitude Network (DeepNav-q)**:
    * **Input**: Raw time-series data from inertial sensors (accelerometer, gyroscope, magnetometer).
    * **Output**: Predicts the change in attitude, represented as a quaternion ($\Delta q$).
    * **Logic**: This network forms the foundation of the hierarchy. An accurate attitude is crucial for correctly interpreting acceleration measurements in the next stage.

2.  **Velocity Network (DeepNav-v)**:
    * **Input**: Raw sensor data combined with the predicted attitude output from the first network.
    * **Output**: Predicts the change in velocity ($\Delta v$).
    * **Logic**: By receiving attitude context, this network can more accurately differentiate between gravitational and linear acceleration to predict velocity changes.

3.  **Position Network (DeepNav-p)**:
    * **Input**: Raw sensor data combined with the outputs from both the attitude and velocity networks.
    * **Output**: Predicts the change in position ($\Delta p$).
    * **Logic**: As the final stage, this network integrates all prior motion information to produce the most accurate possible position estimate, completing the navigation state.

This sequential structure explicitly models the physical dependencies of motion, providing a rich, growing context for each stage of the estimation process.

---

### Key Features

* **Innovative Hierarchical Architecture**: A unique three-tiered LSTM structure that sequentially estimates attitude, velocity, and position, improving overall system accuracy.
* **End-to-End Deep Learning**: The system learns the complex, non-linear dynamics of flight directly from data, avoiding the need for complex analytical models.
* **Real-World Data Training**: The models are trained and validated on a large dataset of real quadcopter flight logs from the "Flight Review" platform, ensuring they learn from diverse and realistic scenarios.
* **Custom Analysis GUI**: A powerful Graphical User Interface (GUI) built with PyQt5 and pyqtgraph for in-depth, visual analysis of flight data. It allows for 3D trajectory visualization, 2D path comparison, and time-series plotting of all navigation state variables and their errors.
* **Robust EKF Integration**: The project includes a full 22-state Extended Kalman Filter (EKF) implementation, which is used as a ground truth reference during training and for performance comparison.
* **Custom Loss Functions**: Utilizes specialized loss functions, such as a weighted MAE for velocity/position and quaternion angular distance for attitude, to optimize for physically meaningful accuracy.

---

### Results

The models were evaluated on a validation dataset unseen during training. After a GNSS outage of over 13 minutes, the system achieved the following Mean Absolute Error (MAE):

* **Euler Angles**: 0.2°
* **Velocity**: 3.3 m/s
* **Position**: 15 m

Crucially, the system demonstrated strong resistance to the unbounded error growth typical of traditional inertial navigation systems. The **Median Maximum Position Error (Median MPE)** across all validation flights was **28 meters**.

While the results are promising, the analysis also identified a challenge with **overfitting**, where the models performed significantly better on training data than on validation data. This indicates an opportunity for future improvement through advanced regularization techniques.

---

### Getting Started

To get a local copy up and running, follow these simple steps.

#### Prerequisites

* Python 3.8+
* pip package manager

#### Installation

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/yousef-elwan/DeepNav.git](https://github.com/yousef-elwan/DeepNav.git)
    cd DeepNav
    ```

2.  **Create a virtual environment (recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install required packages**
    ```sh
    pip install -r requirements.txt
    ```

---

### Usage

The project is structured into several components, allowing for modular execution.

#### Data Preprocessing

The `create_dataset.py` script within each model's `preprocessing` directory is used to process raw flight log CSVs into windowed datasets suitable for LSTM training.

* Run `DeepNav-q/preprocessing/create_dataset.py` for the attitude model.
* Run `DeepNav-v/preprocessing/create_dataset.py` for the velocity model.
* Run `DeepNav-p/preprocessing/create_dataset.py` for the position model.

#### Training the Models

You can train each of the three hierarchical models by running their respective main scripts:

```sh
# Train the Attitude Model (DeepNav-q)
python DeepNav-q/DeepNav.py --mode 0 --epochs 50 --batch_size 1024 --lr 0.001

# Train the Velocity Model (DeepNav-v)
python DeepNav-v/DeepNav.py --mode 0 --epochs 50 --batch_size 3500 --lr 0.001

# Train the Position Model (DeepNav-p)
python DeepNav-p/DeepNav.py --mode 0 --epochs 50 --batch_size 1024 --lr 0.001
```

* `--mode`: Session Mode (0=Fresh, 1=Resume, 2=Evaluate).
* `--epochs`: Number of training epochs.
* See the `argparse` section in each script for more options.

#### Evaluation

To evaluate a trained model, run the main script in `Evaluate` mode (`--mode 2`), which will load the latest weights and run post-processing.

#### Using the GUI

The analysis GUI can be launched to visualize flight data CSVs that contain EKF, LSTM, and custom EKF outputs.

```sh
python "GUI & EKF/GUI.py"
```

From the GUI, you can:
1.  Click **"Import CSV"** to load a flight log file.
2.  Use the simulation controls in the **"3D View"** tab to play, pause, and scrub through the flight.
3.  Switch between tabs to view detailed 2D plots for position, velocity, orientation, and sensor data.

---

### License

Distributed under the MIT License. See `LICENSE` for more information.

---

### Contact

Yousef Elwan - [elwanyousef1@gmail.com](mailto:elwanyousef1@gmail.com)

Project Link: [https://github.com/yousef-elwan/DeepNav](https://github.com/yousef-elwan/DeepNav)

---

### Acknowledgments
* This project is based on the B.Sc. thesis submitted to the Higher Institute for Applied Sciences and Technology (HIAST).
* Special thanks to the supervisors and everyone who supported this work.
* The "Flight Review" data platform community.
* The open-source community for tools like TensorFlow and PyQt.
