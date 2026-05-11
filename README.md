# 🚦 Density-Based Smart Traffic Light System

A Final Year Project (FYP) that uses real-time vehicle detection to intelligently control traffic signal timings based on road density. Built with YOLOv8 for computer vision, Raspberry Pi for hardware control, and a React + Flask web dashboard for live monitoring.

![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-purple?style=flat-square&logo=github&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=flat-square&logo=raspberrypi&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)

---

## 📋 Project Overview

Traditional traffic lights use fixed timers regardless of actual road conditions. This system solves that by:

- Detecting the number of vehicles on each road in real time using **YOLOv8**
- Calculating road density and **prioritizing the signal** for the busiest lane
- Controlling physical **LED traffic lights** via Raspberry Pi GPIO
- Displaying live traffic data on a **React web dashboard**

---

## 🏗️ System Architecture

```
Camera Feed
    ↓
YOLOv8 (Vehicle Detection on Raspberry Pi)
    ↓
Density Calculation Engine (Python)
    ↓
Signal Switching Logic → GPIO → LED Traffic Lights
    ↓
Flask API → React Dashboard (Live Monitoring)
```

---

## 📋 Features

- **Real-Time Vehicle Detection** — YOLOv8 detects and counts vehicles from camera input
- **Density-Based Signal Control** — Higher vehicle count = longer green light duration
- **Automated Signal Switching** — Python logic controls which road gets priority
- **GPIO Control** — Raspberry Pi drives physical LED traffic light hardware
- **Live Web Dashboard** — React + Tailwind UI displays real-time traffic density data
- **Flask REST API** — Backend serves live data from Raspberry Pi to the frontend
- **Optimized for Low-End Hardware** — System tuned to run efficiently on Raspberry Pi

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Computer Vision | YOLOv8 (Ultralytics) |
| Hardware | Raspberry Pi (GPIO for LED control) |
| Backend | Python, Flask |
| Frontend | React.js, Tailwind CSS |
| Communication | REST API (Flask ↔ React) |
| File Transfer | SCP (Secure Copy Protocol) |

---

## 📦 Getting Started

### Prerequisites

**On Raspberry Pi:**
- Raspberry Pi (any model with GPIO)
- Python 3.8+
- Camera module or USB webcam
- LED traffic light circuit connected to GPIO pins

**On Local Machine:**
- [Node.js](https://nodejs.org/) (v16+)
- npm

---

## 🚀 Running the Project

### On Raspberry Pi

**1. Navigate to project directory**
```bash
cd /home/pi/traffic_project
```

**2. Activate virtual environment**
```bash
source venv/bin/activate
```

**3. Run LED test (verify hardware is connected)**
```bash
python test_led2.py
```

**4. Run the main traffic system**
```bash
python traffic_system.py
```

---

### On Local Machine (Web Dashboard)

**1. Start the React frontend**
```bash
cd frontend
npm install
npm start
```
Frontend runs at `http://localhost:3000`

**2. Start the Flask backend**
```bash
cd backend
python app.py
```
Backend runs at `http://localhost:5000`

---

### Transferring Files to Raspberry Pi (Same Network)

**Transfer a single file:**
```bash
scp FILE.py pi@192.168.43.173:/home/pi/
```

**Transfer an entire folder:**
```bash
scp -r folder_name/ pi@192.168.43.173:/home/pi/
```

> Replace `192.168.43.173` with your Raspberry Pi's actual IP address on the network.

---

## 📁 Project Structure

```plaintext
final-year-project/
├── backend/
│   └── app.py               # Flask API server
├── frontend/
│   ├── src/
│   │   ├── components/      # Dashboard UI components
│   │   └── App.js           # Main React app
│   └── package.json
├── traffic_system.py        # Main traffic control logic
├── test_led2.py             # LED hardware test script
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🔧 Hardware Setup

- Connect LED traffic lights to Raspberry Pi GPIO pins as defined in `traffic_system.py`
- Ensure camera is connected and accessible
- Make sure Raspberry Pi and your laptop are on the **same Wi-Fi network** for the dashboard to receive live data

---

## 👨‍💻 Developer

**Muhammad Fawaz ul Hassan**
- 🌐 [portfolio-website-bwxe.vercel.app](https://portfolio-website-bwxe.vercel.app/)
- 💼 [linkedin.com/in/muhammad-fawaz-ul-hassan](https://linkedin.com/in/muhammad-fawaz-ul-hassan/)
- 📧 fawazulhassan@gmail.com

---

## 📄 License

This project was developed as a Final Year Project at NUML, Lahore.
© 2026 Muhammad Fawaz ul Hassan
