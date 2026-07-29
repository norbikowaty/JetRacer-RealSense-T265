# JetRacer + Intel RealSense T265 Autonomous System

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano-green.svg)
![Hardware](https://img.shields.io/badge/Hardware-Intel%20RealSense%20T265-orange.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20OpenCV-red.svg)

Repozytorium gromadzące kompletny kod, notatniki Jupyter oraz eksperymenty związane z budową autonomicznego pojazdu **JetRacer** (NVIDIA Jetson Nano) z wykorzystaniem kamery śledzącej **Intel RealSense T265** (odometria wizualna / VIO) oraz estymacji głębi.

---

## Architektura Sprzętowa i Programowa

* **Pojazd:** Waveshare JetRacer / NVIDIA Jetson Nano (JetPack 4.x / Ubuntu 18.04)
* **Kamery:** Intel RealSense T265 (Tracking Camera), Kamera 8MP HD, szerokokątna 160° FOV
* **Kluczowe biblioteki:** `pyrealsense2`, PyTorch, OpenCV, `traitlets`

---
