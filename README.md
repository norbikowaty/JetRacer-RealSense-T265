# JetRacer + Intel RealSense T265 Autonomous System

![Python](https://img.shields.io/badge/Python-3.6.9-blue.svg)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Nano%20(4GB)-green.svg)
![JetPack](https://img.shields.io/badge/JetPack-4.5.1%20%7C%20L4T%2032.5.2-brightgreen.svg)
![Hardware](https://img.shields.io/badge/Hardware-Intel%20RealSense%20T265-orange.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20OpenCV-red.svg)

Kompletny system autonomicznego kierowania pojazdem **JetRacer** (oparty na platformie NVIDIA Jetson Nano), zintegrowany z kamerą śledzącą **Intel RealSense T265** do wizualnej odometrii (VIO), estymacji pozycji w przestrzeni 3D oraz szerokokątną kamerą CSI do śledzenia toru jazdy i analizy otoczenia.

---

> **Wersja z wykorzystaniem algorytmu YOLO**  
>  W trakcie prac nad niniejszym projektem wykorzystano oraz zaadaptowano moduły, skrypty sterujące i notatniki z repozytorium bazowego opisującego sterowanie tą samą jednostką pojazdu **Jetracer bez użycia kamery Intel RealSense T265**.
> * **[Repozytorium powiązane](https://github.com/rumcajzzzz/JetRacer)**

---

##  Główne Funkcjonalności

* **Wizualna Odometria VIO (T265):** Precyzyjna lokalizacja pojazdu w przestrzeni 3D realizowana na dedykowanym układzie VPU kamery T265 (bez obciążania zasobów Jetsona Nano).
* **Autonomiczna Jazda (Deep Learning):** Estymacja kąta skrętu oraz prędkości przy użyciu sieci neuronowych w frameworku PyTorch (ResNet-18).
* **Śledzenie Toru (Kamera CSI):** Analiza obrazu w czasie rzeczywistym z szerokokątnej kamery 160° FOV.
* **Teleoperacja i Zbiór Danych:** Notatniki Jupyter umożliwiające sterowanie padem, rejestrację trajektorii oraz klatek obrazu.
* **Korekta i Bezpieczeństwo:** Integracja danych pozycjonowania z pętlą sterowania serwomechanizmem i silnikiem w oparciu o bibliotekę `jetracer`.

---

##  Architektura Sprzętowa i Programowa

### Specyfikacja Platformy (Hardware & Board Info)
* **Kontroler:** NVIDIA Jetson Nano Developer Kit (B01 / P3448-0000 / Board P3449-0000, kod: `porg`) <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/f94e1c95-65dc-4fd1-8dbc-899128adef62" />

* **Architektura GPU:** Maxwell (CUDA ARCH_BIN: 5.3)
* **Pojazd:** Waveshare JetRacer AI Kit (sterownik PCA9685, podwozie RC)

<p align="center">
  <img src="https://github.com/user-attachments/assets/6f0194a3-2c6e-4840-87c0-1abf6fd49d41" width="45%" alt="JetRacer AI Kit 1" />
  &nbsp;
  <img src="https://github.com/user-attachments/assets/9d32d7b3-efa6-4f4e-8f6e-d6a17ba7dc37" width="45%" alt="JetRacer AI Kit 2" />
</p>
W celu realizacji projektu wykonano autorski model 3D uchwytu na kamerę T265 zamontowany nad zderzakiem pojazdu (zdjęcie po prawej stronie). 

* **Sensory Wizyjne:**
  * **Tracking Camera:** Intel RealSense T265 (VIO, 2x Fisheye, IMU, VPU Movidius Myriad 2)
  <img width="582" height="277" alt="image" src="https://github.com/user-attachments/assets/3528f942-c902-4abb-8d07-348fe71c5fb3" />

  * **Kamera Widokowa:** Szerokokątna kamera CSI 8MP (160° FOV)

### Środowisko Programistyczne (Software Stack)
* **OS / L4T:** Ubuntu 18.04 LTS / **L4T 32.5.2** (JetPack 4.5.1)
* **Python:** 3.6.9
* **Sterowanie zasilaniem:** NV Power Mode: `MAXN` (Mode 0) z aktywną usługą `jetson_stats.service` (v3.1.0)
* **Biblioteki i Frameworki:**
  * **CUDA:** 10.2.89
  * **cuDNN:** 8.0.0.180
  * **TensorRT:** 7.1.3.0
  * **VisionWorks:** 1.6.0.501
  * **VPI:** 1.0.15 (`libnvvpi1`)
  * **OpenCV:** 4.1.1 (kompilacja CPU)
  * **PyTorch / torchvision:** (wersje dedykowane dla JetPack 4.x / ARM64)
  * **`pyrealsense2`:** budowane ze źródeł pod architekturę `aarch64`
  * **`jetracer` SDK:** kontrola serwa i silników via I2C/PCA9685

---

## Struktura Repozytorium

```text
JetRacer-RealSense-T265/
├── t265_basics/                           # Podstawowe notatniki i eksperymenty z kamerą T265
│   ├── coords_T265.ipynb                 # Odczyt i analiza współrzędnych przestrzennych
│   ├── tracking_T265.ipynb               # Testy śledzenia pozycji oraz odometrii VIO wraz z prostą wizualizacją
│   ├── triangulate_T265.ipynb            # Triangulacja i obliczenia geometrii 3D, mierzenie dystansu obiektów od kamery
│   └── video_T265.ipynb                  # Odbiór i wyświetlanie strumieni wideo z kamer fisheye
├── disparity_map_t265.py                 # Skrypt generujący mapę dysparycji z kamer stereo (wersja dla Windows)
├── road_following_obstacle_avoidance.py  # Skrypt śledzenia drogi z omijaniem przeszkód z wykorzystaniem obu kamer 
├── road_following_t265.ipynb             # Skrypt śledzenia drogi bez omijania służący do czytania i zapisy pozycji pojazdu
├── teleop_T265_mapping.ipynb             # Teleoperacja padem połączona z mapowaniem pozycji
├── tracking_visualization.ipynb          # Wizualizacja zarejestrowanych trajektorii i ścieżek, porównanie tracking'u T265 z algorytmem YOLO
└── README.md
```
---

## Instalacja i uruchomienie

### 1. Budowa `librealsense2` na Jetson Nano

Z uwagi na specyfikację architektury ARM64 i JetPacka 4.5.1 (`L4T 32.5.2`) zalecanym jest skorzystać z poniższych repozytoriów użytkownika **[JetsonHacksNano](https://github.com/JetsonHacksNano)**:
* **[Instalacja swapu](https://github.com/JetsonHacksNano/installSwapfile)**
* **[Instalacja biblioteki Librealsense](https://github.com/JetsonHacksNano/installLibrealsense)**
Podczas instalowania paczki realsense należy zwrócić uwagę na fakt, że **model T265 nie jest już wspierany** i wymaganym jest zainstalowanie starszej wersji `Librealsense` (v2.50.0) Więcej informacji o tym w powyższym repozytorium.

### 2. Weryfikacja instalacji `pyrealsense2`

Po zakończeniu procesu kompilacji i instalacji należy upewnić się, że Python 3.6 w systemie poprawnie importuje wygenerowane moduły. 

Weryfikacji dokonuje się poprzez wykonanie poniższego polecenia w terminalu:

```bash
python -c "import pyrealsense2 as rs; print('version:', rs.__version__)"
```

Aby sprawdzić, czy system "widzi" kamerę można skorzystać z polecenia:

```bash
rs-enumerate-devices
```
Jeśli Python nie odnajduje modułu, należy się upewnić, że ścieżka do zainstalowanych bibliotek została dodana do zmiennej środowiskowej `PYTHONPATH`:
```bash
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2' >> ~/.bashrc
source ~/.bashrc
```

