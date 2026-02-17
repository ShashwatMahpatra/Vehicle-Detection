An end-to-end vehicle detection and counting system built using YOLOv8, GAN-based fog removal, and ROI-based tracking, with a Tkinter GUI for interactive usage.
The system is designed for real-world traffic videos, including foggy and low-visibility conditions, and supports Indian traffic categories such as autorickshaws, rickshaws, minitrucks, etc.

KEY FEATURES:-
 1)Video-based vehicle counting
 2)Interactive ROI selection using mouse (polygon drawing)
 3)Multi-class vehicle detection:-
   (i)Bicycle
   (ii)Motorbike
   (iii)Autorickshaw
   (iv)Rickshaw
   (v)Car
   (vi)Van
   (vii)Truck
   (viii)Mini-truck
   (ix)Bus
 4)Automatic fog detection
 5)GAN-based fog removal (U-Net Generator)
 6)UDA preprocessing using ResNet-18
 7)Real-time count dashboard (Tkinter GUI)
 8)GPU acceleration (CUDA supported)

 SYSTEM ARCHITECTURE:-
 Input Video
   ↓
Fog Detection
   ↓ (if fog detected)
GAN Fog Removal
   ↓
UDA Preprocessing (ResNet18)
   ↓
ROI Masking
   ↓
YOLOv8 Detection
   ↓
Multi-Frame Tracking (KD-Tree)
   ↓
Vehicle Counting + GUI Display


TECH STACK:-
1)Python
2)PyTorch
3)YOLOv8 (Ultralytics)
4)OpenCV
5)NumPy
6)SciPy
7)Tkinter
8)PIL
9)Torchvision

# Fog Removal GAN Model

The fog-removal generator weights are not included in this repository.

Download `generator1.pth` (207 MB) from:
<https://drive.google.com/file/d/13uTE9yv_Y0FmaETF4lD7SAJ0M0hABxcH/view?usp=sharing>

Place the file here:
models/generator1.pth

For Videos dowload from here:
<https://drive.google.com/file/d/1Jcdt0zRls4g661MGNf5_a94EQRQgDSVK/view?usp=sharing>
<https://drive.google.com/file/d/1OPopmOcnyCSHQqpFJrk1oOtQCvX9Zv0P/view?usp=sharing>


