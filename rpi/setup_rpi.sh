#!/bin/bash
# PURPOSE: Install everything needed on Raspberry Pi 4
# Run once after flashing Raspberry Pi OS
# Command: bash setup_rpi.sh

echo "================================================"
echo "  FasalAstra — Raspberry Pi 4 Setup Script"
echo "================================================"

# Update system
echo "[1/7] Updating system..."
sudo apt update && sudo apt upgrade -y

# Enable Pi Camera
echo "[2/7] Enabling Pi Camera interface..."
sudo raspi-config nonint do_camera 0
# If above fails, run: sudo raspi-config
# → Interface Options → Camera → Enable

# Install Python dependencies
echo "[3/7] Installing Python packages..."
pip3 install ultralytics          # YOLOv8
pip3 install picamera2            # Pi Camera library
pip3 install RPi.GPIO             # GPIO control
pip3 install smbus2               # I2C for MPU6050
pip3 install opencv-python        # Image processing
pip3 install numpy                # Math

# Install ONNX Runtime for ARM (faster inference)
echo "[4/7] Installing ONNX Runtime for ARM..."
pip3 install onnxruntime

# Install MPU6050 library
echo "[5/7] Installing IMU library..."
pip3 install mpu6050-raspberrypi

# Enable I2C for MPU6050
echo "[6/7] Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Test camera
echo "[7/7] Testing Pi Camera..."
python3 -c "
from picamera2 import Picamera2
cam = Picamera2()
cam.start()
import time; time.sleep(1)
frame = cam.capture_array()
print(f'Camera OK — frame shape: {frame.shape}')
cam.stop()
"

echo ""
echo "================================================"
echo "  ✅ RPi4 Setup Complete!"
echo "  Next: python3 rpi/main_rpi.py"
echo "================================================"
