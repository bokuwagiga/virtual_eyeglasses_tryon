# Virtual Eyeglass Try-On System

A real-time virtual eyeglass try-on system that uses an ESP32-S3 microcontroller with integrated camera module, computer vision techniques, and generative AI. The system streams video from an ESP32-S3 camera over WiFi and processes it using facial detection algorithms to overlay virtual eyeglasses onto the user's face.

![System UI](https://i.imgur.com/Evqyav6.png)

## Features

- Real-time video streaming from ESP32-S3 camera
- Face detection and facial landmark tracking
- Virtual eyeglass overlay with realistic positioning
- User-friendly interface to browse different eyeglass styles
- Integration with GAN-generated eyeglass frames
- Motion smoothing for stable eyeglass positioning
- Automatic transparent background processing for eyeglasses

## Technologies Used

- **Hardware**: ESP32-S3 with OV2640 camera module
- **ESP32 Firmware**: ESP-IDF framework, C language
- **Backend**: Python, OpenCV, Tkinter
- **Computer Vision**: Haar Cascade Classifier, LBF facial landmark model
- **Communication**: HTTP MJPEG streaming
- **Image Processing**: Alpha blending, perspective transformation

## Hardware Requirements

- ESP32-S3 development board
- OV2640 camera module
- Wi-Fi router for local network connectivity
- Computer for running the Python application

## Prerequisites

Before running this project, you need:

1. ESP-IDF environment set up for ESP32-S3 development
2. Python 3.6+ with the following packages:
   - OpenCV
   - NumPy
   - Tkinter
   - urllib3
3. The `lbfmodel.yaml` file for facial landmark detection (see setup instructions)

## Setting Up

### 1. Adding `lbfmodel.yaml`

To ensure the application works correctly, you need to add the `lbfmodel.yaml` file:

1. Obtain the `lbfmodel.yaml` file from the official OpenCV GitHub repository or other trusted sources
2. Place the file in the `app` folder
3. Verify that the file is correctly located at `app/lbfmodel.yaml`

### 2. ESP32-S3 Setup

1. Clone this repository
2. Configure Wi-Fi credentials in the ESP32 firmware
   ```c
   #define WIFI_SSID "your_wifi_ssid"
   #define WIFI_PASS "your_wifi_password"
   ```
3. Build and flash the firmware:
   ```bash
   idf.py build
   idf.py -p (PORT) flash monitor
   ```
4. Note the IP address displayed on the serial monitor

### 3. Python Application Setup

1. Navigate to the `app` directory
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Ensure the ESP32-S3 is powered on and connected to your Wi-Fi network
2. Launch the Python application
3. The application will display the video feed with virtual eyeglasses overlaid
4. Browse through different eyeglass options using the gallery at the bottom
5. Click on any eyeglass image to try it on

## System Architecture

### Hardware Configuration
The system uses an ESP32-S3 development board with an integrated OV2640 camera module, connected to a local Wi-Fi network.

### Software Components
1. **ESP32-S3 Firmware**: Written in C using the ESP-IDF framework, handling:
   - Camera initialization and configuration
   - Image capture
   - MJPEG encoding
   - HTTP server for video streaming

2. **Python Application**: Developed with OpenCV, Tkinter, and other libraries, responsible for:
   - Connecting to the ESP32-S3 video stream
   - Face detection and facial landmark tracking
   - Virtual eyeglass overlay processing
   - User interface for eyeglass selection

### Data Flow
1. ESP32-S3 captures camera frames at approximately 30 FPS
2. Frames are JPEG-encoded and sent over Wi-Fi via HTTP MJPEG stream
3. Python application receives and decodes the JPEG frames
4. Face detection algorithms locate faces in each frame
5. Facial landmark detection identifies key facial points
6. Virtual eyeglasses are resized, rotated, and positioned based on facial geometry
7. The augmented frame is displayed in the user interface

## Performance

- **Frame Rate**: 25-30 FPS (dependent on network conditions)
- **Face Detection Accuracy**: >95% for front-facing faces under normal lighting
- **System Latency**: 100-200ms end-to-end from capture to display
- **UI Responsiveness**: Smooth scrolling and eyeglass selection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- Espressif Systems for ESP32-S3 platform and documentation
- GAN research that enabled synthetic eyeglass frame generation

## Contact

Giga Shubitidze - [GitHub Profile](https://github.com/bokuwagiga)