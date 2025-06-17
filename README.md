# PCB Component Detection with YOLOv8

This project is a Streamlit web app for real-time PCB component detection using a YOLOv8 model. It supports detection from webcam, image, and video inputs.

## Features

- **Webcam Detection:** Real-time detection using your computer's webcam (local only).
- **Image Upload:** Upload an image for component detection.
- **Video Upload:** Upload a video file for frame-by-frame detection.

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- NumPy

Install dependencies:

```sh
pip install streamlit opencv-python ultralytics numpy
```

## Usage

1. Place your trained YOLOv8 model file (e.g., `best.pt`) in the project directory.
2. Run the app locally:
   ```sh
   streamlit run app.py
   ```
3. Open the web browser link provided by Streamlit.
4. Choose your input source (Webcam or Image/Video Upload) and follow the instructions.

> **Note:**
> Webcam and real-time video detection only work when running locally. Cloud platforms do not support webcam access.

## File Structure

```
.
├── app.py
├── best.pt
├── requirements.txt
└── README.md
```

## Troubleshooting

- **Webcam not working:** Make sure you are running the app locally and your webcam is connected.
- **OpenCV errors:** Ensure all dependencies are installed and your Python environment is set up correctly.
- **Cloud deployment:** Use only image/video upload features; webcam will not work.

## License

This project is for educational and research purposes. Please check the licenses of YOLOv8 and other dependencies for
