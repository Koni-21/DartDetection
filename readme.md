# Sub Project: Dart Localization

 This sub-project focuses on localizing steel darts on a dartboard using two (or more) cameras. To detect the darts in the camera images a CNN is used (YOLO). The localization of the darts from two perspectives is done by stereo camera calibration. 

Read more in:
[Sub Project: Dart localization](assets\240229_Projectwork_dart_localization.pdf)

The full e-steel dart project is still in development and not yet released.

Concept and realization of the full e-Steel dart project.
![Dart Localization](assets\concept_and_realization_of_the_main_e_steel_dart_project.png)
Single dart on the triple 10 field of the dartboard with the input images to test the dart detection process:
![Dart Localization](assets\dart_localization_example_triple10.png)
Calculated position of the dart (blue) with the corresponding camera images with detected arrow and line fit (red):
![Dart Localization](assets\dart_localization_example_triple10_results.png)

This sub project contains the core functionality to detect a dart, calibrate the cameras and localize the dart on the dartboard.

The stereo calibration is derived from: https://github.com/TemugeB/python_stereo_camera_calibrate.

## How to Install

1. **Clone the Repository:**
```bash
   git clone https://github.com/Koni-21/DartDetection.git
   cd DartDetection
```
2. **Set Up the Virtual Environment:**
```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
3. **Install the Required Packages:**
```bash
    pip install -r requirements.txt
```
4. **Install the DartDetection Package:**
```bash
    pip install -e . # use the editable flag in case you want to play around
```
5. **Test the localization with simulated camera outputs**
```bash
    python src\dartdetect\stereolocalize.py
```
6. **Test the AI-based dart detection with an example image**
```bash
    python src\dartdetect\dartlocalize.py    
```

7. **Train your own model and calibrate your own setup**

See [Sub Project: Dart localization](assets\240229_Projectwork_dart_localization.pdf) and use the example code at the end of each module as an entry point to play around. Also, take into account the tests to determine the appropriate usage of the functions.