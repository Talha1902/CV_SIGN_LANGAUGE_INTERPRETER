# CV_SIGN_LANGAUGE_INTERPRETER
Sign Language Interpreter using Deep learning Model CNN.
Gesture Recognition System

This project implements a real-time hand gesture recognition system using a pre-trained deep learning model. The system captures video frames from a webcam, processes these frames to extract hand gestures, and predicts the corresponding gestures using a convolutional neural network (CNN) model.

Introduction
The gesture recognition system aims to identify various hand gestures in real-time using a webcam. The system loads a pre-trained model to predict gestures and displays the results along with confidence scores on the video feed.

 Features
- Real-time hand gesture recognition
- High accuracy for well-defined gestures
- Real-time feedback with confidence scores
- Logs predictions with timestamps for further analysis

  Requirements
- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- JSON

Installation
1. Clone the repository:

2. Install the required packages:
   ```bash
   pip install -r install_packages.txt
   ```

3. Ensure you have a webcam connected to your system.

 Usage
 
1. Place the pre-trained model (`gesture_model.h5`) and the label map (`label_map.json`) in the project directory.

2. Run the `final.py` script:
   ```bash
   python final.py
   ```

3. The system will start capturing video from the webcam, recognize hand gestures, and display predictions with confidence scores in real-time.

4. Press 'q' to quit the application.

 Methodology
1. Model and Label Loading: Load the pre-trained gesture recognition model and label map.
2. Video Capture Initialization: Initialize the webcam for real-time video capture.
3. Frame Processing: Continuously capture and process video frames to extract the region of interest (ROI) containing hand gestures.
4. Gesture Prediction: Preprocess the ROI and use the model to predict the gesture.
5. Display and Logging: Display the predicted gesture and confidence score on the video feed and log the predictions with timestamps.

 Results and Analysis
  Accuracy: High confidence scores (>90%) for well-defined gestures.
  Performance**: Real-time processing with minimal lag, suitable for interactive applications.
  Logging**: Predictions saved with timestamps for further analysis.

 Conclusion
  The system effectively recognizes hand gestures in real-time with high accuracy, making it suitable for interactive applications. While it performs well with clear gestures, it may   
  struggle with ambiguous ones and varying environmental conditions. Future improvements can enhance its robustness and accuracy.

 Future Improvements
 Dynamic adjustment of the region of interest (ROI) to better track hand movements.
 Expanding the training dataset to include more diverse gestures.
 Implementing background subtraction and other noise reduction techniques.

License
This project is licensed under the MIT License. See the [LICENSE] file for details.
