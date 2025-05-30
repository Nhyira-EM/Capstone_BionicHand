import ncnn
import numpy as np
from adafruit_servokit import ServoKit
import serial
import time

#initializing serial port
ser = serial.Serial('/dev/ttyUSB0', 115200)  #replace with your serial port
kit = ServoKit(channels=16)

#load ncnn model
net = ncnn.Net()
net.load_param("rnn_lstm.ncnn.param")
net.load_model("rnn_lstm.ncnn.bin")

#gesture to servo angle mappings (no thumb)
gesture_to_angles = {
    "Hand open": {
        0: 180,  #pinky
        1: 0,    #ring
        3: 180,  #middle
        8: 180,  #index
    },
    "Hand close": {
        0: 0,
        1: 150,
        3: 70,
        8: 60,
    },
    "Middle and Index Extension": {
        0: 0,
        1: 150,
        3: 180,
        8: 180,
    },
}

#preprocessing parameters
sampling_rate = 1000  # Hz
window_size = 200     #samples per inference

#rolling window to collect data
window = np.zeros((window_size, 3), dtype=np.float32)
index = 0

#preprocessing function
def preprocess_data(window):
    #mean removal
    window = window - np.mean(window, axis=0)

    #rectification
    window = np.abs(window)

    #moving average
    window_size_smooth = int(0.01 * sampling_rate)  #10 samples for 10ms window
    smoothed_window = np.zeros_like(window)
    for ch in range(window.shape[1]):
        smoothed_window[:, ch] = np.convolve(
            window[:, ch], np.ones(window_size_smooth) / window_size_smooth, mode='same'
        )

    #normalization
    min_vals = np.min(smoothed_window, axis=0)
    max_vals = np.max(smoothed_window, axis=0)
    normalized_window = (smoothed_window - min_vals) / (max_vals - min_vals + 1e-8)

    return normalized_window

#main loop
while True:
    if ser.in_waiting > 0:
        raw_data = ser.readline().decode('utf-8').strip()
        try:
            sensor_data = list(map(float, raw_data.split(',')))  #should be 3 values
            if len(sensor_data) == 3:
                window[index % window_size] = sensor_data
                index += 1

                if index >= window_size:
                    processed_data = preprocess_data(window)
                    input_data = np.reshape(processed_data, (1, 3, window_size))
                    input_array = ncnn.Mat(input_data)

                    #running inference
                    ex = net.create_extractor()
                    ex.input("input", input_array)
                    output = ex.extract("output")

                    #predicted gesture
                    predicted_index = int(np.argmax(np.array(output)))
                    gesture_list = list(gesture_to_angles.keys())
                    if predicted_index < len(gesture_list):
                        predicted_gesture = gesture_list[predicted_index]
                        print(f"Inference: {predicted_gesture}")

                        if predicted_gesture in gesture_to_angles:
                            angles = gesture_to_angles[predicted_gesture]
                            for pin, angle in angles.items():
                                kit.servo[pin].angle = angle
                            time.sleep(1)
                            for pin in angles:
                                kit.servo[pin].angle = 0

        except ValueError:
            print("Invalid sensor data:", raw_data)

    time.sleep(0.01)  #slight delay for serial communication