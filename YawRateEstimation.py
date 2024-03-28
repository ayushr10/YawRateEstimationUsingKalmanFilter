import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Kalman Filter Initialization
def initialize_kalman_filter(initial_state, process_noise, measurement_noise):
    # State vector [remaining_range]
    x = np.array([[initial_state]])
    # State transition matrix [F]
    F = np.array([[1]])
    # Observation matrix [H]
    H = np.array([[1]])
    # Covariance matrix of the process noise [Q]
    Q = np.array([[process_noise]])
    # Covariance matrix of the measurement noise [R]
    R = np.array([[measurement_noise]])
    # Initial covariance matrix
    P = np.eye(1)  # Identity matrix
    return x, F, H, Q, R, P

# Kalman Filter Prediction
def predict(x, F, P, Q):
    x = np.dot(F, x)
    P = np.dot(np.dot(F, P), F.T) + Q
    return x, P

# Kalman Filter Update
def update(x, P, z, H, R):
    y = z - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    P = np.dot((np.eye(len(x)) - np.dot(K, H)), P)
    return x, P
#df = pd.read_csv('dataset.csv')
df = pd.read_csv('DataSetSteering.csv')
print(df.columns)

# Mathematical model for range estimation.
time = df['Time(s)'] # time in seconds
wheelbase=2.618;
#YawRate=df['YawRate(rad/s)']
YawRateCalculated = df['Velocity(m/s)'] * df['SteerAngle(rad)'].apply(math.tan) / wheelbase


plt.plot(time, YawRateCalculated, label='Yaw Rate by mathematical model', color='green')
#plt.plot(time,YawRate , label='Actual Yaw Rate',color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Yaw Rate Estimation from Mathematical Model (vs Time)')
plt.legend()
plt.show()


noisy_estimated_yawRate = YawRateCalculated + np.random.normal(0, 0.1, df.shape[0])

# Kalman Filter Initialization
initial_state = 0
process_noise = 0.001  # Adjust as needed
measurement_noise = 0.01  # Adjust as needed
x, F, H, Q, R, P = initialize_kalman_filter(initial_state, process_noise, measurement_noise)
print(x, F, H, Q, R, P)



# Arrays to store results for plotting
kalman_estimated_remaining_yawRate = np.zeros(df.shape[0])

# total_estimated_remaining_range_with_noise[0]=0 # correction
# Kalman Filter Simulation loop
for i in range(df.shape[0]):
    # Kalman Filter prediction
    x, P = predict(x, F, P, Q)
    # Kalman Filter update using the actual distance covered as the measurement
    measurement = noisy_estimated_yawRate[i]
    # print(total_estimated_remaining_range_with_noise[i], distance_covered_till_now[i])
    x, P = update(x, P, measurement, H, R)
    # print(x)
    # Store the estimated remaining range for plotting
    kalman_estimated_remaining_yawRate[i] = x[0, 0]

plt.plot(time, kalman_estimated_remaining_yawRate, label='Yaw Rate', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Yaw Rate Estimation from Kalman Filter (vs Time)')
plt.legend()
plt.show()


plt.plot(time, kalman_estimated_remaining_yawRate, label='Kalman Filter Estimateed Yaw Rate',color='blue')
plt.plot(time, YawRateCalculated, label='Estimated Yaw Rate',color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Estimated Remaining Yaw Rate by mathematical model and Kalman over Time')
plt.legend()
plt.show()