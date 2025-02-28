import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from scipy.signal import butter, filtfilt, find_peaks, welch
from geopy.distance import geodesic

# Load data
acceleration_file = "https://raw.githubusercontent.com/Hikutaatti/FysiikkaFinal/refs/heads/main/Linear%20Acceleration.csv"
gps_file = "https://raw.githubusercontent.com/Hikutaatti/FysiikkaFinal/refs/heads/main/Location.csv"

accel_data = pd.read_csv(acceleration_file)
gps_data = pd.read_csv(gps_file)

# Determine best acceleration component
st.title("Step Analysis and GPS Visualization")

st.subheader("Raw Acceleration Data")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(accel_data["Time (s)"], accel_data["Linear Acceleration x (m/s^2)"], label="X-axis")
ax.plot(accel_data["Time (s)"], accel_data["Linear Acceleration y (m/s^2)"], label="Y-axis")
ax.plot(accel_data["Time (s)"], accel_data["Linear Acceleration z (m/s^2)"], label="Z-axis")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s²)")
ax.legend()
st.pyplot(fig)

# Choose best axis based on frequency analysis
sampling_rate = 1 / np.mean(np.diff(accel_data["Time (s)"]))

def compute_psd(data, fs):
    freqs, psd = welch(data, fs=fs, nperseg=256)
    return freqs, psd

freqs_x, psd_x = compute_psd(accel_data["Linear Acceleration x (m/s^2)"], sampling_rate)
freqs_y, psd_y = compute_psd(accel_data["Linear Acceleration y (m/s^2)"], sampling_rate)
freqs_z, psd_z = compute_psd(accel_data["Linear Acceleration z (m/s^2)"], sampling_rate)

# Plot PSD
st.subheader("Power Spectral Density")
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(freqs_x, psd_x, label="X-axis")
ax.semilogy(freqs_y, psd_y, label="Y-axis")
ax.semilogy(freqs_z, psd_z, label="Z-axis")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density")
ax.legend()
st.pyplot(fig)

# Choose best axis (z-axis assumed best)
selected_axis = "Linear Acceleration z (m/s^2)"

# Bandpass filter for step detection
def bandpass_filter(data, lowcut=0.8, highcut=3.0, fs=sampling_rate, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

filtered_accel = bandpass_filter(accel_data[selected_axis])

# Detect peaks (steps)
peaks, _ = find_peaks(filtered_accel, height=0.5, distance=sampling_rate/2)
step_count_filtered = len(peaks)

# Plot filtered acceleration with steps
st.subheader(f"Filtered Acceleration Data (Steps Counted: {step_count_filtered})")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(accel_data["Time (s)"], filtered_accel, label="Filtered Acceleration", color="red")
ax.plot(accel_data["Time (s)"].iloc[peaks], filtered_accel[peaks], "bo", label="Detected Steps")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s²)")
ax.legend()
st.pyplot(fig)

# Step count using Fourier analysis
dominant_freq = freqs_z[np.argmax(psd_z)]
total_duration = accel_data["Time (s)"].iloc[-1] - accel_data["Time (s)"].iloc[0]
step_count_fourier = int(dominant_freq * total_duration)

# Compute distance and speed from GPS data
total_distance = sum(
    geodesic((gps_data["Latitude (°)"].iloc[i], gps_data["Longitude (°)"].iloc[i]),
             (gps_data["Latitude (°)"].iloc[i+1], gps_data["Longitude (°)"].iloc[i+1])).meters
    for i in range(len(gps_data) - 1) 
)
average_speed = gps_data["Velocity (m/s)"].mean(skipna=True)

# Compute step length
step_length = total_distance / step_count_filtered if step_count_filtered > 0 else 0

# Display computed values
st.subheader("Computed Metrics")
st.write(f"**Step Count (Filtered Acceleration):** {step_count_filtered}")
st.write(f"**Step Count (Fourier Analysis):** {step_count_fourier}")
st.write(f"**Total Distance Traveled:** {total_distance:.2f} meters")
st.write(f"**Average Speed:** {average_speed:.2f} m/s")
st.write(f"**Step Length:** {step_length:.2f} meters")

# Route visualization with Folium
st.subheader("Route Visualization")
route_map = folium.Map(location=[gps_data["Latitude (°)"].iloc[0], gps_data["Longitude (°)"].iloc[0]], zoom_start=15)
route_coords = list(zip(gps_data["Latitude (°)"], gps_data["Longitude (°)"]))
folium.PolyLine(route_coords, color="blue", weight=3, opacity=0.7).add_to(route_map)
folium_static(route_map)
