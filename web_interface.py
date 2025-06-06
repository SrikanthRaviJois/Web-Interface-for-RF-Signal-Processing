import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from rtlsdr import RtlSdr
import asyncio
from collections import deque
import time
import pandas as pd
import socket

st.set_page_config(layout="wide")
FFT_SIZE = 4096
BUFFER_SIZE = 100
UDP_IP = "172.16.129.180"  
UDP_PORT = 8080
RECEIVE_BUFFER_SIZE = 4096
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

st.sidebar.title("Menu")
input_source = st.sidebar.selectbox("Input Source", ["SDR", "UDP Socket"])
menu = st.sidebar.selectbox("Navigate",["Home", "Settings", "About"])

spectrogram_buffer = deque(maxlen=BUFFER_SIZE)
power_buffer = deque(maxlen=BUFFER_SIZE)
mean_buffer = deque(maxlen=40)  

st.title("FFT and Spectrogram")
center_freq_mhz = st.slider("Center Frequency (MHz)", min_value=88.0, max_value=108.0, step=0.1, value=91.9)
# center_freq_mhz = st.slider("Center Frequency (MHz)", min_value=900.0, max_value=1800.0, step=10.0, value=1500.0)
center_freq = center_freq_mhz * 1e6
# plot = st.empty()
plot = st.empty()
plot2 = st.empty()
plot3 = st.empty()
csv_button = st.button("Save Data to CSV")

if menu == "Home":
    st.sidebar.write("You are in Home.")
elif menu == "Settings":
    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
elif menu == "About":
    st.sidebar.markdown("""
    
                        
    """)
if input_source == "UDP Socket":
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", UDP_PORT))
    recv_sock.setblocking(False)

st.sidebar.markdown("" \
"" \
"" \
"" \
"" \
"" \
"" \
"" \
"Developed by SRIKANTH, CHANDANA, SHISHIR, MEENAKSHI ")
# spec_plot = st.empty()
if input_source == "SDR":
    if "sdr" not in st.session_state:
        sdr = RtlSdr()
        sdr.sample_rate = 2.048e6
        sdr.gain = 'auto'
        st.session_state.sdr = sdr
    else:
        sdr = st.session_state.sdr

    sdr.center_freq = center_freq

while True:
    sample_list = []
    if input_source == "SDR":
        samples = sdr.read_samples(FFT_SIZE)
    else:  # Receiving from UDP
        try:
            data, _ = recv_sock.recvfrom(FFT_SIZE * 8 * 2)  # complex64 = 8 bytes (2x float32)
            samples = np.frombuffer(data, dtype=np.complex64)
        except BlockingIOError:
            continue
    window = np.hanning(FFT_SIZE)
    spectrum = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/2.048e6)) + center_freq
    power = 20 * np.log10(np.abs(spectrum)/50)
    spectrogram_buffer.appendleft(power)
    fig_fft, ax_fft = plt.subplots(nrows=1, figsize=(5, 3))
    fig_spec, (ax_spec, ax_pow) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3), width_ratios=[3, 1])  
    plt.tight_layout(pad=0.1)
    ax_fft.plot(freqs / 1e6, power)
    ax_fft.set_title("Real-Time FFT")
    ax_fft.set_ylabel("Power (dB)")
    ax_fft.tick_params(axis='both', which='major', labelsize=4)
    ax_spec.tick_params(axis='both', which='major', labelsize=4)
    power_list = np.sum(spectrogram_buffer[0])
    mean_buffer.append(power_list)
    power_buffer.append(np.mean(mean_buffer))
    sample_list.append(power)
    samples_bytes = samples.astype(np.complex64).tobytes()
    sock.sendto(samples_bytes, (UDP_IP, UDP_PORT))


    ax_pow.plot(power_buffer, np.arange(len(power_buffer)))
    ax_spec.set_title("Spectrogram")
    ax_pow.set_title("Real-Time Power")
    ax_pow.set_ylabel("Power (dB)")
    ax_pow.tick_params(axis='both', which='major', labelsize=4)

    ax_fft.grid(True)
    ax_pow.grid(True)
    ax_pow.set_xlabel("Time")
    ax_pow.set_ylabel("Total Power (dB)")
    # ax_fft.xlabel("Frequency (MHz)")
    ax_fft.set_xlim(freqs[0]/1e6, freqs[-1]/1e6)
    # tick_freqs = np.linspace(freqs[0], freqs[-1], num=10)
    # ax_fft.set_xticks(tick_freqs / 1e6) 
    # ax_fft.set_xlabel("Frequency (MHz)", color='white')
    ax_fft.set_ylim(-60, 100)
    # fft_plot.pyplot(fig_fft)
    # plt.close(fig_fft)

    img = ax_spec.imshow(np.array(spectrogram_buffer), origin='upper', aspect='auto', extent=[freqs[0]/1e6, freqs[-1]/1e6, 0, 60], cmap='jet', vmin=-40, vmax=5)
    ax_spec.set_yticks([])
    
    fig_fft.patch.set_facecolor('black')
    fig_spec.patch.set_facecolor('black')
    ax_fft.set_facecolor('black')
    ax_spec.set_facecolor('black')
    ax_pow.set_facecolor('black')
    ax_pow.tick_params(colors='white')
    ax_pow.xaxis.label.set_color('white')
    ax_pow.yaxis.label.set_color('white')
    ax_pow.title.set_color('white')

    ax_fft.tick_params(colors='white')
    ax_fft.xaxis.label.set_color('white')
    ax_fft.yaxis.label.set_color('white')
    ax_fft.title.set_color('white')

    ax_spec.tick_params(colors='white')
    ax_spec.xaxis.label.set_color('white')
    ax_spec.yaxis.label.set_color('white')
    ax_spec.title.set_color('white')
    
    if len(spectrogram_buffer) > 1:
        spec_array = np.array(spectrogram_buffer)
        avg_power_density = np.mean(spec_array, axis=0)
        freqs_mhz = freqs / 1e6
        avg_power_density -= np.min(avg_power_density)
        probability = avg_power_density / np.sum(avg_power_density)\
        # num_bins = 100
        num_bins = 50
        bin_edges = np.linspace(freqs_mhz[0], freqs_mhz[-1], num_bins + 1)
        bin_indices = np.digitize(freqs_mhz, bin_edges) - 1
        binned_prob = np.zeros(num_bins)
        for i in range(num_bins):
            binned_prob[i] = np.sum(probability[bin_indices == i])

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig_hist, ax_hist = plt.subplots(figsize=(6, 2.5))
        ax_hist.bar(bin_centers, binned_prob, width=(bin_edges[1] - bin_edges[0]), color='cyan')
        ax_hist.set_title("Frequency Distribution")
        ax_hist.set_xlabel("Frequency (MHz)")
        ax_hist.set_ylabel("Probability")
        ax_hist.grid(True)

        fig_hist.patch.set_facecolor('black')
        ax_hist.set_facecolor('black')
        ax_hist.tick_params(colors='white')
        ax_hist.xaxis.label.set_color('white')
        ax_hist.yaxis.label.set_color('white')
        ax_hist.title.set_color('white')

        plot3.pyplot(fig_hist)
        plt.close(fig_hist)
    # fig_fft.colorbar(img, ax=ax_spec, label="Power (dB)")

    df = pd.DataFrame(samples, columns=['samples'])
    if csv_button:
        df.to_csv("data.csv",mode='a', index=False, header=False)
    
    plot.pyplot(fig_fft)
    plt.close(fig_fft)

    plot2.pyplot(fig_spec)
    plt.close(fig_spec)

    time.sleep(0.01)
# sdr.close()
