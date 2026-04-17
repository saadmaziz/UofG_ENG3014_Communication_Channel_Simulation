# Communication Channel Simulation:

This repository contains a simulation of a digital communication downlink developed for the **ENG3014 Communications Systems 3** project. The system models a wireless communication link, approximating **3GPP LTE Release 8** logic to analyze performance in urban environments.

---

## Project Overview

The simulation was designed to meet specific core requirements while incorporating advanced "creative additions" to improve robustness and efficiency.

### Core Requirements
* **Data Transmission:** Conversion of a message comprising more than 2048 bits from string to binary.
* **Spectral Constraints:** Use of an output filter to limit transmitter bandwidth to less than 20% of the carrier frequency.
* **SNR Benchmarking:** Performance analysis at 6dB, 18dB, and 24dB signal-to-noise ratios (SNR) relative to AWGN.
* **Analysis:** Evaluation of the Bit Error Rate (BER) and signal recovery over multiple simulated retransmissions.

### Creative Additions (3GPP LTE-inspired)
* **Adaptive Modulation and Coding (AMC):** The system dynamically switches between **QPSK** (for noise tolerance) and **16-QAM** (for higher throughput) based on the measured BER.
* **Convolutional Coding:** Implementation of a $R=1/2$, $K=3$ non-recursive encoder with **Viterbi decoding**.
* **Error Control:** A **Stop-and-Wait Automatic Repeat Request (ARQ)** system that requests retransmission if errors are detected, up to a limit of 16 attempts.
* **Fading Models:** Emulation of urban Non-Line-of-Sight (NLOS) conditions using **Jakes' Model Rayleigh fading**.

---

## Simulation Parameters

The following parameters were utilized to characterize the wireless link:

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Carrier Frequency** | 233.5 kHz | Scaled for simulation efficiency. |
| **Modulation** | QPSK, 16-QAM | LTE Rel-8 Standard schemes. |
| **Coding Rate** | 1/2 | Standard convolutional coding rate. |
| **Velocity** | 5.5 km/h | Representative of an urban pedestrian. |
| **Roll-off ($\alpha$)** | 0.35 | Standard pulse shaping factor. |

---

## Key Results and Analysis

The simulation performance was validated through Monte Carlo analysis over 100 transmissions for various scenarios.

* **Waterfall Characteristic:** The system exhibits a typical "waterfall" BER curve where low SNR results in a ~70% BER.
* **Zero Error Regime:** The system reaches a near-perfect signal recovery state at SNRs above 18dB.
* **Efficiency Gains:** The implementation of AMC led to a **3x reduction in transmission time**
