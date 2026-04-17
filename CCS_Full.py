# Importing required libraries
import math
import textwrap
import numpy as np
import matplotlib.pyplot as plt

# ASCII string to BIN String
def ascii2bin(str):
    """ input string of ASCII characters and returns string of binary bits with no deliminatrors""" 
    byte  = [format(ord(char), '08b') for char in str]
    bin = "".join(byte)
    return bin,

# Data packetization
def packetize(data, packetLength):
    """input data split into a sections of legth packetLength """
    # math and textwrap packages required
    dataLen = len(data)
    nPackets = math.ceil(dataLen/packetLength)
    data = data + "0"*(int(packetLength * ( nPackets-dataLen/packetLength)))
    packets = textwrap.wrap(data, packetLength)
    return packets

# Covolutional encoding
def convolutionEncodePacket(packet_data, packet_number, knownString):
    """
    Applies rate 1/2 K=3 (7,5) convolution to a SINGLE packet.
    Input: Single binary string payload, Integer packet number, Known binary string.
    Output: Single convoluted binary string.
    """
    
    convoludedpacket = list()

    # 1. Create the Header and Full Packet
    # We use the passed 'packet_number' integer here instead of an internal counter
    header = str(format(packet_number, '08b') + knownString)
    currentpacket = header + str(packet_data) + "00" # Header + Data + Tail Flush
    
    # 2. Setup Convolution Registers
    bits = textwrap.wrap(currentpacket, 1)
    bit0 = 0
    bit1 = 0
    bit2 = 0
    
    itterator2 = len(bits)
    counter2 = 0
    
    # 3. Process bits
    while counter2 < itterator2:
        # Shift
        bit2 = bit1
        bit1 = bit0
        bit0 = int(bits[counter2])
        
        # Calculate Logic
        out1 = bit0 ^ bit1 ^ bit2
        out2 = bit0 ^ bit2
        
        # Append output
        outstr = str(out1) + str(out2)
        convoludedpacket.append(outstr)
        
        counter2 += 1
    
    # 4. Join the list into a single string
    # (Replaced the manual while loop string builder with .join for speed/cleanliness)
    convPackStr = "".join(convoludedpacket)
    
    return convPackStr

# Modulation logic for QPSK encoding
def QPSKmodulate(datapacket, baud, simulationTimestep, carrierFrequency, currentSimulationTime):
    # 1. Validate Input Data (Pad with '0' if odd length)
    packet_str = str(datapacket)
    if len(packet_str) % 2 != 0:
        packet_str += '0'
        
    # 2. Setup Time Vector
    # Calculate theoretical total duration
    total_duration = (len(packet_str) / 2) * (1 / baud)
    packetEnd = currentSimulationTime + total_duration
    
    # Use np.arange for floating point steps
    time = np.arange(currentSimulationTime, packetEnd, simulationTimestep)

    # 3. Calculate Samples Per Bit
    sampPerSymbol = int(round((1 / baud) / simulationTimestep))

    # 4. Split Bits into I and Q
    bits = np.array([int(b) for b in packet_str])
    
    I_bits = bits[0::2] # Even bits
    Q_bits = bits[1::2] # Odd bits

    # Map 0 -> -1, 1 -> 1
    I_vals = 2 * I_bits - 1
    Q_vals = 2 * Q_bits - 1

    # 5. Upsample
    I_samples = np.repeat(I_vals, sampPerSymbol)
    Q_samples = np.repeat(Q_vals, sampPerSymbol)

    # 6. Safety Check (Truncate to minimum length)
    # We use min_len to ensure I, Q, and Time arrays are identical in size
    min_len = min(len(time), len(I_samples), len(Q_samples))
    
    time = time[:min_len]
    I_samples = I_samples[:min_len]
    Q_samples = Q_samples[:min_len]

    # 7. Modulation
    signal = I_samples * np.cos(2 * np.pi * carrierFrequency * time) - \
             Q_samples * np.sin(2 * np.pi * carrierFrequency * time)
    
    # 8. Calculate Next Simulation Time
    # We calculate the start time for the NEXT packet based on exactly 
    # how many samples we just generated to prevent drift.
    nextSimulationTime = currentSimulationTime + (min_len * simulationTimestep)
             
    return signal, nextSimulationTime

# Modulation logic for 16-QAM encoding
def QAM16modulate(datapacket, baud, simulationTimestep, carrierFrequency, currentSimulationTime):
    # 1. Validate Input Data (Pad to multiple of 4)
    packet_str = str(datapacket)
    remainder = len(packet_str) % 4
    if remainder != 0:
        packet_str += '0' * (4 - remainder)
        
    # 2. Setup Time Vector
    # 4 bits per symbol
    total_duration = (len(packet_str) / 4) * (1 / baud)
    packetEnd = currentSimulationTime + total_duration
    
    time = np.arange(currentSimulationTime, packetEnd, simulationTimestep)

    # 3. Calculate Samples Per Symbol
    sampPerSymbol = int(round((1 / baud) / simulationTimestep))

    # 4. Map Bits to Voltage Levels
    # We take chunks of 2 bits: '00'->-3, '01'->-1, '11'->+1, '10'->+3
    # This is a specific Gray Code mapping to minimize bit errors
    mapping_table = {
        '00': -3,
        '01': -1,
        '11': 1,
        '10': 3
    }

    # Split packet into chunks of 4 bits (e.g., "1001")
    # First 2 bits -> I, Last 2 bits -> Q
    I_vals = []
    Q_vals = []
    
    for i in range(0, len(packet_str), 4):
        chunk = packet_str[i:i+4]
        i_bits = chunk[0:2]
        q_bits = chunk[2:4]
        
        I_vals.append(mapping_table[i_bits])
        Q_vals.append(mapping_table[q_bits])

    # 5. Upsample (Repeat values)
    I_samples = np.repeat(I_vals, sampPerSymbol)
    Q_samples = np.repeat(Q_vals, sampPerSymbol)

    # 6. Safety Truncate
    min_len = min(len(time), len(I_samples), len(Q_samples))
    time = time[:min_len]
    I_samples = I_samples[:min_len]
    Q_samples = Q_samples[:min_len]

    # 7. Modulation (I*cos - Q*sin)
    # We normalize by dividing by sqrt(10) to keep average power similar to QPSK
    # But for raw logic, we can leave the levels as -3,-1,1,3. 
    signal = I_samples * np.cos(2 * np.pi * carrierFrequency * time) - \
             Q_samples * np.sin(2 * np.pi * carrierFrequency * time)
    
    # 8. Calculate Next Time
    nextSimulationTime = currentSimulationTime + (min_len * simulationTimestep)
             
    return signal, nextSimulationTime

# Add noise
def add_awgn_noise(signal_matrix, snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal matrix based on a specific SNR.
    
    Parameters:
        signal_matrix (np.array): The input signal (1D or 2D numpy array).
        snr_db (float): The desired Signal-to-Noise Ratio in Decibels (dB).
        
    Returns:
        np.array: The noisy signal matrix.
    """
    # 1. Calculate Signal Power
    # Power = Mean of squares
    signal_power = np.mean(signal_matrix ** 2)
    
    # 2. Calculate Noise Power required
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # Therefore: P_noise = P_signal / 10^(SNR_dB / 10)
    snr_linear = 10 ** (snr_db / 10)
    required_noise_power = signal_power / snr_linear
    
    # 3. Calculate Noise Standard Deviation (RMS voltage)
    # For Gaussian noise, Power = Variance = (StdDev)^2
    noise_sigma = np.sqrt(required_noise_power)
    
    # 4. Generate Noise
    # Creates a noise matrix with the exact same shape as the input signal
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=signal_matrix.shape)
    
    # 5. Add noise
    noisy_signal = signal_matrix + noise
    
    return noisy_signal

# HATA model with Religh fading
def apply_hata_and_rayleigh_fading(signal, fs, dist_km, freq_MHz, speed_kmph, tx_height_m=30, rx_height_m=1.5):
    """
    Applies Hata Path Loss (Large City) and Time-Varying Rayleigh Fading to a signal.

    Parameters:
        signal (np.array): The input signal vector (complex or float).
        fs (float): Sampling frequency in Hz.
        dist_km (float): Distance between Tx and Rx in Kilometers.
        freq_MHz (float): Carrier frequency in MHz (Valid range: 150-1500 MHz for Hata).
        speed_kmph (float): Receiver speed in km/h.
        tx_height_m (float): Transmitter height in meters (30-200m).
        rx_height_m (float): Receiver height in meters (1-10m).

    Returns:
        np.array: The faded, attenuated signal (Complex).
    """
    
    # --- PART 1: Hata Model for Large Cities (Path Loss) ---
    # Formula: L_dB = 69.55 + 26.16*log(f) - 13.82*log(h_tx) - a(h_rx) + (44.9 - 6.55*log(h_tx))*log(d)
    
    # 1. Calculate correction factor a(h_rx) for Large Cities
    # The formula changes based on frequency >= 300MHz or < 300MHz
    if freq_MHz >= 300:
        a_hr = 3.2 * (np.log10(11.75 * rx_height_m))**2 - 4.97
    else:
        a_hr = 8.29 * (np.log10(1.54 * rx_height_m))**2 - 1.1

    # 2. Calculate Hata Path Loss in dB
    # Note: log10 must be used
    term1 = 69.55
    term2 = 26.16 * np.log10(freq_MHz)
    term3 = -13.82 * np.log10(tx_height_m)
    term4 = -a_hr
    term5 = (44.9 - 6.55 * np.log10(tx_height_m)) * np.log10(dist_km)
    
    path_loss_db = term1 + term2 + term3 + term4 + term5
    
    # 3. Convert dB Loss to Linear Amplitude Scaling Factor
    # Gain_dB = -Loss_dB
    # V_out = V_in * 10^(Gain_dB / 20)  <-- Division by 20 for Voltage/Amplitude
    large_scale_attenuation = 10 ** (-path_loss_db / 20)
    
    print(f"Distance: {dist_km}km | Path Loss: {path_loss_db:.2f} dB")


    # --- PART 2: Rayleigh Fading with Doppler (Jakes' Model) ---
    # We simulate time-correlated fading based on speed.
    
    # 1. Calculate Maximum Doppler Shift (fd)
    # fd = (v / c) * f
    c = 3e8 # Speed of light m/s
    speed_ms = speed_kmph / 3.6 # Convert km/h to m/s
    freq_hz = freq_MHz * 1e6
    fd = (speed_ms / c) * freq_hz
    
    print(f"Speed: {speed_kmph} km/h | Doppler Freq: {fd:.2f} Hz")
    
    # 2. Generate Time Vector
    num_samples = len(signal)
    duration = num_samples / fs
    t = np.linspace(0, duration, num_samples)
    
    # 3. Sum-of-Sinusoids (Jakes' Simulator) to create Rayleigh Coefficient h(t)
    # This creates the "Tub" shape spectrum characteristic of mobile fading
    N_paths = 50 # Number of multipath rays (higher = better statistical accuracy)
    
    # Initialize complex fading vector
    h_rayleigh = np.zeros(num_samples, dtype=complex)
    
    # Generate N random paths arriving at different angles with random phases
    # We perform a summation of sinusoids
    for n in range(1, N_paths + 1):
        alpha_n = (2 * np.pi * n - np.pi) / (4 * N_paths)  # Angle of arrival
        phi_n = np.random.uniform(0, 2 * np.pi)            # Random phase
        theta_n = np.random.uniform(0, 2 * np.pi)          # Random phase
        
        # Doppler shift for this specific path
        doppler_shift = 2 * np.pi * fd * np.cos(alpha_n)
        
        # Accumulate the sinusoid
        h_rayleigh += np.exp(1j * (doppler_shift * t + phi_n))
        
    # Normalize power to 1 (0 dB gain) so we don't artificially amplify
    h_rayleigh = h_rayleigh / np.sqrt(N_paths)
    
    
    # --- PART 3: Combine ---
    # Output = Input Signal * Path Loss * Rayleigh Coefficient
    faded_signal = signal * large_scale_attenuation * h_rayleigh
    
    return faded_signal 

# Demodulation logic for QPSK encding
def QPSKdemodulate(signal, baud, simulationTimestep, carrierFrequency):
    # 1. Reconstruct Time Vector
    # We assume the signal starts at t=0 relative to the start of this packet
    t = np.arange(len(signal)) * simulationTimestep
    
    # 2. Generate Local Carriers
    # These must match the frequency of the transmitter
    carrier_I = np.cos(2 * np.pi * carrierFrequency * t)
    carrier_Q = np.sin(2 * np.pi * carrierFrequency * t)
    
    # 3. Down-conversion (Mixing)
    # Multiply received signal by local carriers to separate I and Q
    # Note: We multiply Q by negative sine because the modulator used (I*cos - Q*sin)
    product_I = signal * carrier_I
    product_Q = signal * -carrier_Q 

    # 4. Integrate and Dump (Low Pass Filter)
    # We need to average the signal over the duration of one symbol
    samples_per_symbol = int(round((1/baud) / simulationTimestep))
    
    # Calculate how many full symbols we have received
    num_symbols = len(signal) // samples_per_symbol
    
    # Truncate signal to fit perfectly into symbols (drops any trailing partial samples)
    limit = num_symbols * samples_per_symbol
    product_I = product_I[:limit]
    product_Q = product_Q[:limit]
    
    # Reshape into a matrix: [Number of Symbols x Samples per Symbol]
    # taking the .mean(axis=1) mathematically integrates over the symbol period
    I_averages = product_I.reshape(num_symbols, samples_per_symbol).mean(axis=1)
    Q_averages = product_Q.reshape(num_symbols, samples_per_symbol).mean(axis=1)
    
    # 5. Decision (Threshold Detection)
    # If average > 0, bit is 1. If average < 0, bit is 0.
    I_bits = (I_averages > 0).astype(int)
    Q_bits = (Q_averages > 0).astype(int)
    
    # 6. Reassemble (Parallel to Serial)
    # We create an empty array of the correct length (2 bits per symbol)
    detected_bits = np.zeros(num_symbols * 2, dtype=int)
    
    # Place I bits in even positions (0, 2, 4...)
    detected_bits[0::2] = I_bits
    
    # Place Q bits in odd positions (1, 3, 5...)
    detected_bits[1::2] = Q_bits
    
    # Convert integer array back to a string '10110...'
    return "".join(map(str, detected_bits))

# Demodulation logic for 16-QAM encding
def QAM16demodulate(signal, baud, simulationTimestep, carrierFrequency):
    # 1. Reconstruct Time Vector
    t = np.arange(len(signal)) * simulationTimestep
    
    # 2. Down-conversion
    carrier_I = np.cos(2 * np.pi * carrierFrequency * t)
    carrier_Q = np.sin(2 * np.pi * carrierFrequency * t)
    
    # Multiply (x2 is often used in math to recover original amplitudes)
    # Depending on your exact math, you might need to scale the result
    # Here we multiply by 2 to counteract the 1/2 from the cosine identity
    product_I = signal * carrier_I * 2
    product_Q = signal * -carrier_Q * 2

    # 3. Integrate and Dump
    samples_per_symbol = int(round((1/baud) / simulationTimestep))
    num_symbols = len(signal) // samples_per_symbol
    limit = num_symbols * samples_per_symbol
    
    # Get averages per symbol
    I_averages = product_I[:limit].reshape(num_symbols, samples_per_symbol).mean(axis=1)
    Q_averages = product_Q[:limit].reshape(num_symbols, samples_per_symbol).mean(axis=1)
    
    # 4. Decision Logic (The De-mapper)
    # We need to reverse the mapping:
    # < -2  --> -3 ('00')
    # -2 to 0 --> -1 ('01')
    # 0 to 2  --> +1 ('11')
    # > 2   --> +3 ('10')
    
    def decode_level(val):
        if val < -2: return '00'
        if val < 0:  return '01'
        if val < 2:  return '11'
        return '10'

    detected_bits = ""
    for i in range(num_symbols):
        i_str = decode_level(I_averages[i])
        q_str = decode_level(Q_averages[i])
        detected_bits += i_str + q_str
        
    return detected_bits

# Viterbi Decode to undo convolution

def viterbi_decode_packet(received_packet):
    """
    Decodes a single (7,5) convolutional packet.
    Returns the raw decoded bitstring (strips the 2-bit flush tail).
    """
    
    # 1. Setup the State Machine
    machine = {}
    states = [0, 1, 2, 3]
    for state in states:
        machine[state] = {}
        bit1 = (state >> 1) & 1
        bit2 = (state >> 0) & 1
        for input_bit in [0, 1]:
            bit0 = input_bit
            out1 = bit0 ^ bit1 ^ bit2
            out2 = bit0 ^ bit2
            output_bits = str(out1) + str(out2)
            next_state_val = (bit0 << 1) | bit1
            machine[state][input_bit] = (output_bits, next_state_val)

    # 2. Viterbi Algorithm
    path_metrics = {0: (0, ""), 1: (float('inf'), ""), 2: (float('inf'), ""), 3: (float('inf'), "")}
    
    # Process inputs in pairs (Rate 1/2)
    pairs = [received_packet[i:i+2] for i in range(0, len(received_packet), 2)]
    
    for pair in pairs:
        new_path_metrics = {0: (float('inf'), ""), 1: (float('inf'), ""), 2: (float('inf'), ""), 3: (float('inf'), "")}
        
        for state in states:
            current_score, current_history = path_metrics[state]
            if current_score == float('inf'): continue
                
            for input_bit in [0, 1]:
                expected_out, next_state = machine[state][input_bit]
                
                # Calculate Hamming Distance
                dist = 0
                if pair[0] != expected_out[0]: dist += 1
                if pair[1] != expected_out[1]: dist += 1
                
                new_score = current_score + dist
                
                # Compare & Select
                if new_score < new_path_metrics[next_state][0]:
                    new_path_metrics[next_state] = (new_score, current_history + str(input_bit))
                        
        path_metrics = new_path_metrics

    # 3. Traceback
    best_state = min(path_metrics, key=lambda k: path_metrics[k][0])
    full_decoded_bits = path_metrics[best_state][1]

    # 4. Remove Tail (Flush bits) only
    # We strip the last 2 bits because they were the "00" added by the 
    # encoder to flush the state machine, not part of the original message.
    if len(full_decoded_bits) > 2:
        decoded_data = full_decoded_bits[:-2]
    else:
        decoded_data = ""


    return decoded_data

# Dynamic Modulation Logic
def check_ber_pass(current_ber, ber_threshold):
    """
    Compares the current Bit Error Rate (BER) against a maximum threshold.
    
    Parameters:
        current_ber (float): The calculated BER from your simulation (e.g., 0.005).
        ber_threshold (float): The maximum allowed BER (e.g., 0.01 or 1e-3).
        
    Returns:
        int: 1 if the BER is acceptable (<= threshold), 0 if it failed (> threshold).
    """
    if current_ber <= ber_threshold:
        return 1
    else:
        return 0

#Headder and data extaraction
def extract_packet_data(decoded_bitstring):
    """
    Parses the raw decoded bitstring into its components.
    Structure: [8-bit Packet Num] + [32-bit Known String] + [Payload]
    
    Returns:
        (int, str, str): (Packet Number, Known String, Payload)
    """
    HEADER_SIZE = 8
    KNOWN_STR_SIZE = 32
    TOTAL_METADATA = HEADER_SIZE + KNOWN_STR_SIZE
    
    # Check if the packet is long enough to contain the headers
    if len(decoded_bitstring) < TOTAL_METADATA:
        # Return error values for corrupt/short packets
        return -1, "", ""
        
    # 1. Extract Packet Number (First 8 bits) -> Convert to Int
    header_bin = decoded_bitstring[:HEADER_SIZE]
    packet_number = int(header_bin, 2)
    
    # 2. Extract Known String (Next 32 bits) -> Keep as Binary String
    known_string = decoded_bitstring[HEADER_SIZE : TOTAL_METADATA]
    
    # 3. Extract Payload (The rest)
    payload_data = decoded_bitstring[TOTAL_METADATA:]
    
    return packet_number, known_string, payload_data

# Packet stitching and Data retrival
def bin2ascii(binary_data):
    
    # 1. Stitch the data if it is a list
    if isinstance(binary_data, list):
        full_binary_string = "".join(binary_data)
    else:
        full_binary_string = binary_data
        
    ascii_text = ""
    
    # 2. Iterate in chunks of 8 bits
    for i in range(0, len(full_binary_string), 8):
        byte_chunk = full_binary_string[i : i+8]
        
        # Only convert if we have a full byte (8 bits)
        # This ignores any trailing bits that don't make a full character
        if len(byte_chunk) == 8:
            decimal_value = int(byte_chunk, 2)
            ascii_text += chr(decimal_value)
            
    return ascii_text

# Calculate Bit error rate 
def calculate_ber(original_bits, received_bits):
    """
    Calculates the Bit Error Rate (BER) by comparing two binary strings.
    
    Parameters:
        original_bits (str): The transmitted binary string (e.g., "10110").
        received_bits (str): The demodulated binary string (e.g., "10010").
        
    Returns:
        tuple: (ber_value, number_of_errors)
    """
    # 1. Standardize Inputs to Strings
    # This allows the user to pass in lists of ints [0, 1] or strings "01"
    if isinstance(original_bits, (list, np.ndarray)):
        s_tx = "".join(map(str, original_bits))
    else:
        s_tx = str(original_bits)
        
    if isinstance(received_bits, (list, np.ndarray)):
        s_rx = "".join(map(str, received_bits))
    else:
        s_rx = str(received_bits)

    # 2. Handle Length Mismatches
    # In simulations, Rx might be shorter or longer due to delays or buffers.
    # We compare only the overlapping section.
    len_tx = len(s_tx)
    len_rx = len(s_rx)
    compare_len = min(len_tx, len_rx)
    
    if len_tx != len_rx:
        print(f"Warning: Length mismatch (Tx: {len_tx}, Rx: {len_rx}).")
        print(f"Comparing the first {compare_len} bits only.")

    if compare_len == 0:
        return 0.0, 0

    # 3. Count Errors
    error_count = 0
    for i in range(compare_len):
        if s_tx[i] != s_rx[i]:
            error_count += 1
            
    # 4. Calculate BER
    ber = error_count / compare_len
    
    return ber, error_count


# MAIN CODE

# Input data and simulation parameters
Data = 'from00923312511101;to00447878503732;20080312T1400Z;National Semiconductor fired 1725 people, Bernie Madoff plead guilty, America blew up 12 people in Pakistan, Sikorsky 92A crashed in Canada and killed 1, ISS astronauts shelterd from space debris in the Russian escape pod.'

downlinkCarrierFreq = 233500 # center of downlink for 3GPP relese 8 LTE band 7 /10000 to speed up simulation
uplinkCarrierFreq = 267000 # center of uplink for 3GPP relese 8 LTE band 7 /10000 to speed up simulation

alpha = 0.35 # filter rolloff

uplinkBaud = int(round((0.2*uplinkCarrierFreq)/(1+alpha), 2)/1.5) # calculate bandwith using design requirement given with a 50% margin
downlinkBaud = int(round((0.2*downlinkCarrierFreq)/(1+alpha), 2)/1.5) # calculate bandwith using design requirement given with a 50% margin

knownStr = str('test')
knownStrBIN, knownStrlen = ascii2bin(knownStr)

startTime = 0 
simulationTimeStep = 1/(10*uplinkCarrierFreq)

# Data conversion and packetization
DataBIN, DataLen = ascii2bin(Data)
packetized, nPackets = packetize(DataBIN, 64)
print(packetized)


# Transmit and recive loop (single transmission)
currentTime = startTime
rxDataBINPackets = list()
packetCounter = 0
while packetCounter < nPackets:
    convoludedData = convolutionEncodePacket(packetized[packetCounter], str(knownStrBIN))
    sig, newtime = QPSKmodulate(convoludedData[0], downlinkBaud, simulationTimeStep, downlinkCarrierFreq, currentTime)
    demod = QPSKdemodulate(sig, downlinkBaud, simulationTimeStep, downlinkCarrierFreq)
    decoded = viterbi_decode_packet(demod)
    print(decoded)
    rxDataBINPackets.append(decoded)
    
    currentTime = newtime
    packetCounter += 1

packetCounter = 0
rxDataBIN = str()
while packetCounter < nPackets:
    rxDataBIN += rxDataBINPackets[packetCounter]
    packetCounter += 1

print(bin2ascii(rxDataBIN))