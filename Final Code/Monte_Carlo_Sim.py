import textwrap
import numpy as np

# Utility function to convert an ASCII string into a binary string
def ascii2bin(str_input):
    # Convert each character to its 8-bit binary representation
    byte_list = [format(ord(char), '08b') for char in str_input]
    # Join all byte strings into a single string
    bin_str = "".join(byte_list)
    return bin_str, len(bin_str)

# Utility function to convert binary data back to an ASCII string
def bin2ascii(binary_data):
    # Handle case where input is a list of strings instead of a single string
    if isinstance(binary_data, list):
        full_binary_string = "".join(binary_data)
    else:
        full_binary_string = binary_data
        
    ascii_text = ""
    # Iterate through the binary string in chunks of 8 bits
    for i in range(0, len(full_binary_string), 8):
        byte_chunk = full_binary_string[i : i+8]
        # Only convert if we have a full 8-bit byte
        if len(byte_chunk) == 8:
            decimal_value = int(byte_chunk, 2)
            ascii_text += chr(decimal_value)
            
    return ascii_text

# Function to split a long binary string into smaller packets
def packetize(data, packetLength):
    dataLen = len(data)
    # Calculate if padding is needed to make the data length a multiple of packetLength
    remainder = dataLen % packetLength
    if remainder != 0:
        padding_needed = packetLength - remainder
        data = data + "0" * padding_needed
    
    # Wrap the data into a list of strings
    packets = textwrap.wrap(data, packetLength)
    return packets

# Function to parse the decoded bitstring and separate metadata from payload
def extract_packet_data(decoded_bitstring):
    HEADER_SIZE = 8
    KNOWN_STR_SIZE = 32
    TOTAL_METADATA = HEADER_SIZE + KNOWN_STR_SIZE
    
    # Return failure values if the bitstring is too short to contain metadata
    if len(decoded_bitstring) < TOTAL_METADATA:
        return -1, "", ""
        
    # Extract the packet number from the header
    header_bin = decoded_bitstring[:HEADER_SIZE]
    try:
        packet_number = int(header_bin, 2)
    except ValueError:
        packet_number = -1
    
    # Extract the known validation string and the actual payload
    known_string = decoded_bitstring[HEADER_SIZE : TOTAL_METADATA]
    payload_data = decoded_bitstring[TOTAL_METADATA:]
    
    return packet_number, known_string, payload_data

# Function to calculate the Bit Error Rate BER between sent and received bits
def calculate_ber(original_bits, received_bits):
    # Ensure inputs are string format for comparison
    if isinstance(original_bits, (list, np.ndarray)):
        s_tx = "".join(map(str, original_bits))
    else:
        s_tx = str(original_bits)
        
    if isinstance(received_bits, (list, np.ndarray)):
        s_rx = "".join(map(str, received_bits))
    else:
        s_rx = str(received_bits)

    len_tx = len(s_tx)
    len_rx = len(s_rx)
    # Compare only up to the length of the shorter string
    compare_len = min(len_tx, len_rx)
    
    if compare_len == 0:
        return 0.0, 0

    # Count mismatches
    error_count = 0
    for i in range(compare_len):
        if s_tx[i] != s_rx[i]:
            error_count += 1
            
    ber = error_count / compare_len
    return ber, error_count

# Function to generate Root Raised Cosine filter coefficients for pulse shaping
def generate_rrc_coeffs(alpha, sps, span):
    # Create time vector based on symbol span and samples per symbol
    t = np.arange(-span*sps//2, span*sps//2 + 1) / sps
    
    # Handle division by zero or invalid operations gracefully
    with np.errstate(divide='ignore', invalid='ignore'):
        coeffs = (np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))) / \
                 (np.pi * t * (1 - (4 * alpha * t)**2))

    # Handle the singularity at t=0
    coeffs[t == 0] = 1 - alpha + (4 * alpha / np.pi)
    
    # Handle singularities where the denominator becomes zero
    if alpha != 0:
        idx = np.where(np.abs(np.abs(t) - 1/(4*alpha)) < 1e-5)[0]
        if len(idx) > 0:
            val = (alpha / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))
            coeffs[idx] = val

    # Normalize the coefficients to unit energy
    coeffs = coeffs / np.sqrt(np.sum(coeffs**2))
    return coeffs

# Function to perform Convolutional Encoding on a packet
def convolutionEncodePacket(packet_data, packet_number, knownString):
    convoludedpacket = list()
    
    # Create header containing packet number and known string
    header = format(packet_number % 256, '08b') + knownString
    # Append flush bits 00 to reset the encoder state at the end
    currentpacket = header + str(packet_data) + "00" 
    
    bits = [int(b) for b in currentpacket]
    # Initialize shift register state
    bit0 = 0
    bit1 = 0
    bit2 = 0
    
    # Process bits through the shift register
    for b in bits:
        bit2 = bit1
        bit1 = bit0
        bit0 = b
        
        # Generator polynomials for the two outputs
        out1 = bit0 ^ bit1 ^ bit2
        out2 = bit0 ^ bit2
        
        convoludedpacket.append(str(out1) + str(out2))
    
    return "".join(convoludedpacket)

# Function to decode received bits using the Viterbi Algorithm
def viterbi_decode_packet(received_packet):
    # Precompute the state machine transitions and outputs
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

    # Initialize path metrics state 0 is the starting state
    path_metrics = {0: (0, ""), 1: (float('inf'), ""), 2: (float('inf'), ""), 3: (float('inf'), "")}
    
    # Process received bits in pairs
    pairs = [received_packet[i:i+2] for i in range(0, len(received_packet), 2)]
    
    for pair in pairs:
        if len(pair) < 2: continue 
        
        new_path_metrics = {0: (float('inf'), ""), 1: (float('inf'), ""), 2: (float('inf'), ""), 3: (float('inf'), "")}
        
        # Iterate through current states to find best path to next states
        for state in states:
            current_score, current_history = path_metrics[state]
            if current_score == float('inf'): continue
                
            for input_bit in [0, 1]:
                expected_out, next_state = machine[state][input_bit]
                
                # Calculate Hamming distance branch metric
                dist = 0
                if pair[0] != expected_out[0]: dist += 1
                if pair[1] != expected_out[1]: dist += 1
                
                new_score = current_score + dist
                
                # Update path if this route has a lower error score
                if new_score < new_path_metrics[next_state][0]:
                    new_path_metrics[next_state] = (new_score, current_history + str(input_bit))
                        
        path_metrics = new_path_metrics

    # Select the survivor path with the lowest accumulated error metric
    best_state = min(path_metrics, key=lambda k: path_metrics[k][0])
    full_decoded_bits = path_metrics[best_state][1]

    # Remove the flush bits from the end
    if len(full_decoded_bits) > 2:
        decoded_data = full_decoded_bits[:-2]
    else:
        decoded_data = ""

    return decoded_data

# Function to modulate data using QPSK
def QPSKmodulate(datapacket, baud, simulationTimestep, carrierFrequency, currentSimulationTime):
    packet_str = str(datapacket)
    # Ensure even number of bits for I/Q pairing
    if len(packet_str) % 2 != 0:
        packet_str += '0'
        
    sampPerSymbol = int(round((1 / baud) / simulationTimestep))
    
    # Map bits to NRZ values
    bits = np.array([int(b) for b in packet_str])
    I_bits = bits[0::2]
    Q_bits = bits[1::2]

    I_vals = 2 * I_bits - 1
    Q_vals = 2 * Q_bits - 1

    # Pad the sequence before filtering
    pad_len = 10
    I_vals = np.pad(I_vals, (pad_len, pad_len), 'constant', constant_values=0)
    Q_vals = np.pad(Q_vals, (pad_len, pad_len), 'constant', constant_values=0)

    # Generate pulse shaping filter
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, sampPerSymbol, span=6)
    
    # Upsample symbols to sample rate
    I_impulses = np.zeros(len(I_vals) * sampPerSymbol)
    Q_impulses = np.zeros(len(Q_vals) * sampPerSymbol)
    I_impulses[::sampPerSymbol] = I_vals
    Q_impulses[::sampPerSymbol] = Q_vals
    
    # Apply RRC filter
    I_shaped = np.convolve(I_impulses, rrc_taps, 'same')
    Q_shaped = np.convolve(Q_impulses, rrc_taps, 'same')

    # Mix with carrier frequency
    num_samples = len(I_shaped)
    rel_time = np.arange(num_samples) * simulationTimestep 

    signal = I_shaped * np.cos(2 * np.pi * carrierFrequency * rel_time) - \
             Q_shaped * np.sin(2 * np.pi * carrierFrequency * rel_time)
    
    nextSimulationTime = (num_samples * simulationTimestep) + currentSimulationTime
             
    return signal, nextSimulationTime

# Function to modulate data using 16-QAM
def QAM16modulate(datapacket, baud, simulationTimestep, carrierFrequency, currentSimulationTime):
    packet_str = str(datapacket)
    # Ensure bit count is a multiple of 4
    remainder = len(packet_str) % 4
    if remainder != 0:
        packet_str += '0' * (4 - remainder)
        
    sampPerSymbol = int(round((1 / baud) / simulationTimestep))

    # Define symbol mapping for 16-QAM
    mapping_table = {'00': -3, '01': -1, '11': 1, '10': 3}

    I_vals = []
    Q_vals = []
    
    # Map groups of 4 bits to I and Q levels
    for i in range(0, len(packet_str), 4):
        chunk = packet_str[i:i+4]
        i_bits = chunk[0:2]
        q_bits = chunk[2:4]
        I_vals.append(mapping_table[i_bits])
        Q_vals.append(mapping_table[q_bits])

    # Pad sequence
    pad_len = 10
    I_vals = np.pad(I_vals, (pad_len, pad_len), 'constant', constant_values=0)
    Q_vals = np.pad(Q_vals, (pad_len, pad_len), 'constant', constant_values=0)

    # Pulse shaping setup
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, sampPerSymbol, span=6)

    # Upsample
    I_impulses = np.zeros(len(I_vals) * sampPerSymbol)
    Q_impulses = np.zeros(len(Q_vals) * sampPerSymbol)
    I_impulses[::sampPerSymbol] = I_vals
    Q_impulses[::sampPerSymbol] = Q_vals

    # Filter
    I_shaped = np.convolve(I_impulses, rrc_taps, 'same')
    Q_shaped = np.convolve(Q_impulses, rrc_taps, 'same')

    num_samples = len(I_shaped)
    rel_time = np.arange(num_samples) * simulationTimestep

    # Mix with carrier and normalize power
    signal = (I_shaped * np.cos(2 * np.pi * carrierFrequency * rel_time) - \
              Q_shaped * np.sin(2 * np.pi * carrierFrequency * rel_time)) / np.sqrt(10)
    
    nextSimulationTime = (num_samples * simulationTimestep) + currentSimulationTime
             
    return signal, nextSimulationTime

# Function to demodulate QPSK signals
def QPSKdemodulate(signal, baud, simulationTimestep, carrierFrequency):
    t = np.arange(len(signal)) * simulationTimestep
    
    # Generate local carrier reference
    carrier_I = np.cos(2 * np.pi * carrierFrequency * t)
    carrier_Q = np.sin(2 * np.pi * carrierFrequency * t)
    
    # Downconvert to baseband
    product_I = signal * carrier_I * 2
    product_Q = signal * -carrier_Q * 2

    samples_per_symbol = int(round((1/baud) / simulationTimestep))
    
    # Matched filter using RRC
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, samples_per_symbol, span=6)
    
    filtered_I = np.convolve(product_I, rrc_taps, 'same')
    filtered_Q = np.convolve(product_Q, rrc_taps, 'same')
    
    # Downsample to symbol rate
    I_samples = filtered_I[::samples_per_symbol]
    Q_samples = filtered_Q[::samples_per_symbol]
    
    # Remove padding
    pad_len = 10
    if len(I_samples) > 2*pad_len:
        I_samples = I_samples[pad_len:-pad_len]
        Q_samples = Q_samples[pad_len:-pad_len]
    
    num_symbols = len(I_samples)
    
    # Decision slicing for QPSK
    I_bits = (I_samples > 0).astype(int)
    Q_bits = (Q_samples > 0).astype(int)
    
    # Interleave bits to reconstruct stream
    detected_bits = np.zeros(num_symbols * 2, dtype=int)
    detected_bits[0::2] = I_bits
    detected_bits[1::2] = Q_bits
    
    return "".join(map(str, detected_bits))

# Function to demodulate 16-QAM signals
def QAM16demodulate(signal, baud, simulationTimestep, carrierFrequency):
    t = np.arange(len(signal)) * simulationTimestep
    
    carrier_I = np.cos(2 * np.pi * carrierFrequency * t)
    carrier_Q = np.sin(2 * np.pi * carrierFrequency * t)
    
    scaling = 2 * np.sqrt(10)
    product_I = signal * carrier_I * scaling
    product_Q = signal * -carrier_Q * scaling

    samples_per_symbol = int(round((1/baud) / simulationTimestep))
    
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, samples_per_symbol, span=6)
    
    filtered_I = np.convolve(product_I, rrc_taps, 'same')
    filtered_Q = np.convolve(product_Q, rrc_taps, 'same')

    I_samples = filtered_I[::samples_per_symbol]
    Q_samples = filtered_Q[::samples_per_symbol]
    
    pad_len = 10
    if len(I_samples) > 2*pad_len:
        I_samples = I_samples[pad_len:-pad_len]
        Q_samples = Q_samples[pad_len:-pad_len]

    # Automatic Gain Control AGC to normalize levels
    all_samples = np.concatenate((I_samples, Q_samples))
    rx_power = np.mean(all_samples**2)
    
    if rx_power > 0:
        gain = np.sqrt(5 / rx_power)
        I_samples *= gain
        Q_samples *= gain
    
    num_symbols = len(I_samples)

    # Helper to slice 4-level symbols
    def decode_level(val):
        if val < -2: return '00'
        if val < 0:  return '01'
        if val < 2:  return '11'
        return '10'

    detected_bits = ""
    for i in range(num_symbols):
        i_str = decode_level(I_samples[i])
        q_str = decode_level(Q_samples[i])
        detected_bits += i_str + q_str
        
    return detected_bits

# Function to simulate Additive White Gaussian Noise
def add_awgn_noise(signal_matrix, snr_db, baud, fs):
    # Calculate signal power on active part of signal
    active_signal = signal_matrix[signal_matrix != 0]
    if len(active_signal) == 0: active_signal = signal_matrix
    
    signal_power = np.mean(active_signal ** 2)
    
    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10.0) 
    # Calculate noise bandwidth adjustment
    bandwidth_factor = fs / baud
    
    # Calculate required noise power
    required_noise_power = signal_power * bandwidth_factor / snr_linear
    required_noise_rms = np.sqrt(required_noise_power)
    
    # Generate noise and add to signal
    noise = np.random.normal(loc=0.0, scale=required_noise_rms, size=signal_matrix.shape)
    return signal_matrix + noise

# Function to simulate Path Loss and Rayleigh Fading
def apply_fading_and_loss(signal, fs, dist_km, freq_Hz, speed_kmph):
    # Calculate Free Space Path Loss
    c = 3e8
    wavelength = c / freq_Hz
    path_loss_linear = (4 * np.pi * (dist_km * 1000) / wavelength) ** 2
    attenuation = 1.0 / np.sqrt(path_loss_linear)
    
    # Calculate Doppler shift
    speed_ms = speed_kmph / 3.6 
    fd = (speed_ms / c) * freq_Hz
    
    num_samples = len(signal)
    duration = num_samples / fs
    t = np.linspace(0, duration, num_samples)
    
    # Simulate Rayleigh fading using Jakes model Sum of Sinusoids
    N_paths = 50 
    h_rayleigh = np.zeros(num_samples, dtype=complex)
    
    for n in range(1, N_paths + 1):
        alpha_n = (2 * np.pi * n - np.pi) / (4 * N_paths)
        phi_n = np.random.uniform(0, 2 * np.pi)
        doppler_shift = 2 * np.pi * fd * np.cos(alpha_n)
        h_rayleigh += np.exp(1j * (doppler_shift * t + phi_n))
        
    h_rayleigh = h_rayleigh / np.sqrt(N_paths)
    
    # Apply fading envelope to the signal
    faded_signal = signal * attenuation * np.abs(h_rayleigh)
    
    return faded_signal

if __name__ == "__main__":
    
    # Define the Message Data
    Data = r'[20080312T1400Z][frm]00923312511101[\frm][to]00447878503732[\to][MSG]National Semiconductor fired 1725 people, Bernie Madoff plead guilty, US drone strike kills 12 people in Pakistan, A Sikorsky 92A crashed in Canada and killed 1, ISS astronauts shelterd from space debris in the Russian escape pod[\MSG]'

    # Simulation Configuration and System Parameters
    downlinkCarrierFreq = 233500 
    uplinkCarrierFreq = 267000 
    alpha = 0.35 
    uplinkBaud = int(round((0.2*uplinkCarrierFreq)/(1+alpha), 2)/1.5) 
    downlinkBaud = int(round((0.2*downlinkCarrierFreq)/(1+alpha), 2)/1.5) 
    
    knownStr = 'test'
    knownStrBIN, _ = ascii2bin(knownStr)

    # Timing Configuration
    simulationTimeStep = 1/(10*uplinkCarrierFreq)
    fs = 1/simulationTimeStep

    # Input Data Preparation Phase
    DataBIN, DataLen = ascii2bin(Data)
    packet_size = 64 
    packetized = packetize(DataBIN, packet_size)
    nPackets = len(packetized)
    
    # Monte Carlo Simulation Constraints
    MAX_RETRIES = 16
    distance_km = 1
    speed_kmph = 50 
    ber_threshold = 0.001 
    
    SNR_VALUES = [2, 4, 6, 8, 12, 18, 24, 32]
    SIMULATION_REPEATS = 100

    # Print Simulation Header
    print("=" * 60)
    print(f"STARTING MONTE CARLO SIMULATION")
    print(f"Repeats per SNR: {SIMULATION_REPEATS}")
    print(f"Total Bits per Message: {len(DataBIN)}")
    print("=" * 60)

    final_results = []

    # Main Simulation Loop - Iterating through SNR values
    for snr in SNR_VALUES:
        print(f"\nProcessing SNR: {snr} dB...")
        
        snr_total_ber = 0
        snr_total_time = 0
        snr_efficiency_ratio = 0
        
        # Monte Carlo Iteration Phase
        for run in range(SIMULATION_REPEATS):
            
            currentTime = 0 
            rxDataBINPackets = list()
            
            use_16qam = False 
            last_ber = 0.0
            
            run_errors = 0
            run_retransmissions = 0
            
            packet_idx = 0
            # Packet Transmission Loop
            while packet_idx < nPackets:
                current_payload = packetized[packet_idx]
                
                retries = 0
                success = False
                
                # Automatic Repeat Request ARQ Logic
                while retries <= MAX_RETRIES and not success:
                    
                    # Convolutional Encoding Phase
                    convoludedData = convolutionEncodePacket(current_payload, packet_idx, str(knownStrBIN))
                    
                    # Adaptive Modulation Control Phase
                    if last_ber < ber_threshold and packet_idx > 0 and retries == 0:
                        use_16qam = True
                        mod_type = "16-QAM"
                    else:
                        use_16qam = False
                        mod_type = "QPSK" 
                    
                    # Signal Modulation Phase
                    if use_16qam:
                        sig, newtime = QAM16modulate(convoludedData, downlinkBaud, simulationTimeStep, downlinkCarrierFreq, currentTime)
                    else:
                        sig, newtime = QPSKmodulate(convoludedData, downlinkBaud, simulationTimeStep, downlinkCarrierFreq, currentTime)

                    # Channel Simulation Phase Fading and Path Loss
                    faded_sig = apply_fading_and_loss(sig, fs, distance_km, downlinkCarrierFreq, speed_kmph)
                    
                    # Channel Simulation Phase Noise Addition
                    noisy_sig = add_awgn_noise(faded_sig, snr, downlinkBaud, fs)

                    # Signal Demodulation Phase
                    if use_16qam:
                        demod_bits = QAM16demodulate(noisy_sig, downlinkBaud, simulationTimeStep, downlinkCarrierFreq)
                    else:
                        demod_bits = QPSKdemodulate(noisy_sig, downlinkBaud, simulationTimeStep, downlinkCarrierFreq)

                    # Error Correction Decoding Phase
                    decoded_packet_full = viterbi_decode_packet(demod_bits)
                    
                    # Packet Extraction and Metadata Parsing Phase
                    p_num, k_str, payload = extract_packet_data(decoded_packet_full)
                    
                    # Validation and Success Checking Phase
                    if k_str == knownStrBIN and p_num == packet_idx:
                        _, packet_errs = calculate_ber(current_payload, payload)
                        run_errors += packet_errs
                        last_ber = packet_errs / len(current_payload) if len(current_payload) > 0 else 0
                        success = True
                        currentTime = newtime 
                    else:
                        # Failure Handling and Retransmission Decision Phase
                        last_ber = 1.0 
                        retries += 1
                        run_retransmissions += 1
                        currentTime = newtime 
                
                # Packet Drop Handling
                if not success:
                    run_errors += len(current_payload)
                
                packet_idx += 1
            
            # Metrics Calculation Phase
            run_total_bits = len(DataBIN)
            run_final_ber = run_errors / run_total_bits
            
            total_transmissions = nPackets + run_retransmissions
            run_efficiency = nPackets / total_transmissions
            
            snr_total_ber += run_final_ber
            snr_total_time += currentTime
            snr_efficiency_ratio += run_efficiency
            
            print(".", end="", flush=True)

        # Averaging and Reporting Phase
        avg_ber = snr_total_ber / SIMULATION_REPEATS
        avg_time = snr_total_time / SIMULATION_REPEATS
        avg_efficiency = snr_efficiency_ratio / SIMULATION_REPEATS
        
        final_results.append({
            'snr': snr,
            'ber': avg_ber,
            'time': avg_time,
            'eff': avg_efficiency
        })
        
        print(f"\n--- RESULTS FOR SNR {snr} dB ---")
        print(f"Average Total BER: {avg_ber:.6f}")
        print(f"Average Transmission Time: {avg_time:.4f} s")
        print(f"Transmission Efficiency (Packets/TotalTx): {avg_efficiency:.4f}")
        print("-" * 50)

    # Final Result Compilation Phase
    print("\n" + "="*80)
    print(f"{'SNR (dB)':<10} | {'Avg BER':<15} | {'Avg Time (s)':<15} | {'Efficiency Ratio':<20}")
    print("-" * 80)
    for res in final_results:
        print(f"{res['snr']:<10} | {res['ber']:<15.6f} | {res['time']:<15.4f} | {res['eff']:<20.4f}")
    print("="*80)