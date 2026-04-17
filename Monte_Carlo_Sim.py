import textwrap
import numpy as np

# --- HELPER FUNCTIONS ---

def ascii2bin(str_input):
    """Input string of ASCII characters and returns string of binary bits."""
    byte_list = [format(ord(char), '08b') for char in str_input]
    bin_str = "".join(byte_list)
    return bin_str, len(bin_str)

def bin2ascii(binary_data):
    if isinstance(binary_data, list):
        full_binary_string = "".join(binary_data)
    else:
        full_binary_string = binary_data
        
    ascii_text = ""
    for i in range(0, len(full_binary_string), 8):
        byte_chunk = full_binary_string[i : i+8]
        if len(byte_chunk) == 8:
            decimal_value = int(byte_chunk, 2)
            ascii_text += chr(decimal_value)
            
    return ascii_text

def packetize(data, packetLength):
    """input data split into a sections of length packetLength"""
    dataLen = len(data)
    remainder = dataLen % packetLength
    if remainder != 0:
        padding_needed = packetLength - remainder
        data = data + "0" * padding_needed
    
    packets = textwrap.wrap(data, packetLength)
    return packets

def extract_packet_data(decoded_bitstring):
    HEADER_SIZE = 8
    KNOWN_STR_SIZE = 32
    TOTAL_METADATA = HEADER_SIZE + KNOWN_STR_SIZE
    
    if len(decoded_bitstring) < TOTAL_METADATA:
        return -1, "", ""
        
    header_bin = decoded_bitstring[:HEADER_SIZE]
    try:
        packet_number = int(header_bin, 2)
    except ValueError:
        packet_number = -1
    
    known_string = decoded_bitstring[HEADER_SIZE : TOTAL_METADATA]
    payload_data = decoded_bitstring[TOTAL_METADATA:]
    
    return packet_number, known_string, payload_data

def calculate_ber(original_bits, received_bits):
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
    compare_len = min(len_tx, len_rx)
    
    if compare_len == 0:
        return 0.0, 0

    error_count = 0
    for i in range(compare_len):
        if s_tx[i] != s_rx[i]:
            error_count += 1
            
    ber = error_count / compare_len
    return ber, error_count

# --- FILTERING ---

def generate_rrc_coeffs(alpha, sps, span):
    t = np.arange(-span*sps//2, span*sps//2 + 1) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        coeffs = (np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))) / \
                 (np.pi * t * (1 - (4 * alpha * t)**2))
    coeffs[t == 0] = 1 - alpha + (4 * alpha / np.pi)
    if alpha != 0:
        idx = np.where(np.abs(np.abs(t) - 1/(4*alpha)) < 1e-5)[0]
        if len(idx) > 0:
            val = (alpha / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))
            coeffs[idx] = val
    coeffs = coeffs / np.sqrt(np.sum(coeffs**2))
    return coeffs

# --- ENCODING & DECODING ---

def convolutionEncodePacket(packet_data, packet_number, knownString):
    convoludedpacket = list()
    header = format(packet_number % 256, '08b') + knownString
    currentpacket = header + str(packet_data) + "00" 
    bits = [int(b) for b in currentpacket]
    bit0 = 0
    bit1 = 0
    bit2 = 0
    for b in bits:
        bit2 = bit1
        bit1 = bit0
        bit0 = b
        out1 = bit0 ^ bit1 ^ bit2
        out2 = bit0 ^ bit2
        convoludedpacket.append(str(out1) + str(out2))
    return "".join(convoludedpacket)

def viterbi_decode_packet(received_packet):
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

    path_metrics = {0: (0, ""), 1: (float('inf'), ""), 2: (float('inf'), ""), 3: (float('inf'), "")}
    pairs = [received_packet[i:i+2] for i in range(0, len(received_packet), 2)]
    
    for pair in pairs:
        if len(pair) < 2: continue
        new_path_metrics = {0: (float('inf'), ""), 1: (float('inf'), ""), 2: (float('inf'), ""), 3: (float('inf'), "")}
        for state in states:
            current_score, current_history = path_metrics[state]
            if current_score == float('inf'): continue
            for input_bit in [0, 1]:
                expected_out, next_state = machine[state][input_bit]
                dist = 0
                if pair[0] != expected_out[0]: dist += 1
                if pair[1] != expected_out[1]: dist += 1
                new_score = current_score + dist
                if new_score < new_path_metrics[next_state][0]:
                    new_path_metrics[next_state] = (new_score, current_history + str(input_bit))
        path_metrics = new_path_metrics

    best_state = min(path_metrics, key=lambda k: path_metrics[k][0])
    full_decoded_bits = path_metrics[best_state][1]
    if len(full_decoded_bits) > 2:
        decoded_data = full_decoded_bits[:-2]
    else:
        decoded_data = ""
    return decoded_data

# --- MODULATION ---

def QPSKmodulate(datapacket, baud, simulationTimestep, carrierFrequency, currentSimulationTime):
    packet_str = str(datapacket)
    if len(packet_str) % 2 != 0:
        packet_str += '0'
    sampPerSymbol = int(round((1 / baud) / simulationTimestep))
    bits = np.array([int(b) for b in packet_str])
    I_bits = bits[0::2]
    Q_bits = bits[1::2]
    I_vals = 2 * I_bits - 1
    Q_vals = 2 * Q_bits - 1
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, sampPerSymbol, span=6)
    I_impulses = np.zeros(len(I_vals) * sampPerSymbol)
    Q_impulses = np.zeros(len(Q_vals) * sampPerSymbol)
    I_impulses[::sampPerSymbol] = I_vals
    Q_impulses[::sampPerSymbol] = Q_vals
    I_shaped = np.convolve(I_impulses, rrc_taps, 'same')
    Q_shaped = np.convolve(Q_impulses, rrc_taps, 'same')
    num_samples = len(I_shaped)
    time = np.arange(num_samples) * simulationTimestep + currentSimulationTime
    signal = I_shaped * np.cos(2 * np.pi * carrierFrequency * time) - \
             Q_shaped * np.sin(2 * np.pi * carrierFrequency * time)
    nextSimulationTime = time[-1] + simulationTimestep
    return signal, nextSimulationTime

def QAM16modulate(datapacket, baud, simulationTimestep, carrierFrequency, currentSimulationTime):
    packet_str = str(datapacket)
    remainder = len(packet_str) % 4
    if remainder != 0:
        packet_str += '0' * (4 - remainder)
    sampPerSymbol = int(round((1 / baud) / simulationTimestep))
    mapping_table = {'00': -3, '01': -1, '11': 1, '10': 3}
    I_vals = []
    Q_vals = []
    for i in range(0, len(packet_str), 4):
        chunk = packet_str[i:i+4]
        i_bits = chunk[0:2]
        q_bits = chunk[2:4]
        I_vals.append(mapping_table[i_bits])
        Q_vals.append(mapping_table[q_bits])
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, sampPerSymbol, span=6)
    I_impulses = np.zeros(len(I_vals) * sampPerSymbol)
    Q_impulses = np.zeros(len(Q_vals) * sampPerSymbol)
    I_impulses[::sampPerSymbol] = I_vals
    Q_impulses[::sampPerSymbol] = Q_vals
    I_shaped = np.convolve(I_impulses, rrc_taps, 'same')
    Q_shaped = np.convolve(Q_impulses, rrc_taps, 'same')
    num_samples = len(I_shaped)
    time = np.arange(num_samples) * simulationTimestep + currentSimulationTime
    signal = (I_shaped * np.cos(2 * np.pi * carrierFrequency * time) - \
             Q_shaped * np.sin(2 * np.pi * carrierFrequency * time)) / np.sqrt(10)
    nextSimulationTime = time[-1] + simulationTimestep
    return signal, nextSimulationTime

def QPSKdemodulate(signal, baud, simulationTimestep, carrierFrequency):
    t = np.arange(len(signal)) * simulationTimestep
    carrier_I = np.cos(2 * np.pi * carrierFrequency * t)
    carrier_Q = np.sin(2 * np.pi * carrierFrequency * t)
    product_I = signal * carrier_I * 2
    product_Q = signal * -carrier_Q * 2
    samples_per_symbol = int(round((1/baud) / simulationTimestep))
    alpha = 0.35
    rrc_taps = generate_rrc_coeffs(alpha, samples_per_symbol, span=6)
    filtered_I = np.convolve(product_I, rrc_taps, 'same')
    filtered_Q = np.convolve(product_Q, rrc_taps, 'same')
    I_samples = filtered_I[::samples_per_symbol]
    Q_samples = filtered_Q[::samples_per_symbol]
    num_symbols = len(I_samples)
    I_bits = (I_samples > 0).astype(int)
    Q_bits = (Q_samples > 0).astype(int)
    detected_bits = np.zeros(num_symbols * 2, dtype=int)
    detected_bits[0::2] = I_bits
    detected_bits[1::2] = Q_bits
    return "".join(map(str, detected_bits))

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
    num_symbols = len(I_samples)
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

# --- CHANNEL MODELS ---

def add_awgn_noise(signal_matrix, snr_db):
    signal_power = np.mean(signal_matrix ** 2)
    snr_linear = 10 ** (snr_db / 10)
    required_noise_power = signal_power / snr_linear
    noise_sigma = np.sqrt(required_noise_power)
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=signal_matrix.shape)
    return signal_matrix + noise

def apply_hata_and_rayleigh_fading(signal, fs, dist_km, freq_MHz, speed_kmph):
    tx_height_m = 30
    rx_height_m = 1.5
    if freq_MHz >= 300:
        a_hr = 3.2 * (np.log10(11.75 * rx_height_m))**2 - 4.97
    else:
        a_hr = 8.29 * (np.log10(1.54 * rx_height_m))**2 - 1.1
    term1 = 69.55
    term2 = 26.16 * np.log10(freq_MHz)
    term3 = -13.82 * np.log10(tx_height_m)
    term4 = -a_hr
    term5 = (44.9 - 6.55 * np.log10(tx_height_m)) * np.log10(dist_km)
    path_loss_db = term1 + term2 + term3 + term4 + term5
    large_scale_attenuation = 10 ** (-path_loss_db / 20)
    c = 3e8 
    speed_ms = speed_kmph / 3.6 
    freq_hz = freq_MHz * 1e6
    fd = (speed_ms / c) * freq_hz
    num_samples = len(signal)
    duration = num_samples / fs
    t = np.linspace(0, duration, num_samples)
    N_paths = 50 
    h_rayleigh = np.zeros(num_samples, dtype=complex)
    for n in range(1, N_paths + 1):
        alpha_n = (2 * np.pi * n - np.pi) / (4 * N_paths)
        phi_n = np.random.uniform(0, 2 * np.pi)
        doppler_shift = 2 * np.pi * fd * np.cos(alpha_n)
        h_rayleigh += np.exp(1j * (doppler_shift * t + phi_n))
    h_rayleigh = h_rayleigh / np.sqrt(N_paths)
    fading_amplitude = np.abs(h_rayleigh) 
    faded_signal = signal * large_scale_attenuation * fading_amplitude
    return faded_signal 




# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # --- 1. SETUP FIXED PARAMETERS (Outside Loop) ---
    Data = 'from00923312511101;to00447878503732;20080312T1400Z;National Semiconductor fired 1725 people, Bernie Madoff plead guilty, America blew up 12 people in Pakistan, Sikorsky 92A crashed in Canada and killed 1, ISS astronauts shelterd from space debris in the Russian escape pod.'
    
    downlinkCarrierFreq = 233500 
    uplinkCarrierFreq = 267000 
    alpha = 0.35 
    uplinkBaud = int(round((0.2*uplinkCarrierFreq)/(1+alpha), 2)/1.5) 
    downlinkBaud = int(round((0.2*downlinkCarrierFreq)/(1+alpha), 2)/1.5) 
    
    knownStr = 'test'
    knownStrBIN, _ = ascii2bin(knownStr)

    simulationTimeStep = 1/(10*uplinkCarrierFreq)
    fs = 1/simulationTimeStep

    DataBIN, DataLen = ascii2bin(Data)
    packet_size = 64 
    packetized = packetize(DataBIN, packet_size)
    nPackets = len(packetized)
    
    # Simulation Constants
    MAX_RETRIES = 16
    distance_km = 0.5
    speed_kmph = 100 
    ber_threshold = 0.001 
    
    SNR_VALUES = [2, 4, 6, 12, 18, 24]
    SIMULATION_REPEATS = 100

    print("=" * 60)
    print(f"STARTING MONTE CARLO SIMULATION")
    print(f"Repeats per SNR: {SIMULATION_REPEATS}")
    print(f"Total Bits per Message: {len(DataBIN)}")
    print("=" * 60)

    # Store results for final table
    final_results = []

    # --- 2. SNR LOOP (Outer) ---
    for snr in SNR_VALUES:
        print(f"\nProcessing SNR: {snr} dB...")
        
        # Accumulators for this SNR
        snr_total_ber = 0
        snr_total_time = 0
        snr_efficiency_ratio = 0
        
        # --- 3. REPETITION LOOP (Inner) ---
        for run in range(SIMULATION_REPEATS):
            
            # Reset Run-Specific Variables
            currentTime = 0 
            rxDataBINPackets = list()
            
            use_16qam = False 
            last_ber = 0.0
            
            run_errors = 0
            run_retransmissions = 0
            
            packet_idx = 0
            while packet_idx < nPackets:
                current_payload = packetized[packet_idx]
                
                # ARQ Retry Loop
                retries = 0
                success = False
                
                while retries <= MAX_RETRIES and not success:
                    # Encoding
                    convoludedData = convolutionEncodePacket(current_payload, packet_idx, str(knownStrBIN))
                    
                    # Dynamic Modulation
                    if last_ber < ber_threshold and packet_idx > 0 and retries == 0:
                        use_16qam = True
                    else:
                        use_16qam = False
                        
                    if use_16qam:
                        sig, newtime = QAM16modulate(convoludedData, downlinkBaud, simulationTimeStep, downlinkCarrierFreq, currentTime)
                    else:
                        sig, newtime = QPSKmodulate(convoludedData, downlinkBaud, simulationTimeStep, downlinkCarrierFreq, currentTime)

                    # Channel
                    faded_sig = apply_hata_and_rayleigh_fading(sig, fs, distance_km, downlinkCarrierFreq/1e6, speed_kmph)
                    noisy_sig = add_awgn_noise(faded_sig, snr) # Use current loop SNR

                    # Demodulation
                    if use_16qam:
                        demod_bits = QAM16demodulate(noisy_sig, downlinkBaud, simulationTimeStep, downlinkCarrierFreq)
                    else:
                        demod_bits = QPSKdemodulate(noisy_sig, downlinkBaud, simulationTimeStep, downlinkCarrierFreq)

                    # Decoding & Check
                    decoded_packet_full = viterbi_decode_packet(demod_bits)
                    p_num, k_str, payload = extract_packet_data(decoded_packet_full)
                    
                    if k_str == knownStrBIN and p_num == packet_idx:
                        # Success
                        _, packet_errs = calculate_ber(current_payload, payload)
                        run_errors += packet_errs
                        last_ber = packet_errs / len(current_payload) if len(current_payload) > 0 else 0
                        success = True
                        currentTime = newtime 
                    else:
                        # Fail
                        last_ber = 1.0 
                        retries += 1
                        run_retransmissions += 1
                        currentTime = newtime 
                
                # Check Dropped Packet
                if not success:
                    # Add 100% error for this payload size
                    run_errors += len(current_payload)
                
                packet_idx += 1
            
            # End of Run Calculations
            run_total_bits = len(DataBIN) # Approximated as sum of payloads
            run_final_ber = run_errors / run_total_bits
            
            # Efficiency = nPackets / Total Transmissions (Packets + Retries)
            total_transmissions = nPackets + run_retransmissions
            run_efficiency = nPackets / total_transmissions
            
            # Add to Accumulators
            snr_total_ber += run_final_ber
            snr_total_time += currentTime
            snr_efficiency_ratio += run_efficiency
            
            # Optional: Print progress dot every 10 runs
            print(".", end="", flush=True)

        # --- 4. AVERAGE & PRINT RESULTS FOR THIS SNR ---
        avg_ber = snr_total_ber / SIMULATION_REPEATS
        avg_time = snr_total_time / SIMULATION_REPEATS
        avg_efficiency = snr_efficiency_ratio / SIMULATION_REPEATS
        
        # Store for final summary
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
        print("-" * 40)

    # --- 5. FINAL COMPARISON TABLE ---
    print("\n" + "="*80)
    print(f"{'SNR (dB)':<10} | {'Avg BER':<15} | {'Avg Time (s)':<15} | {'Efficiency Ratio':<20}")
    print("-" * 80)
    for res in final_results:
        print(f"{res['snr']:<10} | {res['ber']:<15.6f} | {res['time']:<15.4f} | {res['eff']:<20.4f}")
    print("="*80)