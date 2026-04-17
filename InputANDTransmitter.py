import math
import textwrap
import numpy as np
import matplotlib.pyplot as plt

# Data conversion
def ascii2bin(str):
    # input string of ASCII characters and returns string of binary bits with no deliminatrors 
    byte  = [format(ord(char), '08b') for char in str]
    bin = "".join(byte)
    lenbin = len(bin)
    return bin, lenbin

# Data packetization
# Split the data into sections of 64 bits and add null characters to the end to make them even
def packetize(data, packetLength):
    # input data 
    # math and textwrap packages required
    dataLen = len(data)
    nPackets = math.ceil(dataLen/packetLength)
    data = data + "0"*(int(packetLength * ( nPackets-dataLen/packetLength)))
    packets = textwrap.wrap(data, packetLength)
    return packets, nPackets

# Covolutional encoding
def convolutionEncode(data):
    # applies a rate 1/2 K=3 7,5 convolution to each packet
    # textwrap package required
    convoluded = list()

    itterator1 = len(data)
    counter1 = 0
    while counter1 < itterator1:
        convoludedpacket = list()

        currentpacket = str(ascii2bin(counter1)) + str(data[counter1]) + "00" # Join packet number, main data, and tail flush  
        
        bits = textwrap.wrap(currentpacket, 1)

        bit0 = 0
        bit1 = 0
        bit2 = 0
        
        itterator2 = len(bits)
        counter2 = 0
        while counter2 < itterator2:
            bit2 = bit1
            bit1 = bit0
            bit0 = int(bits[counter2])
            
            out1 = bit0^bit1^bit2
            out2 = bit0^bit2
            outstr = str(out1) + str(out2)
            convoludedpacket.append(outstr)
            
            counter2 += 1
        
        convPackStr = str()

        itterator2 = len(convoludedpacket)
        counter2 = 0
        while counter2 < itterator2:
            convPackStr = convPackStr + convoludedpacket[counter2]
            counter2 += 1
        
        convoluded.append(convPackStr)
        
        counter1 += 1
    return convoluded    

# Modulation logic
# QPSK encoding
def QPSKmodulate (datapacket, baud, simulationTimestep, carrierFrequerncy, currentSimulationTime):
    # input data is modulated using a QPSK algorithm
    packetEnd = currentSimulationTime + (len(datapacket)/2)*(1/baud)
    time = list(range(currentSimulationTime, packetEnd, simulationTimeStep))

    sampPerBit = (1/baud)/simulationTimestep
    
    bits = textwrap.wrap(str(datapacket), 1)
    
    I_samples = list()
    Q_samples = list()

    iterator1 = len(bits)
    counter1 = 0
    while counter1 < iterator1:
        currentBit = bits[counter1]

        counter2 = 0
        if counter1 % 2 == 0:
            while counter2 < sampPerBit:
                I_samples.append(2*int(currentBit)-1)
                counter2 += 1
        else:
            while counter2 < sampPerBit:
                Q_samples.append(2*int(currentBit)-1)
                counter2 += 1
        counter1 += 1
    
    signal = I_samples*np.cos(2*np.pi*carrierFrequerncy*time) - Q_samples*np.sin(2*np.pi*carrierFrequerncy*time)
    return signal




# input data and simulation parameters
Data = "from00923312511101;to00447878503732;20080312T1400Z;National Semiconductor fired 1725 people, Bernie Madoff plead guilty, America blew up 12 people in Pakistan, Sikorsky 92A crashed in Canada and killed 1, ISS astronauts shelterd from space debris in the Russian escape pod."

downlinkCarrierFreq = 233500 # center of downlink for 3GPP relese 8 LTE band 7 /10000 to speed up simulation
uplinkCarrierFreq = 267000 # center of uplink for 3GPP relese 8 LTE band 7 /10000 to speed up simulation

alpha = 0.35 # filter rolloff

uplinkBaud = int(round((0.2*uplinkCarrierFreq)/(1+alpha), 2)/1.5) # calculate bandwith using design requirement given with a 50% margin
downlinkBaud = int(round((0.2*downlinkCarrierFreq)/(1+alpha), 2)/1.5) # calculate bandwith using design requirement given with a 50% margin

simulationTimeStep = 1/(10*uplinkCarrierFreq)


# Data conversion
DataBIN, DataLen = ascii2bin(Data)

# Data packetization
# Split the data into sections of 64 bits and add null characters to the end to make them even
packetized, nPackets = packetize(DataBIN, 64)
print(packetized)

# Covolutional coding
convoludedData = convolutionEncode(packetized)
print(convoludedData)
# Modulation logic


# Bandwith filtering
