import numpy as np


# Sensor I/O message entity class

class Message:

    def resetMessage(self):
        print('Reset serial port data')

    def reciveMessage(self):
        print('Receive a hexadecimal serial port data')

    def parseMessage(self):
        print('Parse the received data')


class NeuroSkyMess(Message):
    # initialization
    def __init__(self):
        self.AA = 0  # SYNC byte, expressed in hexadecimal
        self.pLength = 0  # Effective data length
        self.payload = np.zeros(1024, dtype=int)  # Received content, characters
        self.rePoint = 0  # Payload subscript pointer
        self.checksum = 0  # Checksum

    # Data Object Reset
    def resetMessage(self):
        self.AA = 0
        self.pLength = 0
        self.payload[:] = 0
        self.rePoint = 0
        self.checksum = 0
        return True

    # Judge whether the received message is correct
    def reciveMessage(self, ch):
        if 0 < self.pLength:
            if self.rePoint < self.pLength:
                self.payload[self.rePoint] = ch
                self.rePoint += 1  # 指针后移
                self.checksum = self.checksum + ch
            else:  # Check bit reached
                # The lower 8 bits in the calibrator are inverted, and the check bits are calculated
                self.checksum &= 0xff
                self.checksum = ~self.checksum & 0xff
                if self.checksum == ch:  # Compare with check digit
                    # Verification succeeded
                    # Parsing data in payload [], DataRows
                    # +++++++++++++++++++++++++++++++++++++++++++++#
                    return self.parsePayload(self.payload, self.pLength)
                    # +++++++++++++++++++++++++++++++++++++++++++++#
                # Verification failed. After data reading, reset this data
                self.resetMessage()  # Reset this data
        else:  # PLength has not been received
            if 0 > self.pLength:  # 2 SYNCs have been received, and the data has been received
                if 170 > ch:  # pLength is normal
                    # Receive pLength word
                    self.pLength = ch  # Receive pLength, and reassign
                elif 170 < ch:  # PLength is too large, restart receiving data
                    self.resetMessage()
            else:  # No 2 SYNCs received
                if 0xAA == ch:
                    if 0xAA == self.AA:
                        # Received 2 SYNCs, set the flag to - 99, and start receiving data next time
                        self.pLength = -99
                    else:
                        # The previous one is not SYNC, so take this reception as the first SYNC
                        self.AA = 0xAA
                else:
                    self.AA = 0

    # Parse data
    def parsePayload(self, payload, pLength):
        resList = []  # Array type list, last as return value
        bytesParsed = 0
        code = 0
        length = 0  # vlength
        extendedCodeLevel = 0  # Extended code level
        # ---------Loop until all bytes are parsed from the payload[] array...-------- #
        while bytesParsed < pLength:
            # -----------Parse the extendedCodeLevel, code, and length--------------- #
            extendedCodeLevel = 0
            while payload[bytesParsed] == 0x55:
                extendedCodeLevel += 1
                bytesParsed += 1
            code = payload[bytesParsed]  # CODE
            bytesParsed += 1
            if code & 0x80:
                length = payload[bytesParsed]  # If [CODE]>=0x80, interpret the next byte as [VLENGTH]
                bytesParsed += 1
            else:
                length = 1  # Single-byte variable, VLENGTH = 1
            # Based on the extendedCodeLevel, code, length,
            #  and the[CODE] Definitions Table,  handle the next *"length" bytes of data from
            #  the payload as *appropriate for your application.
            # printf("EXCODE level: %d CODE: 0x%02X length: %d\n", extendedCodeLevel, code, length);
            # printf("Data value(s):");
            # ---------------return result：CODE，DataRow----------------------#
            DataRow = np.zeros(length + 1, dtype=int)
            for index in range(DataRow.size - 1):
                DataRow[index + 1] = payload[bytesParsed + index] & 0xFF
            DataRow[0] = code
            # ---------Increment the bytesParsed by the length of the Data Value----------- #
            bytesParsed += length
            resList.append(DataRow)  # return data
        return resList
