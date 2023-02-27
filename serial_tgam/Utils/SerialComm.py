# Sensor I/O communication tool module
import serial
import serial.tools.list_ports


class Communication:

    # initialization
    def __init__(self, com, bps, timeout, name):
        self.port = com
        self.bps = bps
        self.timeout = timeout
        self.name = name
        try:
            self.main_engine = serial.Serial(self.port, self.bps, timeout=self.timeout)
        except Exception as e:
            print("---error---：", e)

    # Print basic information of the device
    def Print_Name(self):
        print(self.name)  # Custom device name
        print(self.main_engine.name)  # device name
        print(self.main_engine.port)  # Read or write port
        print(self.main_engine.baudrate)  # Baud rate
        print(self.main_engine.bytesize)  # Byte size
        print(self.main_engine.parity)  # Check bit
        print(self.main_engine.stopbits)  # Stop bit
        print(self.main_engine.timeout)  # Read timeout setting
        print(self.main_engine.writeTimeout)  # Write timeout setting
        print(self.main_engine.xonxoff)  # Software flow control
        print(self.main_engine.rtscts)  # Software flow control
        print(self.main_engine.dsrdtr)  # Hardware flow control
        print(self.main_engine.interCharTimeout)  # Character interval timeout

    # Open serial port
    def Open_Engine(self):
        self.main_engine.open()

    # Close the serial port
    def Close_Engine(self):
        self.main_engine.close()
        # Verify whether the serial port is open
        print('Serial port status：', self.main_engine.is_open)

    # Print the list of available serial ports
    @staticmethod
    def Print_Used_Com():
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)

    # Judge whether the serial port is open
    def isOpen_Engine(self):
        print(self.main_engine.is_open)

    # Receive data of specified size
    # Read size bytes from the serial port. If a timeout is specified, fewer bytes may be returned after the timeout; If no timeout is specified, it will wait until the specified number of bytes is received。
    def Read_Size(self, size):
        return self.main_engine.read(size=size)

    # Receive a row of data
    def Read_Line(self):
        return self.main_engine.readline()

    # Send data
    def Send_data(self, data):
        self.main_engine.write(data)

    # Receive buffer content bytes
    def In_Waiting(self):
        return self.main_engine.in_waiting