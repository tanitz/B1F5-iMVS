# import serial
import socket
import datetime
import time

# def connect_to_serial_port(port="COM3", baud_rate=115200, timeout=1):
#     ser = serial.Serial(port, baud_rate, timeout=timeout)
#     return ser

def send_scpi_command(ser, command):
    ser.write(command.encode() + b'\n')

def read_scpi_response(ser):
    result = ser.readline().decode().strip('OK/nOK/nOK/n')
    final = result.split('ADC')
    num = float(final[0])
    num = int(num * 1000)
    return num

def measure_current(ser):
    scpi_cmd = "conf:curr"
    send_scpi_command(ser, scpi_cmd)
    print("Current configuration set.")
    time.sleep(1)  # 2-second interval
    scpi_cmd = "meas:show?"
    send_scpi_command(ser, scpi_cmd)

    response = read_scpi_response(ser)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Instrument response for current measurement: {response}")

    with open("instrument_log.txt", "a") as file:
        file.write(f"[{timestamp}] Current Measurement:  {response}\n")

def send_data_to_receiver(ip, port, response):
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_address = (ip, port)
    sender_socket.sendto(str(response).encode(), receiver_address)
    sender_socket.close()
    

def main():
    receiver_ip = "127.0.0.1"
    receiver_port = 139
    # ser1 = connect_to_serial_port()
    print("Connected to the instruments.")
    
    try:
        while True:
            # response1 = measure_current(ser1)
            #response2 = measure_current(ser2)
            
            #response2s = str(response2)
            send_data_to_receiver(receiver_ip, receiver_port, response1)
            #send_data_to_receiver(receiver_ip, receiver_port, response2s)
             # Add a delay between measurements
    except KeyboardInterrupt:
        pass  # Allow the user to exit the loop with Ctrl+C
    # finally:
        # ser1.close()
        
        

if __name__ == "__main__":
    main()