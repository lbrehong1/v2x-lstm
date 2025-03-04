import requests
import json
import argparse
import time
import serial
import re
import threading

URL = "lstm.univ-tlse3.fr"
PORT = 8026
LAT = 43.554669
LON = 1.463952
COMMAND = 'AT#MONI'
COMMAND_GPS = 'AT$GPSACP'
NAN = "NaN"

#%%

def send_at_command(command):
    ser.write((command + '\r\n').encode())
    time.sleep(.01)
    response = ser.read_all().decode('utf-8')
    print("AT response:", response)
    return response

def initialize_gps():
    print("GPSNMUNEX: ", send_at_command('AT$GPSNMUNEX=1,1,1'))
    print("GPSNMUN: ", send_at_command('AT$GPSNMUN=2,1,1,1,1,1,1'))
    print("AT$GPSP=1: ", send_at_command('AT$GPSP=1'))

def isolate_lat_lon(response):
    # Extract latitude and longitude using regex
    match = re.search(r"\$GPSACP: \d+\.\d+,(?P<lat>\d+\.\d+)(?P<lat_dir>[NS]),(?P<lon>\d+\.\d+)(?P<lon_dir>[EW])", response)
    if match:
        latitude = nmea_to_decimal(match.group("lat"), match.group("lat_dir"))
        longitude = nmea_to_decimal(match.group("lon"), match.group("lon_dir"))
        print(f"Latitude: {latitude}, Longitude: {longitude}")
        return latitude, longitude
    else:
        print("AT$GPSACP: Could not find GPS coordinates")
        return NAN, NAN

def nmea_to_decimal(coord, direction):
    """Convert NMEA coordinate to decimal degrees."""
    match = re.match(r"(\d+)(\d{2}\.\d+)", coord)
    if match:
        degrees = int(match.group(1))
        minutes = float(match.group(2))
        decimal = degrees + (minutes / 60)
        if direction in ['S', 'W']:  # South and West are negative
            decimal *= -1
        return round(decimal, 6)
    return None

def isolate_rsrp_sinr(response):
    # Regex pattern to extract NR_RSRP and NR_SINR
    match = re.search(r"NR_RSRP:(-?\d+)\s+NR_SINR:(-?\d+)", response)
    if match:
        nr_rsrp = int(match.group(1))
        nr_sinr = int(match.group(2))
        print(f"NR_RSRP: {nr_rsrp}, NR_SINR: {nr_sinr}")
    else:
        print("NR_RSRP or NR_SINR not found")
        nr_rsrp = NAN
        nr_sinr = NAN
    return str(nr_rsrp), str(nr_sinr)

def generate_query(query, lat, lon):
    return {
        "query": query,
        "lat": lat,
        "lon": lon
    }

def generate_message(query, num, timestamp, lat, lon, sinr, rsrp):
    return {
        "query": query,
        "seqnum": num,
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon,
        "sinr": sinr,
        "rsrp": rsrp
    }

def send_post(url, port, query_data):
    full_url = f"https://{url}:{port}/"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(full_url, data=json.dumps(query_data), headers=headers)
    return response.json()

def send_post_async(url, port, query_data):
    thread = threading.Thread(target=send_post, args=(url, port, query_data))
    thread.start()

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5G SA message generator")
    parser.add_argument('--type', type=str, required=True, help="test_post, test_log or log")
    parser.add_argument('--tx_interval', type=int, required=True, help="TX Interval in ms (only for type log)")
    parser.add_argument('--com_port', type=str, required=True, help="COM port for the modem (e.g., COM3)")
    args = parser.parse_args()

    TYPE = args.type
    TX_INTERVAL = args.tx_interval
    COM = args.com_port

    ser = serial.Serial(
        port=COM_PORT,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )

    if TYPE == "test_post":
        query = "test"
        query_data = generate_query(query, LAT, LON)
        response = send_post(URL, PORT, query_data)
        print(response)

    elif TYPE == "test_log":
        line = send_at_command(COMMAND)
        nr_rsrp, nr_sinr = isolate_rsrp_sinr(line)

        # Initialize GPS and read once
        print("Initializing GPS, wait for 5 seconds...")
        initialize_gps()
        time.sleep(5) # Wait for GPS to initialize
        gps_to_parse = send_at_command(COMMAND_GPS)
        lat, lon = isolate_lat_lon(gps_to_parse)

        query = "test_log"
        timestamp = format(time.time(), '.3f')
        gps_lat = lat
        gps_lon = lon
        sinr = nr_sinr
        rsrp = nr_rsrp
        #log_data = generate_message(query, 0, timestamp, LAT, LON, 0, 0) # THIS IS FOR TESTING PURPOSES
        log_data = generate_message("log", 999, timestamp, gps_lat, gps_lon, sinr, rsrp) #
        response = send_post(URL, PORT, log_data)
        print(response)

    elif TYPE == "log":
        # Initialize GPS and find initial coordinates
        print("Initializing GPS, wait for 5 seconds...")
        initialize_gps()
        time.sleep(5) # Wait for GPS to initialize
        lat = NAN
        while lat == NAN:
            gps_to_parse = send_at_command(COMMAND_GPS)
            lat, lon = isolate_lat_lon(gps_to_parse)
            if lat != NAN:
                print("GPS data found, starting transmission...")
            else:
                print("GPS data not available, retrying in 1 second...")
                time.sleep(1)

        k=0
        timer_at = float(time.time())
        timer_send = float(time.time())
        line = send_at_command(COMMAND)
        nr_rsrp, nr_sinr = isolate_rsrp_sinr(line)
        while True:
            if (float(time.time())-timer_send) > TX_INTERVAL/1000: # Send data every tx_interval
                # Grab radio stats
                line = send_at_command(COMMAND)
                nr_rsrp, nr_sinr = isolate_rsrp_sinr(line)
                # Grab GPS
                gps_to_parse = send_at_command(COMMAND_GPS)
                new_lat, new_lon = isolate_lat_lon(gps_to_parse)
                if new_lat != NAN:
                    lat = new_lat
                    lon = new_lon
                else:
                    print("GPS data not available, keeping previous values")

                # Populate and send data
                query = "log"
                timestamp = format(time.time(), '.3f')
                gps_lat = lat
                gps_lon = lon
                sinr = nr_sinr
                rsrp = nr_rsrp
                #log_data = generate_message(query, k, timestamp, LAT, LON, 0, 0) # THIS IS FOR TESTING PURPOSES
                log_data = generate_message(query, k, timestamp, gps_lat, gps_lon, sinr, rsrp) #
                send_post_async(URL, PORT, log_data)
                k+=1
                timer_send = time.time()

            time.sleep(.001) # sleep in seconds

    else:
        raise ValueError("Invalid type. Must be 'test_post', 'test_log' or 'log'.")