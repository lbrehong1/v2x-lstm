import time
import argparse

from flask import Flask, request, jsonify

app = Flask(__name__)

#LOGFILE_PATH = "logs_5g.txt"

@app.route('/', methods=['POST'])
def handle_request():
    data = request.get_json()

    if "query" in data and "lat" in data and "lon" in data:
        query = data["query"]
        lat = data["lat"]
        lon = data["lon"]

        if query == "test" and isinstance(lat, float) and isinstance(lon, float):
            response_data = {
                "query": "answer",
                "lat": lat,
                "lon": lon,
                "latency": 0.0,
                "pdr": 0.0
            }
            print(f"Hello World! Received test data: lat={lat}, lon={lon}")
            return jsonify(response_data)

        if query == "test_log" and isinstance(lat, float) and isinstance(lon, float):
            seqnum = data["seqnum"]
            timestamp = data["timestamp"]
            sinr = data["sinr"]
            rsrp = data["rsrp"]
            print(f"Received test data: seqnum={seqnum}, timestamp={timestamp}, lat={lat}, lon={lon}, sinr={sinr}, rsrp={rsrp}")

            response_data = {
                "query": "test_logged",
                "lat": lat,
                "lon": lon
            }
            return jsonify(response_data)

        if query == "log" and isinstance(lat, float) and isinstance(lon, float):
            seqnum = data["seqnum"]
            timestamp = data["timestamp"]
            sinr = data["sinr"]
            rsrp = data["rsrp"]
            latency = float(format(time.time(),".6f")) - float(timestamp)
            print(f"Logged data: seqnum={seqnum}, timestamp={timestamp}, lat={lat}, lon={lon}, latency={latency}, sinr={sinr}, rsrp={rsrp}")

            # Write log data to file
            with open(LOGFILE_PATH, "a") as logfile:
                logfile.write(f"{seqnum},{timestamp},{lat},{lon},{latency},{sinr},{rsrp}\n")

            response_data = {
                "query": "logged",
                "lat": lat,
                "lon": lon
            }
            return jsonify(response_data)

        else:
            response_data = {
                "query": "error"
            }
            print("whoops")
            return jsonify(response_data)

    else:
        return jsonify({"error": "Invalid request"}), 400


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HTTP message receiver")
    parser.add_argument('--logfile', type=str, required=True, help="Path to the log file")
    args = parser.parse_args()

    LOGFILE_PATH = args.logfile
    if not os.path.exists(LOGFILE_PATH):
        with open(LOGFILE_PATH, "w") as logfile:
            logfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,sinr,rsrp,latency_ms\n")

    app.run(port=8026)