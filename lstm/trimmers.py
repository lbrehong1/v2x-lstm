import re
import sys

NAN = "NaN"

def trim_pc5(input_file, output_file):
    out = 0
    pc5_data={}
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write header
        outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,latency_ms\n")

        for line in infile:
            if not line[0].isdigit():
                continue  # Ignore non-digit starting lines

            parts = line.strip().split(',')
            latency = parts[10].strip()
            if latency.startswith("0"):
                continue  # Skip malformed lines

            tx_seq_num = str(int(parts[15].strip()))
            tx_timestamp = str(float(parts[17].strip())/1000.0) # convert to ms
            tx_latitude = parts[20].strip()
            tx_longitude = parts[21].strip()

            pc5_data[int(float(tx_timestamp))//1000] = (tx_latitude, tx_longitude)

            outfile.write(f"{tx_seq_num},{tx_timestamp},{tx_latitude},{tx_longitude},{latency}\n")
            out += 1

    return out, pc5_data

def trim_dsrc(input_file, output_file, pc5_data=None):
    out = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write header
        if pc5_data is None:
            outfile.write("tx_seq_num,tx_timestamp_ms,rsrp_1,rsrp_2,latency_ms\n")
        else:
            outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,rsrp_1,rsrp_2,latency_ms\n")

        for line in infile:
            parts = re.split(r'\s+', line.strip())
            #if len(parts) != 17: #TODO uncomment after testing
            if len(parts) < 16 or len(parts) > 17:
                continue  # Skip malformed lines and headers
            #elif len(parts) == 16: #TODO uncomment after testing
            #    print ("This DSRC logfile does not contain latency values. Are you sure you enabled the -g time flag when running llc test-tx?")
            #    raise ValueError("Invalid log format, missing latency values. Output file will be broken.")

            tx_seq_num = str(int(parts[0])) # get rid of leading zeros
            timestamp = str(float(parts[9])*1000.0) # convert to ms
            power_ant1 = parts[5]
            power_ant2 = parts[6]
            latency = parts[10]

            if power_ant1.startswith("1"): # circumvent a bug where 16383.0 value is logged instead of -102.0 dBm
                power_ant1="-102.0"

            if pc5_data is None:
                outfile.write(f"{tx_seq_num},{timestamp},{power_ant1},{power_ant2},{latency}\n")

            else:
                ts = int(float(timestamp))//1000 # we transmitted every 100ms, we round up for every second
                if ts not in pc5_data: # TODO floor to nearest timestamp in ms
                    if ts+1 in pc5_data:
                        latitude, longitude = pc5_data.get(ts+1, ("", ""))
                    elif ts-1 in pc5_data:
                        latitude, longitude = pc5_data.get(ts-1, ("", ""))
                    else:
                        latitude, longitude = NAN, NAN
                else:
                    latitude, longitude = pc5_data.get(ts, ("", ""))
                outfile.write(f"{tx_seq_num},{timestamp},{latitude},{longitude},{power_ant1},{power_ant2},{latency}\n")

            out += 1

    return out




if __name__ == "__main__":
    if len(sys.argv) == 4:
        type = sys.argv[1]
        input_file = sys.argv[2]
        output_file = sys.argv[3]

        if type == "dsrc":
            out = trim_dsrc(input_file, output_file)
            print("Generated " + str(out) + " lines")
        elif type == "pc5":
            out = trim_pc5(input_file, output_file)
            print("Generated " + str(out) + " lines")
        else:
            print("Invalid log type, must be 'dsrc' or 'pc5'. Or you can use both files at once (recommended).")
            print("Usage 1 : python3 trimmers.py <rat_type> <input_file> <output_file>")
            print("Usage 2 : python3 trimmers.py <input_file_dsrc> <input_file_pc5> <output_file_dsrc> <output_file_pc5>")
            sys.exit(1)

    elif len(sys.argv) == 5:
        input_file_dsrc = sys.argv[1]
        input_file_pc5 = sys.argv[2]
        output_file_dsrc = sys.argv[3]
        output_file_pc5 = sys.argv[4]

        (out_pc5,pc5_data) = trim_pc5(input_file_pc5, output_file_pc5)
        out_dsrc = trim_dsrc(input_file_dsrc, output_file_dsrc, pc5_data)

        print("Generated " + str(out_pc5) + " lines for PC5")
        print("Generated " + str(out_dsrc) + " lines for DSRC")

    else:
        print("Usage 1 : python3 trimmers.py <rat_type> <input_file> <output_file>")
        print("Usage 2 : python3 trimmers.py <input_file_dsrc> <input_file_pc5> <output_file_dsrc> <output_file_pc5>")
        sys.exit(1)