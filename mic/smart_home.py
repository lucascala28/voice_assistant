import tinytuya
import json
from pathlib import Path

class DeviceNotFoundError(Exception):
    pass

with open(Path(__file__).parent / "devices.json") as f:
    config = json.load(f)

plugs = config.get("plugs")
bulbs = config.get("bulbs")
tvs = config.get("tvs")

def control_device(dev_type, device, action):
    print("Control device tool invoked")
    if dev_type == "plugs":
        print("Plug device specified")
        deviceInfo = next((d for d in plugs if d["name"] == device), None)
        if deviceInfo is None:
            print(f"Device {device} not found in devices")
            raise(DeviceNotFoundError)
        print(f"Device {device} found!")
        
        dev = tinytuya.OutletDevice(dev_id=deviceInfo["dev_id"], address=deviceInfo["ip"], local_key=deviceInfo["local_key"], version=deviceInfo["version"])
        dev.set_socketTimeout(5)

        if action == "on":
            dev.turn_on()
            print(f"Turned on {device}\n")
        
        elif action == "off":
            dev.turn_off()
            print(f"Turned off {device}\n")

