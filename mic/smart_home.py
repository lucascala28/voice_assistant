import tinytuya
from kasa import SmartPlug
import json
from pathlib import Path

class DeviceNotFoundError(Exception):
    pass

with open(Path(__file__).parent / "devices.json") as f:
    config = json.load(f)

DEVICE_INDEX: dict[str, dict] = {}
for category, devices in config.items():
    for device in devices:
        key = device["name"]
        DEVICE_INDEX[key] = {**device, "category": category}

def resolve_device(name):
    key = name.lower()
    if key in DEVICE_INDEX:
        print(f"{DEVICE_INDEX[key]['category']} device {name} found!")
        return DEVICE_INDEX[key]
    else:
        print(f"Device {name} not found amongst known devices")
        raise(DeviceNotFoundError)

async def control_device(device, action):
    print("Control device tool invoked")

    deviceInfo = resolve_device(device)

    if deviceInfo["category"] == "plugs":

        if deviceInfo["type"] == "tuya":
            print("Using tuya protocol for control...")
            plug = tinytuya.OutletDevice(dev_id=deviceInfo["dev_id"], address=deviceInfo["ip"], local_key=deviceInfo["local_key"], version=deviceInfo["version"])
            plug.set_socketTimeout(5)

            if action == "on":
                plug.turn_on()
                print(f"Turned on {device}\n")
            
            elif action == "off":
                plug.turn_off()
                print(f"Turned off {device}\n")
        
        elif deviceInfo["type"] == "kasa":
            print("Using kasa protocol for control...")
            plug = SmartPlug(deviceInfo["ip"])

            if action == "on":
                await plug.turn_on()
                print(f"Turned on {device}\n")
            
            elif action == "off":
                await plug.turn_off()
                print(f"Turned off {device}\n")




