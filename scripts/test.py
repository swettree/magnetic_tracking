import usb.core
import usb.util

# 查找所有连接的 USB 设备
devices = usb.core.find(find_all=True)

# 打印每个设备的信息
for device in devices:
    print(f"Device: ID {hex(device.idVendor)}:{hex(device.idProduct)}")