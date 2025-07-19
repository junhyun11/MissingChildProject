import socket
import threading
import struct
import cv2
import numpy as np

camera_count = 1  # 카메라 수
ports = [6000 + i for i in range(camera_count)]
frames = [None] * camera_count
locks = [threading.Lock() for _ in range(camera_count)]

def handle_camera(idx, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    print(f"[{idx}] Listening on port {port}...")

    conn, addr = server.accept()
    print(f"[{idx}] Connected by {addr}")

    while True:
        try:
            # Receive frame size (4 bytes)
            length_bytes = conn.recv(4)
            if not length_bytes:
                print(f"[{idx}] Failed to receive length")
                break

            # Unpack frame size
            length = struct.unpack('<I', length_bytes)[0]
            buffer = b''
            while len(buffer) < length:
                packet = conn.recv(length - len(buffer))
                if not packet:
                    break
                buffer += packet

            print(f"[{idx}] Received bytes: {len(buffer)}")

            # Decode frame
            np_data = np.frombuffer(buffer, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[{idx}] Failed to decode frame")
                continue

            with locks[idx]:
                frames[idx] = frame

        except Exception as e:
            print(f"[{idx}] Exception occurred:", e)
            break

    conn.close()

# Start camera threads
for i in range(camera_count):
    threading.Thread(target=handle_camera, args=(i, ports[i]), daemon=True).start()

print("Waiting for camera connections...")

# Display loop using OpenCV
while True:
    for i in range(camera_count):
        with locks[i]:
            frame = frames[i]
        if frame is not None:
            cv2.imshow(f"Camera {i}", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()