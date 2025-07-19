import socket
import threading
import struct
import cv2
import numpy as np
import os
import time

camera_count = 1 #카메라 수
ports = [6000 + i for i in range(camera_count)]
frames = [None] * camera_count
locks = [threading.Lock() for _ in range(camera_count)]

base_save_dir = r"C:\camera_output"
camera_dirs = []
for i in range(camera_count):
    cam_dir = os.path.join(base_save_dir, f"camera_{i}")
    os.makedirs(cam_dir, exist_ok=True)
    camera_dirs.append(cam_dir)

save_interval = 2 # seconds
last_save_time = [0.0] * camera_count
saved_count = [0] * camera_count
max_per_camera = 30
frame_received = [False] * camera_count
connect_time = [None] * camera_count  # 연결 시간 저장

def handle_camera(idx, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    print(f"[{idx}] Listening on port {port}...")

    conn, addr = server.accept()
    print(f"[{idx}] Connected by {addr}")
    connect_time[idx] = time.time()  # 연결 시간 기록

    while True:
        try:
            length_bytes = conn.recv(4)
            if not length_bytes:
                print(f"[{idx}] Failed to receive length")
                break

            length = struct.unpack('<I', length_bytes)[0]
            buffer = b''
            while len(buffer) < length:
                packet = conn.recv(length - len(buffer))
                if not packet:
                    break
                buffer += packet

            print(f"[{idx}] Received bytes: {len(buffer)}")

            np_data = np.frombuffer(buffer, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[{idx}] Failed to decode frame")
                continue

            with locks[idx]:
                frames[idx] = frame
                frame_received[idx] = True

        except Exception as e:
            print(f"[{idx}] Exception occurred:", e)
            break

    conn.close()

# Start camera threads
for i in range(camera_count):
    threading.Thread(target=handle_camera, args=(i, ports[i]), daemon=True).start()

print("Waiting for camera connections...")

# Display and save loop
while True:
    current_time = time.time()

    for i in range(camera_count):
        with locks[i]:
            frame = frames[i]

        if frame is not None:
            cv2.imshow(f"Camera {i}", frame)

            # 몇 초 이상 경과한 후에만 저장 시작 (시간 변경)
            if (
                saved_count[i] < max_per_camera
                and connect_time[i] is not None
                and current_time - connect_time[i] > 10
                and current_time - last_save_time[i] > save_interval
            ):
                filename = os.path.join(camera_dirs[i], f"{int(current_time)}.jpg")
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"[{i}] Saved: {filename}")
                    saved_count[i] += 1
                    last_save_time[i] = current_time
                else:
                    print(f"[{i}] Failed to save: {filename}")

    if cv2.waitKey(1) == 27:
        print("ESC pressed. Exiting...")
        break

    if all(frame_received) and all(saved_count[i] >= max_per_camera for i in range(camera_count)):
        print("All cameras saved 30 images each. Exiting...")
        break

cv2.destroyAllWindows()
