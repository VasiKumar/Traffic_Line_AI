# =========================================
# 🚦 Smart Traffic Light Control System (Colab + Visual Output)
# =========================================
# Author: Vasi Kumar + GPT-5
# Works 100% in Google Colab (no IoT or hardware)
# Upload lane images, detect cars, visualize detections, and simulate control logic
# =========================================

# Step 1: Install YOLOv8 (Ultralytics)
!pip install ultralytics -q
from ultralytics import YOLO
import matplotlib.pyplot as plt
from google.colab import files
import cv2

# Step 2: Load YOLO model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # Small and fast

# Step 3: Upload images for each lane
print("📸 Upload one photo per lane (e.g., lane1.jpg, lane2.jpg, etc.)")
uploaded = files.upload()

# Step 4: Detect vehicles and count per lane
lane_counts = {}
lane_images = {}

print("\n🔍 Detecting vehicles in uploaded lane images...\n")
for filename in uploaded.keys():
    results = model(filename)
    detections = results[0].boxes
    count = 0
    for box in detections:
        cls = int(box.cls)
        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            count += 1

    lane_counts[filename] = count
    lane_images[filename] = results[0].plot()  # Save YOLO output image
    print(f"🚗 Lane '{filename}' → Vehicles detected: {count}")

    # Show detection image
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(lane_images[filename], cv2.COLOR_BGR2RGB))
    plt.title(f"{filename} - {count} vehicles detected")
    plt.axis('off')
    plt.show()

# Step 5: Logic-based traffic control simulation
# ----------------------------------------------
print("\n🚦 Starting traffic light simulation...")
base_green_time = 30  # seconds per lane
car_flow_per_cycle = 10  # cars that pass per green
cycle = 1

while any(count > 0 for count in lane_counts.values()):
    print(f"\n================ Cycle {cycle} ================")

    # 1️⃣ Select lane with most cars
    current_lane = max(lane_counts, key=lane_counts.get)
    current_count = lane_counts[current_lane]

    # Skip cleared lanes
    if current_count == 0:
        print(f"🚫 {current_lane} already clear. Skipping.")
        if all(count == 0 for count in lane_counts.values()):
            break
        cycle += 1
        continue

    # 🟢 Green Light Decision
    print(f"🟢 GREEN light → {current_lane}")
    print(f"⏱ Duration: {base_green_time} seconds")
    print(f"🚗 Cars before: {current_count}")

    # 2️⃣ Visualize current lane (again)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(lane_images[current_lane], cv2.COLOR_BGR2RGB))
    plt.title(f"🟢 GREEN Light - {current_lane}")
    plt.axis('off')
    plt.show()

    # 3️⃣ Simulate car passage
    cars_passed = min(car_flow_per_cycle, current_count)
    lane_counts[current_lane] -= cars_passed
    print(f"🚘 {cars_passed} cars passed in {base_green_time} seconds.")
    print(f"🚗 Cars remaining: {lane_counts[current_lane]}")

    # 4️⃣ Show current traffic state
    print("\n📊 Current Traffic State:")
    for lane, count in lane_counts.items():
        print(f"   {lane}: {count} cars waiting")

    # 5️⃣ Predict next lane
    next_lane = max(lane_counts, key=lane_counts.get)
    if lane_counts[next_lane] > 0:
        print(f"\n➡️ Next lane to be GREEN: {next_lane}")
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(lane_images[next_lane], cv2.COLOR_BGR2RGB))
        plt.title(f"Next Lane Prediction → {next_lane}")
        plt.axis('off')
        plt.show()
    else:
        print("\n✅ All lanes are clear! Traffic flow complete.")
        break

    cycle += 1

print("\n🎯 Simulation completed successfully!")
