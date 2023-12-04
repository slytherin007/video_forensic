import cv2
import numpy as np
import matplotlib.pyplot as plt


video_path = "church.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Convert frames to grayscale and calculate hash for each frame
hashes = []
for frame in frames:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8))
    avg_pixel_value = np.mean(resized)
    hash_value = np.where(resized >= avg_pixel_value, 1, 0)
    hashes.append(hash_value)

# Compare hashes to detect frame duplication
duplicate_frames = set()
for i in range(len(hashes)):
    for j in range(i+1, len(hashes)):
        if np.array_equal(hashes[i], hashes[j]):
            duplicate_frames.add(i)
            duplicate_frames.add(j)

# Display duplicate frames
for frame_idx in duplicate_frames:
    plt.imshow(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB))
    plt.title(f"Duplicate Frame {frame_idx}")
    plt.axis('off')
    plt.show()
print("Duplicate Frames Indices:", duplicate_frames)