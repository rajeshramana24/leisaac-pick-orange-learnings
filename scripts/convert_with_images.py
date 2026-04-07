"""Convert LeIsaac pick-orange dataset to robomimic HDF5 WITH image observations.
Reads parquet (actions, state) + MP4 videos (front camera).
Downsamples images to 84x84 for tractable BC training."""

import os, sys, h5py, numpy as np

# Video decoding
try:
    import av
    USE_AV = True
    print("  Using PyAV for video decoding", flush=True)
except ImportError:
    USE_AV = False
    try:
        import cv2
        print("  Using OpenCV for video decoding", flush=True)
    except ImportError:
        print("  ERROR: Neither PyAV nor OpenCV available!", flush=True)
        sys.exit(1)

import pyarrow.parquet as pq

DATASET_DIR = "/data/datasets/leisaac-pick-orange"
OUTPUT = "/data/mimicgen/seed_demos/pick_orange_with_images.hdf5"
IMG_SIZE = 84  # Downsample to 84x84 for BC training
NUM_EPISODES = 60  # all of them

print("=" * 60, flush=True)
print("  Converting LeIsaac dataset WITH images", flush=True)
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}", flush=True)
print(f"  Output: {OUTPUT}", flush=True)
print("=" * 60, flush=True)

def decode_video_av(path, img_size):
    """Decode video with PyAV, return (N, H, W, 3) uint8 array."""
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        # Resize
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((img_size, img_size), Image.BILINEAR))
        frames.append(img)
    container.close()
    return np.array(frames, dtype=np.uint8)

def decode_video_cv2(path, img_size):
    """Decode video with OpenCV, return (N, H, W, 3) uint8 array."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8)

decode_video = decode_video_av if USE_AV else decode_video_cv2

with h5py.File(OUTPUT, "w") as hf:
    data_grp = hf.create_group("data")
    total_frames = 0

    for ep_idx in range(NUM_EPISODES):
        # Read parquet
        pq_path = os.path.join(DATASET_DIR, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet")
        table = pq.read_table(pq_path)
        df = table.to_pandas()

        actions = np.stack(df["action"].values).astype(np.float32)
        states = np.stack(df["observation.state"].values).astype(np.float32)
        T = len(actions)

        # Read front camera video
        front_path = os.path.join(DATASET_DIR, "videos", "chunk-000",
                                   "observation.images.front", f"episode_{ep_idx:06d}.mp4")
        front_frames = decode_video(front_path, IMG_SIZE)

        # Read wrist camera video
        wrist_path = os.path.join(DATASET_DIR, "videos", "chunk-000",
                                   "observation.images.wrist", f"episode_{ep_idx:06d}.mp4")
        wrist_frames = decode_video(wrist_path, IMG_SIZE)

        # Align lengths (video may have +-1 frame vs parquet)
        min_len = min(T, len(front_frames), len(wrist_frames))
        actions = actions[:min_len]
        states = states[:min_len]
        front_frames = front_frames[:min_len]
        wrist_frames = wrist_frames[:min_len]

        # Write to HDF5
        demo = data_grp.create_group(f"demo_{ep_idx}")
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("rewards", data=np.zeros(min_len, dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros(min_len, dtype=bool))

        obs = demo.create_group("obs")
        obs.create_dataset("joint_pos", data=states)  # state = joint positions
        # Compute joint_vel from finite differences (state is positions only)
        vel = np.zeros_like(states)
        vel[1:] = (states[1:] - states[:-1]) * 30.0  # 30fps
        obs.create_dataset("joint_vel", data=vel)
        # Images stored as uint8 (H, W, C) — robomimic handles normalization
        obs.create_dataset("front_image", data=front_frames, compression="gzip", compression_opts=4)
        obs.create_dataset("wrist_image", data=wrist_frames, compression="gzip", compression_opts=4)

        total_frames += min_len
        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"  Episode {ep_idx}: {min_len} frames, "
                  f"grip=[{actions[:,5].min():.3f}, {actions[:,5].max():.3f}], "
                  f"img={front_frames.shape}", flush=True)

    # Mask for train split
    mask_grp = hf.create_group("mask")
    demo_keys = [f"demo_{i}" for i in range(NUM_EPISODES)]
    mask_grp.create_dataset("train", data=np.array(demo_keys, dtype="S"))

print(f"\n  Done! {NUM_EPISODES} episodes, {total_frames} total frames", flush=True)
print(f"  Output: {OUTPUT}", flush=True)
