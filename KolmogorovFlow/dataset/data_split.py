import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Configuration paths
DATA_DIR = "./"  # Directory to store data
FILE_NAME = "train_omega_traj_40x800x128x128.npy"
os.makedirs(DATA_DIR, exist_ok=True)

def create_input_output_pairs(data_chunk):
    """
    Convert time series data into input-output pairs (u^n -> u^{n+1}) for TRAINING/TESTING.
    Args:
        data_chunk: shape (N_traj, T, H, W)
    Returns:
        input_tensor: shape (N_samples, 1, H, W)
        output_tensor: shape (N_samples, 1, H, W)
    """
    # Input is from 0 to T-1
    # Output is from 1 to T
    x = data_chunk[:, :-1, :, :]  # u^n
    y = data_chunk[:, 1:, :, :]   # u^{n+1}

    # Convert to Tensor
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # Flatten: (N_traj, T-1, H, W) -> (N_traj * (T-1), H, W)
    x = x.reshape(-1, x.shape[-2], x.shape[-1])
    y = y.reshape(-1, y.shape[-2], y.shape[-1])

    # Add Channel dimension -> (Samples, 1, H, W)
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)

    return x, y

def process_trajectory_tensor(data_chunk):
    """
    Process data into full trajectory tensors for VALIDATION.
    Args:
        data_chunk: shape (N_traj, T, H, W)
    Returns:
        traj_tensor: shape (N_traj, T, 1, H, W)
    """
    tensor = torch.from_numpy(data_chunk).float()
    # Add channel dimension at index 2
    # Result: [N, T, 1, H, W]
    # tensor = tensor.unsqueeze(2)
    return tensor

def shuffle_pairs(x, y):
    """
    Synchronously shuffle Input and Output pairs
    """
    print("Shuffling pairs...")
    num_samples = x.shape[0]
    perm = torch.randperm(num_samples)
    return x[perm], y[perm]

def generate_datasets():
    file_path = os.path.join(DATA_DIR, FILE_NAME)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        print("Please ensure the .npy file is in the data directory.")
        return

    print(f"Loading {FILE_NAME} ...")
    full_data = np.load(file_path)
    print(f"Original data shape: {full_data.shape}") # (74, 2001, 128, 128)

    # ==========================================
    # 1. Train Set (First 1000 time steps)
    #    Strategy: Flatten into pairs (u^n, u^n+1), Shuffle
    # ==========================================
    train_raw = full_data[:, :400, :, :]
    print(f"\nProcessing Train Data (Time 0-1000, Shape: {train_raw.shape})...")

    train_x, train_y = create_input_output_pairs(train_raw)
    train_x, train_y = shuffle_pairs(train_x, train_y)

    print(f"Saving Training Pairs: Input {train_x.shape}, Output {train_y.shape}")
    torch.save(train_x, os.path.join(DATA_DIR, "train_input_new.pt"))
    torch.save(train_y, os.path.join(DATA_DIR, "train_output_new.pt"))

    # ==========================================
    # 2. Val & Test Set (Last 1000 time steps)
    #    Strategy: Split trajectories first.
    #              Val -> Keep Trajectory structure
    #              Test -> Convert to Pairs & Shuffle
    # ==========================================
    test_val_raw = full_data[:, -400:, :, :] # (74, 1000, 128, 128)
    print(f"\nProcessing Val/Test Data (Time 1001-2000, Shape: {test_val_raw.shape})...")

    num_traj = test_val_raw.shape[0] # 74

    # Shuffle trajectory indices to randomly assign trajectories to Val or Test
    perm_indices = np.random.permutation(num_traj)
    split_idx = num_traj // 2

    val_indices = perm_indices[:split_idx]
    test_indices = perm_indices[split_idx:]

    val_raw = test_val_raw[val_indices]   # (37, 1000, 128, 128)
    test_raw = test_val_raw[test_indices] # (37, 1000, 128, 128)

    # --- Process Validation (Trajectory) ---
    val_tensor = process_trajectory_tensor(val_raw)
    print(f"Saving Validation Trajectories: {val_tensor.shape}") # [37, 1000, 1, 128, 128]
    torch.save(val_tensor, os.path.join(DATA_DIR, "test_trajectory_new.pt"))

    # --- Process Test (Pairs & Shuffle) ---
    print(f"Processing Test Data into Pairs...")
    test_x, test_y = create_input_output_pairs(test_raw)
    test_x, test_y = shuffle_pairs(test_x, test_y)

    print(f"Saving Test Pairs: Input {test_x.shape}, Output {test_y.shape}")
    torch.save(test_x, os.path.join(DATA_DIR, "test_input_new.pt"))
    torch.save(test_y, os.path.join(DATA_DIR, "test_output_new.pt"))

    print("\nAll files saved successfully.")

# --- Dataset Class Definition (For Train and Test) ---
class ODEPairDataset(Dataset):
    def __init__(self, input_path, output_path, limit=None):
        """
        Dataset for Single-Step Training/Testing (Pairs)
        """
        super().__init__()
        self.input = torch.load(input_path)
        self.output = torch.load(output_path)

        if limit is not None:
            self.input = self.input[:limit]
            self.output = self.output[:limit]

        assert len(self.input) == len(self.output), "Input and Output sizes do not match!"
        print(f"Loaded PairDataset with {len(self.input)} samples.")

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

if __name__ == "__main__":
    generate_datasets()

    # Check output existence and shapes
    if os.path.exists(os.path.join(DATA_DIR, "train_input_new.pt")):
        print("\n--- Verification ---")
        # Check Train
        ds_train = ODEPairDataset(
            os.path.join(DATA_DIR, "train_input_new.pt"),
            os.path.join(DATA_DIR, "train_output_new.pt"),
            limit=8000
        )
        print(f"Train Sample: {ds_train[0][0].shape}")

        # Check Test (Should be pairs now)
        ds_test = ODEPairDataset(
            os.path.join(DATA_DIR, "test_input_new.pt"),
            os.path.join(DATA_DIR, "test_output_new.pt"),
            limit=2000
        )
        print(f"Test Sample:  {ds_test[0][0].shape}")

        # Check Val (Should be trajectory)
        # val_data = torch.load(os.path.join(DATA_DIR, "val_trajectory.pt"))
        # print(f"Val Data:     {val_data.shape}")