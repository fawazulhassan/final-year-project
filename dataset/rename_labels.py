import os

# Paths to your label folders
train_labels_path = r"C:\Users\Fawaz\Desktop\yolov8-vehicle\dataset\labels\train"
val_labels_path = r"C:\Users\Fawaz\Desktop\yolov8-vehicle\dataset\labels\val"

# Function to rename files
def rename_files(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    print(f"Checking files in {folder_path}...")
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        print(f"Found file: {filename}")  # Debug: Print all files
        if filename.lower().endswith(".jpg.txt"):  # Updated to match your files
            print(f"Matched file: {filename}")  # Debug: Confirm matches
            new_filename = filename.replace(".jpg.txt", ".txt").replace(".JPG.txt", ".txt")  # Updated replacement
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)

            if os.path.exists(new_file):
                print(f"Warning: {new_filename} already exists. Skipping {filename}.")
                continue

            try:
                os.rename(old_file, new_file)
                print(f"Renamed: {filename} → {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
        else:
            print(f"Skipped file: {filename} (not a .jpg.txt file)")

# Rename files in train and val folders
print("Renaming files in train folder...")
rename_files(train_labels_path)
print("\nRenaming files in val folder...")
rename_files(val_labels_path)
print("Renaming complete!")