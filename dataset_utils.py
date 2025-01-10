import os, shutil
import pandas as pd
import numpy as np


######################## PRE-PROCESSING ########################

def get_tumor_mask(segmentation_mask):
    tumor_mask = (segmentation_mask > 0).astype(int)
    segmentation_mask
    return tumor_mask


def get_size(tumor):
    return np.sum(tumor)


def get_coordinates(tumor_mask):
    tumor_idx = np.where(tumor_mask == 1)
    
    start_x, start_y = np.min(tumor_idx[0]), np.min(tumor_idx[1])
    end_x, end_y = np.max(tumor_idx[0]), np.max(tumor_idx[1])

    mid_point_x = int((start_x + end_x)/2)
    mid_point_y = int((start_y + end_y)/2)
    
    return mid_point_x, mid_point_y


def get_position_label(tumor_mask):
    mid_point_x, mid_point_y = get_coordinates(tumor_mask)
    bound = 512 / 2
    center_th = 5

    # Tumors on the bottom / top side of the brain
    if mid_point_x > bound + center_th and mid_point_y > bound + center_th:
        label = 'bottom right'

    elif mid_point_x > bound + center_th and mid_point_y < bound - center_th :
        label = 'bottom left'
            
    elif mid_point_x < bound - center_th and mid_point_y > bound + center_th:
        label = 'top right'
        
    elif mid_point_x < bound - center_th and mid_point_y < bound - center_th:
        label = 'top left'
            
    # Tumors on the center of the brain (center left/ right)
    else:
        if mid_point_y > bound + center_th:
            label = 'center right'
        elif mid_point_y < bound - center_th:
            label = 'center left'
        else:
            label = 'center'
            
    return label

# Function to copy images to the target directory
def copy_images(file_names, target_dir, image_folder):
    for file_name in file_names:
        src_path = os.path.join(image_folder, file_name)
        dst_path = os.path.join(target_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist.")

######################## POST-PROCESSING ########################

def text2binary(metadata_path):

    # Read the Excel file into a DataFrame
    df = pd.read_csv(metadata_path)

    # Replace text in the specified column
    df['text'] = df['text'].apply(
        lambda x: 0 if isinstance(x, str) and 'healthy' in x.lower() else 1
    )
    # Display the data
    # print(matadata)
    df.to_csv(metadata_path, index=False)
    
# Combine metadata.csv files from both folders
def combine_metadata(folder1, folder2, output_file):
    metadata1 = os.path.join(folder1, "metadata.csv")
    metadata2 = os.path.join(folder2, "metadata.csv")

    # Load metadata from both folders
    df1 = pd.read_csv(metadata1) if os.path.exists(metadata1) else pd.DataFrame()
    df2 = pd.read_csv(metadata2) if os.path.exists(metadata2) else pd.DataFrame()
    if 'text' in df1.columns:
        df1['text'] = df1['text'].fillna(0).astype(int)  # Handle NaNs and ensure integers
    if 'text' in df2.columns:
        df2['text'] = df2['text'].fillna(0).astype(int)  # Handle NaNs and ensure integers

    # Combine both metadata files
    combined_metadata = pd.concat([df1, df2], ignore_index=True)

    # Save combined metadata to the destination folder
    combined_metadata.to_csv(output_file, index=False)
    print(f"Combined metadata saved at: {output_file}")