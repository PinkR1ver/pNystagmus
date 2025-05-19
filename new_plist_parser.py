import plistlib
from datetime import datetime
import numpy as np

def parse_new_plist(file_obj_or_path, axis_data_key='LeftEyeXDegList'):
    """
    Parses a .plist file (from a file path or a file-like object)
    to extract eye movement data for a specified axis and timestamps.

    Args:
        file_obj_or_path (str or file-like object): The path to the .plist file or a file-like object.
        axis_data_key (str): The key for the eye angle data list (e.g., 'LeftEyeXDegList', 'LeftEyeYDegList').
                             Defaults to 'LeftEyeXDegList'.

    Returns:
        tuple: A tuple containing:
            - timestamps (np.array): Array of timestamps in seconds, relative to the first timestamp.
            - eye_angles (np.array): Array of eye angles from the specified axis_data_key.
            - available_deg_list_keys (list): List of available keys ending with 'DegList' in the plist.
    """
    if hasattr(file_obj_or_path, 'read'): # Check if it's a file-like object
        plist_data = plistlib.load(file_obj_or_path)
    else: # Assume it's a path
        with open(file_obj_or_path, 'rb') as fp:
            plist_data = plistlib.load(fp)

    available_deg_list_keys = [key for key in plist_data.keys() if 'DegList' in key]

    eye_angles = np.array(plist_data.get(axis_data_key, []))

    time_strings = plist_data.get('TimeList', [])
    if not time_strings:
        return np.array([]), eye_angles, available_deg_list_keys

    datetime_objects = []
    for ts in time_strings:
        try:
            datetime_objects.append(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f'))
        except ValueError:
            try:
                datetime_objects.append(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'))
            except ValueError as e:
                raise ValueError(f"Timestamp format error for '{ts}': {e}. Expected YYYY-MM-DD HH:MM:SS[.ffffff]")

    if not datetime_objects:
        return np.array([]), eye_angles, available_deg_list_keys

    start_time = datetime_objects[0]
    timestamps = np.array([(dt - start_time).total_seconds() for dt in datetime_objects])

    return timestamps, eye_angles, available_deg_list_keys

if __name__ == '__main__':
    sample_plist_content = {
        "LeftEyeXDegList": [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8],
        "LeftEyeYDegList": [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, -0.1],
        "AnotherDegList": [10, 20],
        "TimeList": [
            "2024-01-01 10:00:00.000",
            "2024-01-01 10:00:00.100",
            "2024-01-01 10:00:00.200",
            "2024-01-01 10:00:00.300",
            "2024-01-01 10:00:00.400",
            "2024-01-01 10:00:00.500",
            "2024-01-01 10:00:00.600"
        ]
    }
    test_file_path = 'test_data.plist'
    try:
        with open(test_file_path, 'wb') as f:
            plistlib.dump(sample_plist_content, f)
        
        print(f"Attempting to parse {test_file_path} for Horizontal (X-axis)...")
        timestamps_x, eye_angles_x, keys_x = parse_new_plist(test_file_path, axis_data_key='LeftEyeXDegList')
        print("X-Timestamps:", timestamps_x)
        print("X-Eye Angles:", eye_angles_x)
        print("Available DegList Keys:", keys_x)
        print(f"Successfully parsed X. Found {len(timestamps_x)} timestamps and {len(eye_angles_x)} angle readings.")

        print(f"\nAttempting to parse {test_file_path} for Vertical (Y-axis)...")
        timestamps_y, eye_angles_y, keys_y = parse_new_plist(test_file_path, axis_data_key='LeftEyeYDegList')
        print("Y-Timestamps:", timestamps_y)
        print("Y-Eye Angles:", eye_angles_y)
        print("Available DegList Keys:", keys_y)
        print(f"Successfully parsed Y. Found {len(timestamps_y)} timestamps and {len(eye_angles_y)} angle readings.")

        print(f"\nAttempting to parse {test_file_path} with a custom key...")
        timestamps_custom, eye_angles_custom, keys_custom = parse_new_plist(test_file_path, axis_data_key='AnotherDegList')
        print("Custom-Timestamps:", timestamps_custom) # Should be empty if TimeList is not tied to this custom key or if logic changes
        print("Custom-Eye Angles:", eye_angles_custom)
        print("Available DegList Keys:", keys_custom)
        print(f"Parsed custom key. Found {len(timestamps_custom)} timestamps and {len(eye_angles_custom)} angle readings.")

        print(f"\nAttempting to parse {test_file_path} with a missing key...")
        timestamps_missing, eye_angles_missing, keys_missing = parse_new_plist(test_file_path, axis_data_key='MissingEyeDegList')
        print("Missing-Timestamps:", timestamps_missing)
        print("Missing-Eye Angles:", eye_angles_missing)
        print("Available DegList Keys:", keys_missing)
        print(f"Parsed with missing key. Found {len(timestamps_missing)} timestamps and {len(eye_angles_missing)} angle readings.")


    except Exception as e:
        print(f"An error occurred during parsing test: {e}")
    finally:
        import os
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
    pass 