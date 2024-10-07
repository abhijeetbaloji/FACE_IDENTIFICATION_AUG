import json
import numpy as np

def load_saved_landmarks(json_file="averaged_landmarks.json"):
    try:
        with open(json_file, "r") as file:
            data = json.load(file)

            # Check if the structure is a dictionary with "average_landmarks" key
            if isinstance(data, dict) and "average_landmarks" in data:
                return np.array(data["average_landmarks"])
            else:
                print("Error: JSON data does not contain 'average_landmarks' key.")
                return None
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file}'.")
        return None

# Example usage
saved_landmarks = load_saved_landmarks()
if saved_landmarks is not None:
    print("Landmarks loaded successfully.")
else:
    print("Failed to load landmarks.")
