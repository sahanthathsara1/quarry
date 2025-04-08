import trimesh

def get_model_dimensions_in_cm(file_path, original_unit="mm"):
    """
    Load a 3D model and return its dimensions in centimeters (cm).
    
    Parameters:
        file_path (str): Path to the 3D model file (.ply, .obj, .stl, etc.).
        original_unit (str): Original unit of the 3D model ("mm", "m", "inch", etc.).
                             Default is "mm" (millimeters).
    Returns:
        dimensions_cm (list): [length_x, length_y, length_z] in cm.
    """
    # Load the mesh
    mesh = trimesh.load(file_path)
    
    # Get dimensions in the original units
    dimensions = mesh.extents  # [length_x, length_y, length_z]
    
    # Convert to centimeters based on the original unit
    unit_conversion = {
        "mm": 0.1,      # 1 mm = 0.1 cm
        "m": 100,       # 1 m = 100 cm
        "inch": 2.54,   # 1 inch = 2.54 cm
        "ft": 30.48,    # 1 ft = 30.48 cm
    }
    
    if original_unit not in unit_conversion:
        raise ValueError(f"Unsupported unit: {original_unit}. Use 'mm', 'm', 'inch', or 'ft'.")
    
    dimensions_cm = dimensions * unit_conversion[original_unit]
    
    # Print results
    print(f"Model dimensions (original units): {dimensions} {original_unit}")
    print(f"Model dimensions (cm): {dimensions_cm}")
    print(f"Length in X-axis: {dimensions_cm[0]:.2f} cm")
    print(f"Length in Y-axis: {dimensions_cm[1]:.2f} cm")
    print(f"Length in Z-axis: {dimensions_cm[2]:.2f} cm")
    
    return dimensions_cm

# Example usage
model_path = r"C:\Users\Sahan\OneDrive\Desktop\workspace\wk3\pramuu.ply"
dimensions_cm = get_model_dimensions_in_cm(model_path, original_unit="mm")  # Change unit if needed