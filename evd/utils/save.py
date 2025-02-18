import pandas as pd

############################################################################################################
## Save the shape biases across time & categories
############################################################################################################
def save_shape_biases_across_time_csv(shape_biases, epoch, filename="shape_biases_across_timesteps.csv"):
    """
    Save a list of shape_biases (one per timestep) into a CSV file.
    
    Each row will have:
      - time_step: index in the list
      - shape_bias: the bias value for that timestep
    """
    # Create a DataFrame with an index representing the time step.
    df = pd.DataFrame({
        "time_step": list(range(len(shape_biases))),
        "shape_bias": shape_biases,
        'epoch': epoch,
    })
    df.to_csv(filename, index=False)
    print(f"Saved shape_biases across timesteps to {filename}")

def save_shape_biases_across_categories_csv(shape_bias_cat_dict, shape_correct_dict, texture_correct_dict, epoch, filename="combined_results.csv"):
    """
    Combine three dictionaries (each keyed by a timestep and then by category)
    into one CSV table.
    
    The resulting CSV will have columns:
      - time_step
      - category
      - shape_bias: from shape_bias_cat_dict
      - shape_correct: from shape_correct_dict
      - texture_correct: from texture_correct_dict
    """
    rows = []
    
    # Get all timesteps that exist in any of the dictionaries.
    all_timesteps = set(shape_bias_cat_dict.keys()) | set(shape_correct_dict.keys()) | set(texture_correct_dict.keys())
    
    for t in all_timesteps:
        # Get the set of categories for this timestep.
        categories = set()
        if t in shape_bias_cat_dict:
            categories.update(shape_bias_cat_dict[t].keys())
        if t in shape_correct_dict:
            categories.update(shape_correct_dict[t].keys())
        if t in texture_correct_dict:
            categories.update(texture_correct_dict[t].keys())
        
        for cat in categories:
            row = {
                "time_step": t,
                "category": cat,
                "shape_bias": shape_bias_cat_dict.get(t, {}).get(cat, None),
                "shape_correct": shape_correct_dict.get(t, {}).get(cat, None),
                "texture_correct": texture_correct_dict.get(t, {}).get(cat, None),
                'epoch': epoch,
            }
            rows.append(row)
    
    # Convert list of rows to a DataFrame and save as CSV.
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved combined category results to {filename}")