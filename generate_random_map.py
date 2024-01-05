import numpy as np
import os

def generate_random_map(height=26, width=26, num_walls=180, base_shape=("####", "#..#", "#..#")):
    # Create an empty grid with "." symbols
    grid = np.full((height, width), ".", dtype=str)

    # Set the top 2 rows to be empty
    grid[:2, :] = "."

    # Set the base at the bottom
    base_height = len(base_shape)
    base_width = len(base_shape[0])
    base_row = height - base_height
    for i, row in enumerate(base_shape):
        grid[base_row + i, (width - base_width) // 2 : (width + base_width) // 2] = list(row)

    # Number of walls already on the map (base walls)
    current_walls = (grid == "#").sum()

    # Randomly add walls until we reach the required number
    while current_walls < num_walls:
        # Random position
        x, y = np.random.randint(2, height - base_height), np.random.randint(0, width)
        # If it's empty and not part of the base, place a wall
        if grid[x, y] == ".":
            grid[x, y] = "#"
            current_walls += 1

    return grid

# Function to save a single map
def save_map(grid, filename):
    map_str = "\n".join("".join(row) for row in grid)
    with open(filename, "w") as file:
        file.write(map_str)

# Generate and save 100 maps
folder_path = "levels/SeriousTry1/"  # Update this path to your specific environment
os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

for i in range(1, 1001):
    random_map = generate_random_map()
    filename = f"{folder_path}{i}"
    save_map(random_map, filename)

print("1000 random maps generated and saved.")
