def recursive_pipeline(steps, g, extra_param):
    if not steps:  # Base case: no steps left to apply
        return g

    # Get the current step and apply it
    current_step = steps[0]

    # Apply the current step, passing the generator and extra_param
    return current_step(g, lambda g: recursive_pipeline(steps[1:], g, extra_param), extra_param)


def rotate_step(g, next_step, extra_param):
    rotation_directions = ['x', 'y', 'z']  # Example rotation directions
    iterations = 2  # Example number of iterations for each rotation direction

    for rotation_axis in rotation_directions:
        for i in range(iterations):
            # Perform rotation logic for each axis and iteration
            print(f"Rotating by {rotation_axis}, iteration {i + 1} with extra_param: {extra_param}")
            # Yield results from the next step
            yield from next_step(g)  # Call the next step after each rotation iteration


def map_step(g, next_step, extra_param):
    for i in g:
        # Example map step transformation
        print(f"Mapping: {i} with extra_param: {extra_param}")
        # Yield the processed value and pass the generator to the next step
        yield from next_step([i])  # Pass individual element to next step


def third_step(g, next_step=None, extra_param=None):
    # Example third step transformation
    for i in g:
        print(f"Third step processing: {i} with extra_param: {extra_param}")
        # Yield the processed value
        yield f"Processed {i} with extra_param {extra_param}"


# Example input data
input_data = [1, 2, 3, 4]

# List of steps
steps = [rotate_step, map_step, third_step]

# Extra parameter (if needed)
extra_param = "extra data"

# Call the recursive function
result = recursive_pipeline(steps, input_data, extra_param)

# Collect the result from the pipeline
output = list(result)
print(output)
