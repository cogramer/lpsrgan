# Path to the file
file_path = "D:/personalProjects/Anaconda/envs/lpsrgan/lib/site-packages/basicsr/data/degradations.py"

# Open the file, read its contents, and replace the desired line
with open(file_path, 'r') as file:
    code = file.read()

# Replace the import line
modified_code = code.replace(
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms._functional_tensor import rgb_to_grayscale"
)

# Write the modified code back to the file
with open(file_path, 'w') as file:
    file.write(modified_code)

print("OK")