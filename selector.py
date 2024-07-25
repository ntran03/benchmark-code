import os
import random
import shutil

#random seed
random.seed(42)


staple_dir = "/data/drdcad/nicole/outputs/staple"
class_list = [file for file in os.listdir(f"{staple_dir}") if file != "T2w"]



def random_select(class_name):
    
    output_dir = f"/data/drdcad/nicole/outputs/final/{class_name}"

    #make the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    #grab all the sequences from each class folder
    sequence_list = [file for file in os.listdir(f"{staple_dir}/{class_name}")]

    #initialize empty file list
    file_list = []

    #get all files
    for sequence in sequence_list:
        temp_list = [f"{sequence}.{file}" for file in os.listdir(f"{staple_dir}/{class_name}/{sequence}") if file.endswith(".nii")]

        #remove the files that ran with errors when stapling
        #temp_list = remove_errors(sequence, temp_list)

        #append to the file list for this class
        file_list.extend(temp_list)


    # Randomly select 10 segmentations without replacement
    print(class_name)
    random_selection = random.sample(file_list, k=10)
    print(random_selection)
    for file in random_selection:
        filename = file.split(".")[1]
        sequencename = file.split(".")[0]
        shutil.copy(f"/data/drdcad/nicole/outputs/staple/{class_name}/{sequencename}/{filename}.nii", f"{output_dir}/{sequencename}_{filename}.nii")


    

def remove_errors(sequence, files):
    error_file = "/data/drdcad/nicole/outputs/error_final.txt"
    print(sequence)
    errors = []
    clean_list = []

    with open(error_file, "r") as file:
        for line in file:
            if line.startswith(sequence):
                errors.append(line.split()[1])  # Remove newline character if needed

    for file in files:
        filename = file.split(".")[0]
        if filename in errors:
            print(f"Removed {file} from list")
        else:
            clean_list.append(file)


    return clean_list

def combine_error_files():

    file_names = ["error.txt", "errorB.txt", "errorKE.txt", "errorOCQ.txt", "errorQ.txt"]

    # Output file path
    output_file = "/data/drdcad/nicole/outputs/error_final.txt"

    # Function to read contents of a file
    def read_file_contents(file_path):
        with open(file_path, "r") as f:
            return f.read()

    # Combine contents of all files into a single string
    combined_contents = ""
    for file_name in file_names:
        file_path = f"/data/drdcad/nicole/benchmark/{file_name}"
        file_contents = read_file_contents(file_path)
        combined_contents += file_contents + "\n"  # Add a newline between file contents

    # Write combined contents to a new file
    with open(output_file, "w") as f:
        f.write(combined_contents)

    print(f"Combined contents written to {output_file}")



if __name__ == "__main__":
    #for class_name in class_list:
    #    random_select(class_name)
    random_select("T2w")