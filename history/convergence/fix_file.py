import os

folder_list = os.listdir(".")

for idx_folder, folder in enumerate(folder_list):

    if os.path.isdir(folder):

        file_list = os.listdir(f"./{folder}")

        for idx_file, file_old in enumerate(file_list):

            if file_old.startswith("F"):
                if ":" in file_old:
                    func_name, csv_name = file_old.split(":")
                    file_new = f"{func_name}_convergence.csv"
                    os.rename(f"./{folder}/{file_old}", f"./{folder}/{file_new}")
