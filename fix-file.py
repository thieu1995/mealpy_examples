#!/usr/bin/env python
# Created by "Thieu" at 16:20, 16/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

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
