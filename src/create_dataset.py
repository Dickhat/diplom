import os
import csv

def create_csv_from_txt(directory, output_csv):
    data = []
    
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                data.append([file.replace(".txt", ""), text])
    
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "text"])
        writer.writerows(data)
    
    print(f"CSV-файл успешно создан: {output_csv}")


create_csv_from_txt("F:/asr_public_phone_calls_1/0", "./dataset_target.csv")
