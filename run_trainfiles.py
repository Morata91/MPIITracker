import subprocess

files_to_run = "train_v9.py"

for fold in range(12, 15):
    subprocess.run(["python", files_to_run, f'--fold={fold}', '--use=dol'])
