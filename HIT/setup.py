import PyInstaller.__main__
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

PyInstaller.__main__.run([
    'main.py',
    '--name=HIT_Analyzer',
    '--onefile',
    '--windowed',
    f'--add-data={os.path.join(current_dir, "utils.py")};.',
    f'--add-data={os.path.join(current_dir, "data")};data',
    '--icon=NONE',  # 如果您有图标文件，可以在这里指定
    '--clean',
    '--noconfirm'
])