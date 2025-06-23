#!/bin/bash

# 1. 安装所有 Python 依赖
echo "--- Installing Python dependencies... ---"
pip install -r requirements.txt
pip install tqdm

# 2. 运行脚本来分析并打印依赖库的大小
echo "--- Analyzing package sizes... ---"
python -c "import os, pkg_resources; from tqdm import tqdm; pkgs = sorted([(p.project_name, p.location) for p in pkg_resources.working_set], key=lambda x: x[0]); print('--- Package Sizes ---'); total = 0; for name, loc in tqdm(pkgs): size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(loc) for f in fs); print(f'{name:<30} {size / 1024 / 1024:>10.2f} MB'); total += size; print(f'--- Total Size: {total / 1024 / 1024:.2f} MB ---')"

# 3. 创建一个假的 index.html 以满足 Vercel 的构建输出要求
echo "--- Build script finished. Check logs for package sizes. ---"
echo "Build finished" > index.html
