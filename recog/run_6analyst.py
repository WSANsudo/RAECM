#!/usr/bin/env python
"""
6Analyst 启动脚本
可以直接运行: python run_6analyst.py
"""

import sys
import os

# 修复 Windows 控制台编码问题
# 必须在导入其他模块之前设置
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 获取脚本的真实路径
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# 检查是否在6Analyst目录内，如果是则切换到父目录
if os.path.basename(script_dir) == '6Analyst' or os.path.exists(os.path.join(script_dir, '6Analyst')):
    # 如果脚本在6Analyst目录内，或者6Analyst是子目录，使用当前目录
    project_root = script_dir
else:
    project_root = script_dir

# 检查6Analyst包是否存在
if not os.path.exists(os.path.join(project_root, '6Analyst')):
    # 可能脚本在6Analyst目录内，需要切换到父目录
    parent_dir = os.path.dirname(project_root)
    if os.path.exists(os.path.join(parent_dir, '6Analyst')):
        project_root = parent_dir

os.chdir(project_root)
sys.path.insert(0, project_root)

# 导入并运行主函数
from importlib import import_module
run_module = import_module('6Analyst.run')
run_module.main()
