"""
将当前文件夹中所有的 pptx 文件导出为对应的 pdf 格式文件
支持 Windows 系统（使用 PowerPoint COM 接口）
"""

import os
import sys
from pathlib import Path

try:
    import comtypes.client
except ImportError:
    print("错误: 需要安装 comtypes 库")
    print("请运行: pip install comtypes")
    sys.exit(1)


def pptx_to_pdf(pptx_path, pdf_path):
    """
    将单个 pptx 文件转换为 pdf
    
    Args:
        pptx_path: pptx 文件的完整路径
        pdf_path: 输出的 pdf 文件完整路径
    """
    try:
        # 创建 PowerPoint 应用程序对象
        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
        powerpoint.Visible = 1  # 设置为可见（可选，设为 0 则后台运行）
        
        # 打开 pptx 文件
        presentation = powerpoint.Presentations.Open(pptx_path)
        
        # 导出为 PDF
        # FormatType=32 表示 PDF 格式
        presentation.SaveAs(pdf_path, FileFormat=32)
        
        # 关闭演示文稿
        presentation.Close()
        
        # 退出 PowerPoint 应用程序
        powerpoint.Quit()
        
        return True
    except Exception as e:
        print(f"转换失败 {pptx_path}: {str(e)}")
        return False


def convert_all_pptx_to_pdf(directory=None):
    """
    将指定目录（或当前目录）中所有的 pptx 文件转换为 pdf
    
    Args:
        directory: 要处理的目录路径，默认为当前目录
    """
    if directory is None:
        directory = os.getcwd()
    
    directory = Path(directory)
    
    # 查找所有 pptx 文件
    pptx_files = list(directory.glob("*.pptx"))
    
    if not pptx_files:
        print(f"在 {directory} 中没有找到 pptx 文件")
        return
    
    print(f"找到 {len(pptx_files)} 个 pptx 文件")
    print("-" * 50)
    
    success_count = 0
    fail_count = 0
    
    for pptx_file in pptx_files:
        # 生成对应的 pdf 文件名
        pdf_file = pptx_file.with_suffix('.pdf')
        
        print(f"正在转换: {pptx_file.name} -> {pdf_file.name}")
        
        if pptx_to_pdf(str(pptx_file.absolute()), str(pdf_file.absolute())):
            print(f"✓ 成功: {pdf_file.name}")
            success_count += 1
        else:
            print(f"✗ 失败: {pptx_file.name}")
            fail_count += 1
        
        print("-" * 50)
    
    print(f"\n转换完成!")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")


if __name__ == "__main__":
    # 如果提供了命令行参数，使用该目录；否则使用当前目录
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    convert_all_pptx_to_pdf(target_dir)

