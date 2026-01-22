import nbformat
from nbconvert import PythonExporter


def convert_ipynb_to_py(ipynb_path, py_path):
    # 读取.ipynb文件
    with open(ipynb_path,encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 创建PythonExporter对象
    exporter = PythonExporter()

    # 转换并获取输出
    (body, resources) = exporter.from_notebook_node(nb)

    # 写入.py文件
    with open(py_path, 'w') as f:
        f.write(body)


# 使用函数进行转换
convert_ipynb_to_py('..//DNN//temp.ipynb', '..//DNN//temp.py')