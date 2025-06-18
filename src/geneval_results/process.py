import os
from pathlib import Path
from PIL import Image

def convert_jpg_to_png(root_dir: str, remove_jpg: bool = True):
    """
    将指定根目录下所有子目录 samples/ 中的 .jpg 文件转换为 .png。

    :param root_dir: 顶层文件夹路径，例如 "/path/to/IMAGE_FOLDER"
    :param remove_jpg: 是否删除原始的 .jpg 文件，默认删除
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"{root_dir} 不是一个有效目录")

    for subdir in root.iterdir():
        samples_dir = subdir / "samples"
        if not samples_dir.is_dir():
            continue

        for jpg_path in samples_dir.glob("*.jpg"):
            try:
                img = Image.open(jpg_path)
                png_path = jpg_path.with_suffix(".png")
                img.save(png_path)
                print(f"已保存: {png_path}")
                if remove_jpg:
                    jpg_path.unlink()
                    print(f"已删除原图: {jpg_path}")
            except Exception as e:
                print(f"处理失败 {jpg_path}: {e}")

if __name__ == "__main__":
    # TODO: 将下面路径替换为你的 IMAGE_FOLDER 的实际路径
    root_dir = "janus_pro_think"
    convert_jpg_to_png(root_dir)
