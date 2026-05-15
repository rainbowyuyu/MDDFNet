"""
AI-TOD（官方 Google Drive「AI-TOD_wo_xview」）下载与 YOLO 格式转换。

说明：
- 完整 AI-TOD 另需 xView 训练集与 jwwangchn/AI-TOD 的 aitodtoolkit 合成脚本；
  本脚本针对官方发布的「不含 xView 图像」部分：COCO JSON + 对应 zip 图像。
- 使用项目虚拟环境：e:\\python_project\\MDDFNet\\.venv

用法：
  仅下载（可断点续传）：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py download

  仅转换（需已下载并解压 zip）：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py convert

  下载后自动解压并转换：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py all

  仅尝试下载四个图像 zip：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py download-images

  已有 labels、手动放好/解压图像后只拷图到 YOLO 目录：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py sync-images

「全要」（train/val/test/trainval 标注 + 四包图像）：建议浏览器下载 8 个文件，再 convert。
  打印全部直链与保存路径：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py print-urls
  检查本地是否已齐：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py check
  尝试 gdown 拉齐 4 个 JSON + 4 个 zip（常被配额拦截）：
    .venv\\Scripts\\python.exe dataset_process\\aitod_download_and_yolo.py download-all

标注 JSON（保存到 …/complete_annotations/）浏览器直链：
   - aitod_train.json:         id=1MuWMnh07oBHJcdVllGqW4sBsVeAf8nsm
   - aitod_val.json:           id=1-hFjjjAuc0weWhUMPwU7OkaGBKbXNG9N
   - aitod_test_v1_1.0.json:   id=14gT7wEfVR6xA3iAgFGEdzBcgXkTle_8z
   - aitod_trainval_v1_1.0.json: id=16avjvJGXvjzDjTBSM1nyXDvWUkt5ux9Z

图像 zip（保存到 …/images_wo_xview/）见下文；gdown 失败时用 print-urls 复制链接。

1) 浏览器直链下载 zip（登录 Google 后另存为）：
   - test:     https://drive.google.com/uc?export=download&id=1h_gOrkQCJTlSrHMdQbVcc7muY4CBmGxT
   - train:    https://drive.google.com/uc?export=download&id=1dQ8uS-kpvYISvBS3YYCjrBQ9S0BFa6pY
   - trainval: https://drive.google.com/uc?export=download&id=1LYwX4MkK3BTQdUxwRs-sB02S4zrZ90sn
   - val:      https://drive.google.com/uc?export=download&id=1yA6yUsgaC3p83TPTcxbMkvjobLlcVUwi

2) download-images / download-annotations / download-all（gdown -c）

3) 齐后执行 convert（解压 zip 并生成四 split 的 YOLO）；仅补图用 sync-images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 默认路径（可按需改参数覆盖）
DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1uNY_rcOO5LrWibXRY6l2dvqSbK6xikJp"
DEFAULT_DOWNLOAD_DIR = Path(r"e:\python_project\datasets\AI-TOD_wo_xview")
# YOLO 数据集输出与下载目录并列，位于 datasets 下
DEFAULT_YOLO_DATASET = Path(r"e:\python_project\datasets\AI-TOD_yolo")

# 官方 COCO 标注 JSON（complete_annotations/）
GDRIVE_ANNOTATION_JSON: Tuple[Tuple[str, str], ...] = (
    ("1MuWMnh07oBHJcdVllGqW4sBsVeAf8nsm", "aitod_train.json"),
    ("1-hFjjjAuc0weWhUMPwU7OkaGBKbXNG9N", "aitod_val.json"),
    ("14gT7wEfVR6xA3iAgFGEdzBcgXkTle_8z", "aitod_test_v1_1.0.json"),
    ("16avjvJGXvjzDjTBSM1nyXDvWUkt5ux9Z", "aitod_trainval_v1_1.0.json"),
)

# 官方 AI-TOD_wo_xview 图像包（images_wo_xview/）
GDRIVE_IMAGE_ZIPS: Tuple[Tuple[str, str], ...] = (
    ("1h_gOrkQCJTlSrHMdQbVcc7muY4CBmGxT", "aitod_wo_xview_test_imgs.zip"),
    ("1dQ8uS-kpvYISvBS3YYCjrBQ9S0BFa6pY", "aitod_wo_xview_train_imgs.zip"),
    ("1LYwX4MkK3BTQdUxwRs-sB02S4zrZ90sn", "aitod_wo_xview_trainval_imgs.zip"),
    ("1yA6yUsgaC3p83TPTcxbMkvjobLlcVUwi", "aitod_wo_xview_val_img.zip"),
)


def _venv_python() -> Path:
    return Path(r"e:\python_project\MDDFNet\.venv\Scripts\python.exe")


def _images_zip_dir(download_dir: Path) -> Path:
    root = _find_nested_root(download_dir)
    d = root / "images_wo_xview"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ann_dir(download_dir: Path) -> Path:
    root = _find_nested_root(download_dir)
    d = root / "complete_annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_annotation_jsons(download_dir: Path) -> int:
    """逐个下载四个 COCO JSON 到 complete_annotations（gdown -c）。"""
    py = _venv_python()
    if not py.is_file():
        print(f"未找到虚拟环境 Python: {py}", file=sys.stderr)
        return 1
    dest_dir = _ann_dir(download_dir)
    worst = 0
    for fid, name in GDRIVE_ANNOTATION_JSON:
        out_path = dest_dir / name
        url = f"https://drive.google.com/uc?id={fid}"
        cmd = [str(py), "-m", "gdown", url, "-O", str(out_path), "-c"]
        print("执行:", " ".join(cmd))
        code = subprocess.call(cmd)
        worst = max(worst, code)
    if worst != 0:
        print(
            "\n若 gdown 失败，请运行 print-urls 用浏览器下载 JSON 到:",
            dest_dir,
            file=sys.stderr,
        )
    return worst


def download_image_zips(download_dir: Path) -> int:
    """逐个下载四个图像 zip 到 images_wo_xview（gdown -c 续传）。"""
    py = _venv_python()
    if not py.is_file():
        print(f"未找到虚拟环境 Python: {py}", file=sys.stderr)
        return 1
    dest_dir = _images_zip_dir(download_dir)
    worst = 0
    for fid, name in GDRIVE_IMAGE_ZIPS:
        out_path = dest_dir / name
        url = f"https://drive.google.com/uc?id={fid}"
        cmd = [str(py), "-m", "gdown", url, "-O", str(out_path), "-c"]
        print("执行:", " ".join(cmd))
        code = subprocess.call(cmd)
        worst = max(worst, code)
    if worst != 0:
        print(
            "\n若 gdown 失败，请用浏览器打开上文 docstring 中的 uc?export=download 链接下载 zip，"
            f"保存到: {dest_dir}",
            file=sys.stderr,
        )
    return worst


def download_all_remote_parts(download_dir: Path) -> int:
    """gdown 下载 4 个标注 JSON + 4 个图像 zip（「全要」自动化尝试）。"""
    return max(download_annotation_jsons(download_dir), download_image_zips(download_dir))


def print_manual_download_urls(download_dir: Path, yolo_dir: Path) -> None:
    root = _find_nested_root(download_dir)
    ann = root / "complete_annotations"
    img = root / "images_wo_xview"
    print("=== AI-TOD_wo_xview「全要」手动下载 ===\n")
    print("[1] 标注 JSON -> 目录（需自行创建）:\n   ", ann, "\n")
    for fid, name in GDRIVE_ANNOTATION_JSON:
        print(name)
        print(f"  https://drive.google.com/uc?export=download&id={fid}\n")
    print("[2] 图像 zip -> 目录:\n   ", img, "\n")
    for fid, name in GDRIVE_IMAGE_ZIPS:
        print(name)
        print(f"  https://drive.google.com/uc?export=download&id={fid}\n")
    print("[3] 全部就位后生成 YOLO（含四 split + 拷图）:")
    py = _venv_python()
    print(
        f'  "{py}" "{Path(__file__).resolve()}" convert '
        f'--raw-dir "{download_dir}" --yolo-dir "{yolo_dir}"'
    )


def check_local_dataset(download_dir: Path, yolo_dir: Path) -> int:
    """检查 8 个远程文件是否在本地；可选统计 YOLO labels/images。"""
    root = _find_nested_root(download_dir)
    ann = root / "complete_annotations"
    imgd = root / "images_wo_xview"
    print("=== 本地完整性（全要 = 4 JSON + 4 zip）===\n")
    print("数据根:", root, "\n")
    missing = 0
    print("[标注 JSON]", ann)
    for _, name in GDRIVE_ANNOTATION_JSON:
        p = ann / name
        ok = p.is_file() and p.stat().st_size > 0
        print(f"  {'OK ' if ok else '缺失'} {name}")
        if not ok:
            missing += 1
    print("\n[图像 zip]", imgd)
    for _, name in GDRIVE_IMAGE_ZIPS:
        p = imgd / name
        ok = p.is_file() and p.stat().st_size > 0
        mb = round(p.stat().st_size / (1024 * 1024), 1) if ok else 0
        print(f"  {'OK ' if ok else '缺失'} {name}" + (f"  ({mb} MB)" if ok else ""))
        if not ok:
            missing += 1

    # 解压目录内是否已有 png（粗略）
    png_n = sum(1 for _ in root.rglob("*.png"))
    print(f"\n[解压后] 树下共 {png_n} 个 .png（含子目录）")

    yolo_dir = Path(yolo_dir)
    if yolo_dir.is_dir():
        print("\n[YOLO]", yolo_dir)
        for split in ("train", "val", "test", "trainval"):
            li = yolo_dir / "labels" / split
            ii = yolo_dir / "images" / split
            nl = sum(1 for _ in li.glob("*.txt")) if li.is_dir() else 0
            ni = sum(1 for _ in ii.glob("*.*")) if ii.is_dir() else 0
            if nl or ni:
                print(f"  {split}: labels={nl}, images={ni}")
    if missing:
        print(f"\n尚有 {missing} 项未就绪。请运行: print-urls")
        return 1
    print("\n8 项文件已齐，可执行 convert（将解压 zip 并写入 YOLO）。")
    return 0


def download_aitod_wo_xview(out_dir: Path, drive_url: str) -> int:
    """调用 gdown 下载整个 Drive 文件夹，支持 -c 续传。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    py = _venv_python()
    if not py.is_file():
        print(f"未找到虚拟环境 Python: {py}", file=sys.stderr)
        return 1
    cmd = [
        str(py),
        "-m",
        "gdown",
        "--folder",
        drive_url,
        "-O",
        str(out_dir) + os.sep,
        "-c",
    ]
    print("执行:", " ".join(cmd))
    return subprocess.call(cmd)


def _find_nested_root(download_dir: Path) -> Path:
    """gdown 常在子目录再套一层 AI-TOD_wo_xview。"""
    cand = download_dir / "AI-TOD_wo_xview"
    if cand.is_dir():
        return cand
    return download_dir


def unzip_all_zips(root: Path) -> None:
    for zpath in root.rglob("*.zip"):
        target = zpath.parent / zpath.stem
        if target.is_dir() and any(target.iterdir()):
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"解压: {zpath} -> {target}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(target)


def _find_image_for_stem(stem: str, index: Dict[str, Path]) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"):
        p = index.get((stem + ext).lower())
        if p is not None:
            return p
    return None


def sync_images_to_yolo(raw_dir: Path, yolo_dir: Path) -> int:
    """在 raw 目录下解压 zip 并建立索引，按已有 YOLO labels/*.txt 将图像拷入 images/。"""
    raw_root = _find_nested_root(raw_dir)
    if not raw_root.is_dir():
        print(f"raw 目录不存在: {raw_dir}", file=sys.stderr)
        return 1
    unzip_all_zips(raw_root)
    print("索引图像…")
    img_index = _index_images(raw_root)
    yolo_dir = Path(yolo_dir)
    labels_root = yolo_dir / "labels"
    if not labels_root.is_dir():
        print(f"未找到 YOLO labels 目录: {labels_root}", file=sys.stderr)
        return 1
    copied = 0
    missing = 0
    skipped = 0
    for split_dir in sorted(labels_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        out_img_dir = yolo_dir / "images" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        for lbl in split_dir.glob("*.txt"):
            stem = lbl.stem
            src = _find_image_for_stem(stem, img_index)
            if src is None:
                missing += 1
                continue
            dst = out_img_dir / src.name
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
    print(f"sync-images: 复制 {copied} 张，已存在跳过 {skipped}，未找到源文件 {missing} 条标签对应的图")
    return 0 if missing == 0 else 2


def _index_images(root: Path) -> Dict[str, Path]:
    """basename(lower) -> 第一个匹配到的路径。"""
    index: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.PNG", "*.JPG"):
        for p in root.rglob(ext):
            key = p.name.lower()
            index.setdefault(key, p)
    return index


def _split_from_json_name(name: str) -> str:
    n = name.lower()
    if "trainval" in n:
        return "trainval"
    if "train" in n:
        return "train"
    if "val" in n:
        return "val"
    if "test" in n:
        return "test"
    return "split"


def _build_category_map(categories: List[dict]) -> Tuple[Dict[int, int], List[str]]:
    """COCO category_id -> YOLO 0..nc-1，names 与 id 升序一致。"""
    cats = sorted(categories, key=lambda c: int(c["id"]))
    id_to_idx = {int(c["id"]): i for i, c in enumerate(cats)}
    names = [str(c["name"]) for c in cats]
    return id_to_idx, names


def _bbox_coco_to_yolo_line(
    bbox: List[float], img_w: int, img_h: int, cls_idx: int
) -> Optional[str]:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"


def convert_coco_dir_to_yolo(
    raw_root: Path,
    yolo_root: Path,
    copy_images: bool = True,
) -> None:
    raw_root = _find_nested_root(raw_root)
    ann_dir = raw_root / "complete_annotations"
    if not ann_dir.is_dir():
        ann_dir = raw_root
    json_files = sorted(ann_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"未找到 COCO 标注 JSON: {ann_dir}")

    print("索引图像文件（可能较慢）…")
    img_index = _index_images(raw_root)

    yolo_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "trainval"):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    global_names: Optional[List[str]] = None

    for jf in json_files:
        split = _split_from_json_name(jf.name)
        print(f"处理标注: {jf.name} -> split={split}")
        with open(jf, "r", encoding="utf-8") as f:
            coco = json.load(f)
        id_to_idx, names = _build_category_map(coco["categories"])
        if global_names is None:
            global_names = names
        elif global_names != names:
            print("警告: 各类别表与首个 JSON 不一致，仍以首个为准。", file=sys.stderr)

        anns_by_img: Dict[int, List[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            anns_by_img[int(ann["image_id"])].append(ann)

        missing_img = 0
        for im in coco["images"]:
            iid = int(im["id"])
            file_name = im["file_name"]
            w = int(im["width"])
            h = int(im["height"])
            stem = Path(file_name).name.lower()
            src = img_index.get(stem)
            dst_img = yolo_root / "images" / split / Path(file_name).name
            dst_lbl = yolo_root / "labels" / split / (Path(file_name).stem + ".txt")

            if copy_images and src is not None:
                shutil.copy2(src, dst_img)
            elif copy_images:
                missing_img += 1

            lines: List[str] = []
            for ann in anns_by_img.get(iid, []):
                cid = int(ann["category_id"])
                cls_idx = id_to_idx.get(cid)
                if cls_idx is None:
                    continue
                line = _bbox_coco_to_yolo_line(ann["bbox"], w, h, cls_idx)
                if line:
                    lines.append(line)
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.writelines(lines)

        if missing_img:
            print(
                f"  警告: {split} 有 {missing_img} 张图在磁盘上未找到源文件，"
                f"已写入标签；请确认已解压 images zip 或重新运行 download。"
            )

    if global_names:
        yaml_path = yolo_root / "dataset.yaml"
        root_posix = yolo_root.as_posix()
        present_splits = []
        for split in ("train", "val", "test", "trainval"):
            lbl_dir = yolo_root / "labels" / split
            if lbl_dir.is_dir() and any(lbl_dir.glob("*.txt")):
                present_splits.append(split)
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write(f"path: {root_posix}\n")
            for split in present_splits:
                yf.write(f"{split}: images/{split}\n")
            yf.write(f"nc: {len(global_names)}\n")
            yf.write("names:\n")
            for i, n in enumerate(global_names):
                yf.write(f"  {i}: {n}\n")
        print(f"已写入 {yaml_path}（splits: {', '.join(present_splits)}）")


def cmd_download(args: argparse.Namespace) -> int:
    return download_aitod_wo_xview(Path(args.download_dir), args.drive_url)


def cmd_download_images(args: argparse.Namespace) -> int:
    return download_image_zips(Path(args.download_dir))


def cmd_download_annotations(args: argparse.Namespace) -> int:
    return download_annotation_jsons(Path(args.download_dir))


def cmd_download_all(args: argparse.Namespace) -> int:
    return download_all_remote_parts(Path(args.download_dir))


def cmd_print_urls(args: argparse.Namespace) -> int:
    print_manual_download_urls(Path(args.download_dir), Path(args.yolo_dir))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    return check_local_dataset(Path(args.download_dir), Path(args.yolo_dir))


def cmd_sync_images(args: argparse.Namespace) -> int:
    return sync_images_to_yolo(Path(args.raw_dir), Path(args.yolo_dir))


def cmd_convert(args: argparse.Namespace) -> int:
    raw = Path(args.raw_dir)
    if not raw.is_dir():
        print(f"raw 目录不存在: {raw}", file=sys.stderr)
        return 1
    unzip_all_zips(_find_nested_root(raw))
    convert_coco_dir_to_yolo(raw, Path(args.yolo_dir), copy_images=not args.labels_only)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    code = cmd_download(args)
    if code != 0:
        print("下载未完全成功，仍尝试解压与转换已有文件…", file=sys.stderr)
    return cmd_convert(args)


def main() -> int:
    p = argparse.ArgumentParser(description="AI-TOD_wo_xview 下载与 YOLO 转换")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("download", help="gdown 下载 Drive 文件夹（-c 续传）")
    pd.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    pd.add_argument("--drive-url", type=str, default=DEFAULT_DRIVE_FOLDER)
    pd.set_defaults(func=cmd_download)

    pc = sub.add_parser("convert", help="解压 zip 并将 COCO JSON 转为 YOLO")
    pc.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="gdown 输出根目录（内含 AI-TOD_wo_xview 子目录亦可）",
    )
    pc.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DATASET))
    pc.add_argument(
        "--labels-only",
        action="store_true",
        help="只写 labels，不复制图像（源图缺失时可用）",
    )
    pc.set_defaults(func=cmd_convert)

    pa = sub.add_parser("all", help="download + convert")
    pa.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    pa.add_argument("--drive-url", type=str, default=DEFAULT_DRIVE_FOLDER)
    pa.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DATASET))
    pa.add_argument("--labels-only", action="store_true")
    pa.set_defaults(func=cmd_all)

    pz = sub.add_parser(
        "download-images",
        help="仅下载四个图像 zip 到 …/AI-TOD_wo_xview/images_wo_xview（gdown -c）",
    )
    pz.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    pz.set_defaults(func=cmd_download_images)

    ps = sub.add_parser(
        "sync-images",
        help="解压 raw 下 zip 后，按 YOLO labels 将图像拷入 yolo-dir/images（适合已 labels-only）",
    )
    ps.add_argument("--raw-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    ps.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DATASET))
    ps.set_defaults(func=cmd_sync_images)

    pa_json = sub.add_parser(
        "download-annotations",
        help="仅下载四个 COCO JSON 到 …/complete_annotations（gdown -c）",
    )
    pa_json.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    pa_json.set_defaults(func=cmd_download_annotations)

    pall = sub.add_parser(
        "download-all",
        help="gdown 下载 4 个 JSON + 4 个 zip（全要；常被 Drive 配额拦截）",
    )
    pall.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    pall.set_defaults(func=cmd_download_all)

    pchk = sub.add_parser(
        "check",
        help="检查 4 JSON + 4 zip 是否已下载，并列出 YOLO 各 split 数量",
    )
    pchk.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    pchk.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DATASET))
    pchk.set_defaults(func=cmd_check)

    purl = sub.add_parser(
        "print-urls",
        help="打印「全要」8 个文件的浏览器直链与保存目录、以及推荐 convert 命令",
    )
    purl.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    purl.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DATASET))
    purl.set_defaults(func=cmd_print_urls)

    args = p.parse_args()
    # all 需要 raw_dir 与 download_dir 一致
    if args.cmd == "all":
        args.raw_dir = args.download_dir
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
