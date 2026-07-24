import argparse
import glob
import os
import numpy as np

# ===================== 默认路径，可通过命令行参数覆盖 =====================
# 1. 存放 .nc 文件的根文件夹（会自动遍历所有子文件夹）
DEFAULT_NC_FOLDER = "data/raw/Sen12Landslides/data_harmonized/s1asc"
# 2. 导出 TIF 的保存文件夹（自动创建）
DEFAULT_OUTPUT_FOLDER = "data/Sen12Landslides/tif/s1asc"
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Batch convert NetCDF files to GeoTIFF.")
    parser.add_argument("--nc-folder", default=DEFAULT_NC_FOLDER, help="Root folder containing .nc files.")
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER, help="Output folder for generated GeoTIFF files.")
    return parser.parse_args()

def get_nodata_for_dtype(dtype):
    """根据数据类型返回合适的 nodata 值"""
    if np.issubdtype(dtype, np.floating):
        return -9999.0
    elif np.issubdtype(dtype, np.integer):
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.iinfo(dtype).max  # 无符号整型用最大值
        else:
            return np.iinfo(dtype).min  # 有符号整型用最小值
    else:
        return None

def nc_to_tif(nc_file_path, output_folder):
    """
    单个 NC 文件转 TIF 函数
    :param nc_file_path: NC 文件路径
    :param output_folder: TIF 输出目录
    """
    try:
        import rasterio
        import xarray as xr
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError(
            "NetCDF conversion requires xarray and rasterio. "
            "Install them before running this dataset conversion script."
        ) from exc

    # 1. 读取 NC 文件
    try:
        ds = xr.open_dataset(nc_file_path)
        print(f"\n{'='*60}")
        print(f"✅ 成功读取文件：{os.path.basename(nc_file_path)}")
    except Exception as e:
        print(f"❌ 读取失败：{nc_file_path}，错误：{str(e)}")
        return

    # 2. 打印 NC 文件核心信息
    print(f"\n📊 文件维度：{list(ds.dims)}")
    print(f"📍 坐标变量：{list(ds.coords)}")
    print(f"🗂️  数据变量：{list(ds.data_vars)}")

    # 3. 获取地理坐标（适配Sen12Landslides数据：lat/lon 或 y/x）
    try:
        # 优先匹配标准经纬度
        lon = ds["lon"].values if "lon" in ds else ds["x"].values
        lat = ds["lat"].values if "lat" in ds else ds["y"].values
        # 计算栅格分辨率与地理起始点
        res_lon = abs(lon[1] - lon[0])
        res_lat = abs(lat[1] - lat[0])
        # 左上角坐标（栅格地理起点）
        left = lon.min()
        top = lat.max()
        # 栅格变换参数（WGS84 地理坐标系）
        transform = from_origin(left, top, res_lon, res_lat)
    except Exception as e:
        print(f"⚠️  坐标读取失败，使用默认坐标：{str(e)}")
        transform = None

    # 4. 遍历所有 数据变量（排除经纬度/时间等坐标），逐个导出TIF
    for var_name in ds.data_vars:
        # 跳过 spatial_ref 这类非影像变量
        if var_name in ["spatial_ref", "crs"]:
            print(f"\n⏭️  跳过变量：{var_name}")
            continue
            
        print(f"\n🔄 正在导出变量：{var_name}")
        
        # 提取变量数据（自动处理多维数据，取第一个维度切片）
        data = ds[var_name].values
        if data.ndim >= 3:
            data = data[0]  # 多波段/时间序列 → 取第一帧
        
        # 检查数据是否为二维
        if data.ndim != 2:
            print(f"⚠️  变量 {var_name} 不是二维数据，跳过")
            continue

        # 根据数据类型选择合适的 nodata 值
        nodata = get_nodata_for_dtype(data.dtype)
        
        # 去除无效值（NaN）
        if np.issubdtype(data.dtype, np.floating):
            data = np.nan_to_num(data, nan=nodata)

        # 生成输出文件名
        nc_name = os.path.splitext(os.path.basename(nc_file_path))[0]
        tif_name = f"{nc_name}_{var_name}.tif"
        tif_path = os.path.join(output_folder, tif_name)

        # 5. 用 rasterio 写入 GeoTIFF（带地理坐标！）
        try:
            with rasterio.open(
                tif_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs='EPSG:4326',  # WGS84 全球通用坐标系
                transform=transform,
                nodata=nodata
            ) as dst:
                dst.write(data, 1)

            print(f"✅ 导出完成：{tif_name}")
        except Exception as e:
            print(f"❌ 导出失败 {var_name}：{str(e)}")

    # 关闭文件
    ds.close()

def main():
    args = parse_args()
    nc_folder = os.path.expanduser(args.nc_folder)
    output_folder = os.path.expanduser(args.output_folder)

    # 自动创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    print("🚀 开始批量转换 NC → TIF ...")
    
    # 遍历所有 .nc 文件（包含子文件夹）
    nc_files = glob.glob(os.path.join(nc_folder, "**", "*.nc"), recursive=True)
    
    if len(nc_files) == 0:
        print("❌ 未找到任何 .nc 文件，请检查输入路径！")
    else:
        print(f"📂 共找到 {len(nc_files)} 个 NC 文件")
        
        # 批量处理
        for nc_file in nc_files:
            # 计算相对路径以保持文件夹结构
            rel_path = os.path.relpath(nc_file, nc_folder)
            rel_dir = os.path.dirname(rel_path)
            # 构建对应的输出目录
            current_output_folder = os.path.join(output_folder, rel_dir)
            os.makedirs(current_output_folder, exist_ok=True)
            
            nc_to_tif(nc_file, current_output_folder)

    print(f"\n🎉 所有文件处理完成！TIF 保存在：{output_folder}")


if __name__ == "__main__":
    main()
