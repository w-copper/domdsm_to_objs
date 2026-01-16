# DOM+DSM → 分块三维网格（OBJ）生成工具

该脚本用于将**正射影像 DOM（RGB）**与**数字表面模型 DSM（高程）**融合，按瓦片（tile）分块生成三维网格（Poisson 重建），并导出为**OBJ 网格**（当前不输出贴图/MTL，仅保留几何）。坐标会从数据集 CRS 转到 WGS84，再以指定/自动选择的参考点建立 **ENU 局部坐标系**，以获得数值稳定且便于后续三维处理的局部坐标。

## 功能概览

- DOM/DSM 一致性检查（尺寸、仿射变换、CRS 必须一致）
- 按 `tile_size` + `overlap` 滑窗分块处理，适合大影像
- 像素中心点计算 → CRS 坐标 → WGS84 (lon/lat) → ECEF → ENU
- 生成带颜色的点云（来自 DOM RGB）
- 点云体素下采样（加速、降噪）
- Poisson 表面重建生成三角网格
- 网格清理（退化面、重复元素、非流形等）
- “智能简化”：
  - vertex clustering 自适应体素
  - QEM（二次）简化到目标密度
  - 可选误差评估（需要 Open3D RaycastingScene）
- overlap 区域**内缩裁剪**，减少瓦片边界重复/飞边
- 输出目录结构及 `metadata.xml`（记录 ENU 原点）

## 输入数据要求

- `DOM`：GeoTIFF 等栅格（至少 3 个波段 RGB），与 DSM 完全对齐
- `DSM`：单波段高程栅格（float/int 均可），与 DOM 完全对齐
- 二者必须满足：
  - `width/height` 相同
  - `transform` 相同
  - `crs` 相同

> 若不一致，请先进行重投影/配准/裁剪对齐再运行。

## 安装依赖

建议使用 Python 3.10+。

```bash
pip install numpy rasterio tqdm open3d pyproj pillow
```

说明：
- `open3d`：点云/Poisson/简化
- `rasterio`：读取地理栅格、window 分块
- `pyproj`：CRS → WGS84 转换
- `Pillow` 当前脚本中未实际使用到（可保留也可移除）

## 快速开始

修改脚本底部的路径后直接运行：

```python
if __name__ == "__main__":
    dom_path = r"D:/pan_align.tif"
    dsm_path = r"D:/dsm_align.tif"
    out_root = r"D:/out"

    process(
        dom_path, dsm_path, out_root,
        tile_size=2048,
        overlap=256,
        voxel_size=0.2,
        poisson_depth=10,
        normal_radius=1.0,
        density_quantile=0.02,
        lat0=None,
        lon0=None,
        nodata=10000,
        h0=0.0
    )
```

运行后输出类似：

```
out_root/
  metadata.xml
  Data/
    Tile_0_0/
      Tile_0_0.obj
    Tile_1792_0/
      Tile_1792_0.obj
    ...
```

## 参数说明（process）

```python
process(dom_path, dsm_path, out_root,
        tile_size=2048,
        overlap=256,
        voxel_size=0.2,
        poisson_depth=10,
        normal_radius=1.0,
        density_quantile=0.02,
        nodata=None,
        lat0=None, lon0=None, h0=0.0)
```

- `tile_size`：瓦片大小（像素）。越大单块更完整，但更慢、更吃内存。
- `overlap`：瓦片重叠（像素）。用于减少边缘缝隙；脚本会在导出前做“内缩裁剪”避免重复区域。
- `voxel_size`：点云体素下采样大小（ENU 米）。建议开启，否则 Poisson 会很慢。
- `poisson_depth`：Poisson 重建深度，常用 9~12。越大细节越多但更慢。
- `normal_radius`：法线估计半径（ENU 米）。通常设为点间距的 2~5 倍。
- `density_quantile`：删除 Poisson 低密度顶点的分位数阈值（0~1），用于去除边缘伪面（如 0.01~0.05）。
- `nodata`：DSM 的无效值（若不传则读 `dsm_ds.nodata`）。
- `lat0/lon0`：ENU 原点（WGS84）。`None` 时自动取影像中心点经纬度。
- `h0`：ENU 原点高程（米）。可设 0；或设为区域平均高程以减小 U 值幅度。

## 网格简化策略（simplify_mesh_smart）

脚本默认调用：

```python
mesh = simplify_mesh_smart(
    mesh,
    tris_per_m2=80.0,
    max_error=1.25,
    enable_clustering=True,
    clustering_voxel_min=0.05,
    clustering_voxel_max=1.5
)
```

- `tris_per_m2`：目标面密度（每平方米三角形数），越大越精细
- `max_error`：允许的几何偏差（ENU 米）
- 两阶段：
  1) 自适应 `vertex_clustering` 先压掉噪声小面
  2) QEM 简化到目标
- 若 Open3D 支持 `RaycastingScene`（0.16+），会通过采样点到原网格距离做误差控制，超限会回退（增加目标三角形数）

> 你当前示例 `max_error=1.25` 偏“宽松”。如果是城市场景想更贴合原表面，可尝试 0.1~0.3。

## 输出说明

- 每个 tile 输出一个 OBJ（仅几何）
- `metadata.xml` 写入 ENU 原点信息，格式：
  - `SRS`：`ENU:lat0,lon0`
  - `SRSOrigin`：`0,0,0`

当前脚本中存在 UV 相关函数（`add_planar_uv`）但**未在主流程启用**，也没有导出纹理图片与 MTL；`write_obj_no_texture()` 会确保 OBJ 内不包含 `mtllib/usemtl` 行。

## 性能与质量建议

- 首先调整 `voxel_size`：  
  - 地面分辨率 ~0.1m：可试 0.1~0.3  
  - 分辨率更粗：可试 0.3~1.0
- `poisson_depth` 不要盲目拉高：一般 10 足够；12 会明显变慢。
- `normal_radius` 太小会噪声大、破面；太大细节会被抹平。
- overlap 过小可能瓦片边界裂缝；过大则重复计算多。常见 128~512。

## 常见问题

### 1) 报错：DOM and DSM size/transform/CRS mismatch
必须先把 DOM 与 DSM 做到**完全同网格对齐**（同分辨率、同范围、同投影、同 transform）。可用 GDAL/rasterio 进行 warp/resample。

### 2) Poisson 很慢或内存占用高
优先增大 `voxel_size`（点更少），降低 `poisson_depth`，减少 `tile_size`。

### 3) 网格边缘有“飞面/糊边”
提高 `density_quantile`（例如 0.03~0.08），或适当增大 overlap 并保留裁剪。

## 许可与致谢

- 依赖：Open3D、Rasterio、PyProj 等开源库。
