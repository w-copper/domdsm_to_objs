import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import open3d as o3d
from pyproj import CRS, Transformer
from PIL import Image


def _mesh_area(mesh: o3d.geometry.TriangleMesh) -> float:
    # Open3D 有 get_surface_area()（新版本），这里做个兼容兜底
    if hasattr(mesh, "get_surface_area"):
        return float(mesh.get_surface_area())
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    if len(t) == 0:
        return 0.0
    a = v[t[:, 0]]
    b = v[t[:, 1]]
    c = v[t[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _adaptive_cluster_voxel(mesh: o3d.geometry.TriangleMesh,
                            target_tris: int,
                            vmin: float,
                            vmax: float,
                            iters: int = 10) -> float:
    """
    用二分搜索一个 voxel_size，让 vertex_clustering 后的三角形数接近 target_tris。
    返回选出来的 voxel_size（可能达到边界）。
    """
    tri0 = len(mesh.triangles)
    if tri0 == 0 or target_tris <= 0:
        return vmin

    lo, hi = vmin, vmax
    best_v = lo
    best_gap = float("inf")

    for _ in range(iters):
        mid = (lo + hi) * 0.5
        m = mesh.simplify_vertex_clustering(
            voxel_size=float(mid),
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        t = len(m.triangles)
        gap = abs(t - target_tris)
        if gap < best_gap:
            best_gap = gap
            best_v = mid

        # voxel 越大 -> 越粗 -> 三角形越少
        if t > target_tris:
            lo = mid
        else:
            hi = mid

    return float(best_v)


def simplify_mesh_smart(
    mesh: o3d.geometry.TriangleMesh,
    # 你可以二选一：给 target_triangles 或者给 tris_per_m2（面密度）
    target_triangles: int | None = None,
    tris_per_m2: float | None = 50.0,

    # 误差控制：简化后表面到原表面的允许偏差（单位=ENU米）
    max_error: float = 0.20,

    # 两阶段：先 clustering 把“噪声小三角”压掉，再 QEM 精细控制
    enable_clustering: bool = True,
    clustering_share: float = 0.55,   # clustering 先降到目标的 1/0.55 倍左右，再 QEM 到目标
    clustering_voxel_min: float = 0.05,
    clustering_voxel_max: float = 2.00,

    preserve_boundary: bool = False,  # Open3D QEM 不一定真正保边；此参数主要给你语义位
    qem_aggressiveness: int = 2,       # QEM 重试次数（防止误差超限）
) -> o3d.geometry.TriangleMesh:
    """
    更智能的简化：
    1) 计算 mesh 面积 -> 根据 tris_per_m2 得到目标三角形数（或使用 target_triangles）
    2) clustering 自适应找 voxel_size 先做一次“结构化降采样”
    3) QEM 到目标
    4) 用 sample_points + 点到三角形距离估计误差，超限就回退/放宽简化
    """
    if mesh is None:
        return None

    mesh = _cleanup_mesh(mesh)
    tri0 = len(mesh.triangles)
    if tri0 == 0:
        return mesh

    area = _mesh_area(mesh)
    if target_triangles is None:
        if tris_per_m2 is None:
            raise ValueError("target_triangles 和 tris_per_m2 不能同时为 None")
        # 面积太小会导致目标三角形数过小，做个下限
        target_triangles = int(max(2000, area * float(tris_per_m2)))

    target_triangles = int(target_triangles)
    target_triangles = max(2000, target_triangles)  # 防止过度简化导致破碎
    if tri0 <= target_triangles:
        return mesh

    # --------------------------
    # Stage A: vertex clustering (可选)
    # --------------------------
    work = mesh
    if enable_clustering and tri0 > target_triangles * 1.5:
        # clustering 目标：先把三角形压到一个“略高于最终目标”的数量，留给 QEM 精修
        pre_target = int(target_triangles / max(1e-6, clustering_share))
        pre_target = max(target_triangles, min(pre_target, tri0))

        voxel = _adaptive_cluster_voxel(
            work, target_tris=pre_target,
            vmin=clustering_voxel_min, vmax=clustering_voxel_max,
            iters=10
        )
        clustered = work.simplify_vertex_clustering(
            voxel_size=float(voxel),
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        work = _cleanup_mesh(clustered)

    # --------------------------
    # Stage B: QEM decimation to target
    # --------------------------
    def decimate(m, tgt):
        m2 = m.simplify_quadric_decimation(target_number_of_triangles=int(tgt))
        return _cleanup_mesh(m2)

    # 误差评估：从简化后采样点，测到原 mesh 的距离（更关注“简化后是否偏离原表面”）
    # 注意：需要 Open3D 支持 RaycastingScene（0.16+）
    def estimate_error(orig, simp, n_samples=20000) -> float:
        if not hasattr(o3d.t.geometry, "RaycastingScene"):
            return 0.0  # 老版本无法评估，直接跳过误差控制
        # 采样点数按面数自适应
        n = int(max(5000, min(n_samples, len(simp.triangles) * 20)))
        pcd = simp.sample_points_uniformly(number_of_points=n)
        pts = np.asarray(pcd.points).astype(np.float32)

        scene = o3d.t.geometry.RaycastingScene()
        orig_t = o3d.t.geometry.TriangleMesh.from_legacy(orig)
        _ = scene.add_triangles(orig_t)
        d = scene.compute_distance(o3d.core.Tensor(pts))
        # 用较稳健的分位数做误差指标（避免少量离群点）
        dist = d.numpy()
        return float(np.quantile(dist, 0.95))  # 95% 点的误差

    # 第一次到目标
    simplified = decimate(work, target_triangles)

    # 如果支持误差评估，则进行“误差超限回退”
    # 策略：误差太大 -> 不要简太狠（把目标三角形数调高一些）再试几次
    if max_error is not None and max_error > 0:
        err = estimate_error(mesh, simplified)
        if err > max_error:
            # 回退：逐步提高目标三角形数（更保守）
            tgt = target_triangles
            for _ in range(int(max(1, qem_aggressiveness))):
                tgt = int(min(tri0, tgt * 1.5))
                candidate = decimate(work, tgt)
                err2 = estimate_error(mesh, candidate)
                if err2 <= max_error:
                    simplified = candidate
                    break
            # 如果怎么都超限，就返回误差最小的那个（这里保持 simplified）

    return simplified

# ---------------------------
# Tiling
# ---------------------------
def iter_tiles(width, height, tile_size, overlap):
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("tile_size must be > overlap")
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            w = min(tile_size, width - x0)
            h = min(tile_size, height - y0)
            yield x0, y0, w, h

def window_pixel_centers_to_map_xy(transform, x0, y0, w, h):
    """
    X,Y of pixel centers in dataset CRS
    """
    cols = np.arange(x0, x0 + w)
    rows = np.arange(y0, y0 + h)
    cc, rr = np.meshgrid(cols, rows)

    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    X = a * cc + b * rr + c + a * 0.5 + b * 0.5
    Y = d * cc + e * rr + f + d * 0.5 + e * 0.5
    return X, Y

# ---------------------------
# ENU conversion (WGS84)
# ---------------------------
# WGS84 constants
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)

def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = _A / np.sqrt(1 - _E2 * sin_lat * sin_lat)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - _E2) + h) * sin_lat
    return x, y, z

def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, h0):
    """
    Convert ECEF to ENU at reference (lat0,lon0,h0)
    """
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)

    # Rotation matrix ECEF->ENU
    t = np.array([
        [-sin_lon0,            cos_lon0,           0.0],
        [-sin_lat0*cos_lon0,  -sin_lat0*sin_lon0,  cos_lat0],
        [ cos_lat0*cos_lon0,   cos_lat0*sin_lon0,  sin_lat0]
    ], dtype=np.float64)

    # vectorized apply
    # [e;n;u] = t * [dx;dy;dz]
    e = t[0,0]*dx + t[0,1]*dy + t[0,2]*dz
    n = t[1,0]*dx + t[1,1]*dy + t[1,2]*dz
    u = t[2,0]*dx + t[2,1]*dy + t[2,2]*dz
    return e, n, u

def build_transformers(dataset_crs):
    """
    Build transformer dataset CRS -> WGS84 lon/lat
    rasterio/pyproj axis order pitfalls:
    - We request always_xy=True so inputs are (x,y) and outputs are (lon,lat)
    """
    src = CRS.from_user_input(dataset_crs)
    dst = CRS.from_epsg(4326)
    to_wgs84 = Transformer.from_crs(src, dst, always_xy=True)
    return to_wgs84

# ---------------------------
# Point cloud + Poisson
# ---------------------------
def build_point_cloud_from_tile(dom, dsm, X_map, Y_map,
                                to_wgs84,
                                lat0, lon0, h0,
                                dsm_nodata=None):
    """
    dom: (bands,h,w) uint8/uint16 (RGB or RGBA)
    dsm: (h,w) float32
    X_map, Y_map: (h,w) map coordinates in dataset CRS
    """
    # DOM to (h,w,3)
    if dom.ndim != 3:
        raise ValueError("DOM read must be (bands,h,w)")
    bands, h, w = dom.shape
    if bands < 3:
        raise ValueError("DOM must have at least 3 bands (RGB)")
    rgb = np.transpose(dom[:3, :, :], (1, 2, 0))

    Z = dsm
    valid = np.isfinite(Z)
    if dsm_nodata is not None:
        valid &= (Z != dsm_nodata)

    if valid.sum() < 1000:
        return None

    # Convert map XY -> lon/lat
    lon, lat = to_wgs84.transform(X_map[valid], Y_map[valid])

    # Geodetic -> ECEF -> ENU
    x, y, z = geodetic_to_ecef(lat, lon, Z[valid])
    e, n, u = ecef_to_enu(x, y, z, lat0, lon0, h0)

    pts = np.stack([e, n, u], axis=1).astype(np.float64)

    # colors
    if rgb.dtype == np.uint8:
        cols = rgb[valid].astype(np.float32) / 255.0
    else:
        cols = rgb[valid].astype(np.float32)
        denom = cols.max() if cols.max() > 0 else 1.0
        cols = np.clip(cols / denom, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd
def poisson_mesh(pcd, depth=10, normal_radius=1.0, normal_max_nn=30,
                 density_quantile=0.02,
                 orient_mode="towards_up"):
    """
    更稳的 Poisson 重建：不使用 orient_normals_consistent_tangent_plane（会触发Qhull错误）
    orient_mode:
      - "towards_up": 以点云上方视点统一定向（地表/城市强烈推荐）
      - "towards_centroid": 朝向点云中心（也可）
      - "none": 不做定向（可能导致Poisson反面/破碎）
    """
    # 去掉非法/重复点，降低退化概率
    pcd = pcd.remove_non_finite_points()
    pcd = pcd.remove_duplicated_points()

    if len(pcd.points) < 2000:
        return None

    # 法线估计
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
        )
    )

    # 法线定向（避免 Qhull）
    if orient_mode != "none":
        bb = pcd.get_axis_aligned_bounding_box()
        center = bb.get_center()
        extent = bb.get_extent()

        if orient_mode == "towards_up":
            # ENU 坐标里 U=Z 轴，取上方一个点当“相机位置”
            up = float(max(extent[2], 1.0))
            view = center + np.array([0.0, 0.0, 5.0 * up], dtype=np.float64)
            pcd.orient_normals_towards_camera_location(view)
        elif orient_mode == "towards_centroid":
            pcd.orient_normals_towards_camera_location(center)
        else:
            raise ValueError(f"Unknown orient_mode: {orient_mode}")

    # Poisson 重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    mesh.compute_vertex_normals()

    # 删除低密度顶点（减少边缘“糊边/飞面”）
    dens = np.asarray(densities)
    if dens.size > 0 and 0.0 < density_quantile < 1.0:
        thr = np.quantile(dens, density_quantile)
        mesh.remove_vertices_by_mask(dens < thr)

    # 清理一下
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh

# ---------------------------
# UV + OBJ export with texture
# ---------------------------
def add_planar_uv(mesh, texture_w, texture_h, bbox_min_e, bbox_min_n, bbox_max_e, bbox_max_n):
    """
    UV mapping: u = (E - Emin)/(Emax-Emin), v = 1 - (N - Nmin)/(Nmax-Nmin)
    (v翻转让图片坐标与OpenGL/常见纹理一致)
    """
    verts = np.asarray(mesh.vertices)
    E = verts[:, 0]
    N = verts[:, 1]
    denom_e = (bbox_max_e - bbox_min_e) if (bbox_max_e - bbox_min_e) != 0 else 1.0
    denom_n = (bbox_max_n - bbox_min_n) if (bbox_max_n - bbox_min_n) != 0 else 1.0
    u = (E - bbox_min_e) / denom_e
    v = 1.0 - (N - bbox_min_n) / denom_n
    uv = np.stack([u, v], axis=1).astype(np.float64)
    uv = np.clip(uv, 0.0, 1.0)
    # Open3D 需要 per-triangle-vertex UV：长度 = 3 * num_triangles
    tris = np.asarray(mesh.triangles)
    tri_uv = uv[tris.reshape(-1)]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uv)
    return mesh
def write_obj_no_texture(mesh, out_obj_path):
    out_dir = os.path.dirname(out_obj_path)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 写 OBJ（不写UV也行，避免贴图链路）
    o3d.io.write_triangle_mesh(out_obj_path, mesh, write_triangle_uvs=False)

    # 2) 删除/过滤掉 OBJ 里可能存在的 mtllib/usemtl 行
    with open(out_obj_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    lines = [
        l for l in lines
        if not l.lower().startswith("mtllib")
        and not l.lower().startswith("usemtl")
    ]

    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
def compute_inner_crop_box_enu(transform, x0, y0, w, h,
                               overlap_px,
                               to_wgs84, lat0, lon0, h0):
    """
    返回内裁剪框: (emin, nmin, emax, nmax) in ENU
    裁剪量 = 0.5 * overlap_px （像素）
    """
    # tile 外框四角像素中心的 map xy
    # 左上、右上、右下、左下（用像素中心）
    corners_px = np.array([
        [x0,       y0      ],
        [x0 + w-1, y0      ],
        [x0 + w-1, y0 + h-1],
        [x0,       y0 + h-1],
    ], dtype=np.float64)

    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    cc = corners_px[:, 0]
    rr = corners_px[:, 1]
    X = a * cc + b * rr + c + a * 0.5 + b * 0.5
    Y = d * cc + e * rr + f + d * 0.5 + e * 0.5

    lon, lat = to_wgs84.transform(X, Y)
    x, y, z = geodetic_to_ecef(lat, lon, np.full(4, h0, dtype=np.float64))
    E, N, U = ecef_to_enu(x, y, z, lat0, lon0, h0)

    emin, emax = float(E.min()), float(E.max())
    nmin, nmax = float(N.min()), float(N.max())

    # ---- 把 overlap_px/2 换算成 ENU 米 ----
    shrink_px = 0.3 * float(overlap_px)

    # 用 tile 中心附近的“像素步长”估计：E/N 每个像素的尺度
    cx = x0 + w * 0.5
    cy = y0 + h * 0.5

    # 中心与右移1像素的点
    Xc, Yc = window_pixel_centers_to_map_xy(transform, int(cx), int(cy), 1, 1)
    Xr, Yr = window_pixel_centers_to_map_xy(transform, int(cx)+1, int(cy), 1, 1)
    Xu, Yu = window_pixel_centers_to_map_xy(transform, int(cx), int(cy)+1, 1, 1)

    lonc, latc = to_wgs84.transform(float(Xc[0,0]), float(Yc[0,0]))
    lonr, latr = to_wgs84.transform(float(Xr[0,0]), float(Yr[0,0]))
    lonu, latu = to_wgs84.transform(float(Xu[0,0]), float(Yu[0,0]))

    xc, yc, zc = geodetic_to_ecef(latc, lonc, h0)
    xr, yr, zr = geodetic_to_ecef(latr, lonr, h0)
    xu, yu, zu = geodetic_to_ecef(latu, lonu, h0)

    ec, nc, uc = ecef_to_enu(xc, yc, zc, lat0, lon0, h0)
    er, nr, ur = ecef_to_enu(xr, yr, zr, lat0, lon0, h0)
    eu, nu, uu = ecef_to_enu(xu, yu, zu, lat0, lon0, h0)

    # 每个像素在 ENU 的“典型”尺度（右移、下移）
    step_x = float(np.hypot(er-ec, nr-nc))
    step_y = float(np.hypot(eu-ec, nu-nc))
    step = max(step_x, step_y, 1e-6)

    shrink_m = shrink_px * step

    # 内缩裁剪框
    emin2 = emin + shrink_m
    emax2 = emax - shrink_m
    nmin2 = nmin + shrink_m
    nmax2 = nmax - shrink_m

    # 防止极端情况下 shrink 过大
    if emin2 >= emax2 or nmin2 >= nmax2:
        return (emin, nmin, emax, nmax)

    return (emin2, nmin2, emax2, nmax2)
def crop_mesh_to_box_enu(mesh, crop_box_enu):
    emin, nmin, emax, nmax = crop_box_enu
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(emin, nmin, -1e18),
        max_bound=(emax, nmax,  1e18),
    )
    cropped = mesh.crop(aabb)
    cropped = cropped.remove_unreferenced_vertices()
    cropped.remove_degenerate_triangles()
    cropped.remove_duplicated_triangles()
    cropped.compute_vertex_normals()
    return cropped

def simplify_mesh_vertex_clustering(mesh, voxel_size=0.3):
    mesh_s = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average
    )
    mesh_s.remove_degenerate_triangles()
    mesh_s.remove_duplicated_triangles()
    mesh_s.remove_unreferenced_vertices()
    mesh_s.compute_vertex_normals()
    return mesh_s

def simplify_mesh_qem(mesh, target_triangles=200000, preserve_boundary=True):
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_unreferenced_vertices()

    tri0 = len(mesh.triangles)
    if tri0 <= target_triangles:
        return mesh

    mesh_s = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )

    # 清理
    mesh_s.remove_degenerate_triangles()
    mesh_s.remove_duplicated_triangles()
    mesh_s.remove_unreferenced_vertices()
    mesh_s.compute_vertex_normals()
    return mesh_s
# ---------------------------
# metadata.xml
# ---------------------------
def write_metadata_xml(out_root, lat0, lon0):
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<ModelMetadata version="1">
\t<!--Spatial Reference System-->
\t<SRS>ENU:{lat0:.5f},{lon0:.5f}</SRS>
\t<!--Origin in Spatial Reference System-->
\t<SRSOrigin>0,0,0</SRSOrigin>
\t<Texture>
\t\t<ColorSource>Visible</ColorSource>
\t</Texture>
</ModelMetadata>
'''
    with open(os.path.join(out_root, "metadata.xml"), "w", encoding="utf-8") as f:
        f.write(xml)

# ---------------------------
# Main pipeline
# ---------------------------
def process(dom_path, dsm_path, out_root,
            tile_size=2048,
            overlap=256,
            voxel_size=0.2,
            poisson_depth=10,
            normal_radius=1.0,
            density_quantile=0.02,
            nodata=None,
            lat0=None, lon0=None, h0=0.0):
    """
    lat0/lon0: ENU 原点（如果不传，则取影像中心点的经纬度）
    h0: ENU 原点高程（一般可设0，或设为DSM中心高程的均值）
    """
    data_dir = os.path.join(out_root, "Data")
    os.makedirs(data_dir, exist_ok=True)

    with rasterio.open(dom_path) as dom_ds, rasterio.open(dsm_path) as dsm_ds:
        if dom_ds.width != dsm_ds.width or dom_ds.height != dsm_ds.height:
            raise ValueError("DOM and DSM size mismatch. Align first.")
        if dom_ds.transform != dsm_ds.transform:
            raise ValueError("DOM and DSM transform mismatch. Align first.")
        if dom_ds.crs != dsm_ds.crs:
            raise ValueError("DOM and DSM CRS mismatch. Align first.")

        width, height = dsm_ds.width, dsm_ds.height
        transform = dsm_ds.transform
        dsm_nodata = dsm_ds.nodata if nodata is None else nodata

        to_wgs84 = build_transformers(dsm_ds.crs)

        # 选择 ENU 原点：默认取影像中心点
        if lat0 is None or lon0 is None:
            cx = width // 2
            cy = height // 2
            Xc, Yc = window_pixel_centers_to_map_xy(transform, cx, cy, 1, 1)
            lon_c, lat_c = to_wgs84.transform(float(Xc[0, 0]), float(Yc[0, 0]))
            lat0 = float(lat_c)
            lon0 = float(lon_c)

        # 写 metadata.xml
        os.makedirs(out_root, exist_ok=True)
        write_metadata_xml(out_root, lat0, lon0)

        tiles = list(iter_tiles(width, height, tile_size, overlap))

        for (x0, y0, w, h) in tqdm(tiles, desc="Tiles"):
            win = Window(x0, y0, w, h)

            # read
            dsm = dsm_ds.read(1, window=win).astype(np.float32)
            dom = dom_ds.read(window=win)  # (bands,h,w)

            # map xy
            X_map, Y_map = window_pixel_centers_to_map_xy(transform, x0, y0, w, h)
            
            # point cloud in ENU
            pcd = build_point_cloud_from_tile(dom, dsm, X_map, Y_map,
                                              to_wgs84, lat0, lon0, h0,
                                              dsm_nodata=dsm_nodata)
            if pcd is None or len(pcd.points) < 2000:
                continue

            if voxel_size is not None and voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                if len(pcd.points) < 2000:
                    continue

            # Poisson
            mesh = poisson_mesh(pcd, depth=poisson_depth,
                                normal_radius=normal_radius,
                                density_quantile=density_quantile)
            if mesh is None:
                continue
       
            # mesh = simplify_mesh_vertex_clustering(mesh, voxel_size=0.3)
            # mesh = simplify_mesh_qem(mesh, target_triangles=len(mesh.triangles) // 4)
            mesh = simplify_mesh_smart(
                mesh,
                tris_per_m2=80.0,     # 你可以试 30~150：值越大越精细
                max_error=1.25,       # 允许偏差（米），城市场景可 0.1~0.3
                enable_clustering=True,
                clustering_voxel_min=0.05,
                clustering_voxel_max=1.5
            )
            crop_box = compute_inner_crop_box_enu(
                transform, x0, y0, w, h,
                overlap_px=overlap,
                to_wgs84=to_wgs84, lat0=lat0, lon0=lon0, h0=h0
            )
            mesh = crop_mesh_to_box_enu(mesh, crop_box)
            tile_name = f"Tile_{x0}_{y0}"
            tile_dir = os.path.join(data_dir, tile_name)
            os.makedirs(tile_dir, exist_ok=True)

            # # 写 OBJ + MTL
            obj_path = os.path.join(tile_dir, f"{tile_name}.obj")
            write_obj_no_texture(mesh, obj_path)

if __name__ == "__main__":
    dom_path = r"D:/pan_align.tif"
    dsm_path = r"D:/dsm_align.tif"
    out_root = r"D:/out"

    process(
        dom_path, dsm_path, out_root,
        tile_size=2048,
        overlap=256,
        voxel_size=0.2,        # 建议设置，不然Poisson慢
        poisson_depth=10,      # 9~12按质量/速度权衡
        normal_radius=1.0,     # 依据点间距调（约2~5倍点间距）
        density_quantile=0.02, # 去边缘低密度伪面
        lat0=None,         # 你希望的 ENU 原点（也可 None 自动取中心）
        lon0=None,
        nodata=10000,
        h0=0.0
    )
