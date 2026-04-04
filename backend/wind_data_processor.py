
import rasterio
import numpy as np
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import reproject, Resampling
import os
from pathlib import Path

def save_like_reference(out_path, array, ref_path, dtype="float32"):
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype=dtype, compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)

def process_wind_data(grib_path: str | None = None, ref_path: str | None = None, output_dir: str | None = None):
    project_root = Path(__file__).resolve().parent.parent
    grib_path = Path(grib_path) if grib_path else (project_root / "downloadeduwindvwind.grib")
    ref_path = Path(ref_path) if ref_path else (project_root / "data" / "slope.tif")
    output_dir = Path(output_dir) if output_dir else ref_path.parent

    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB file not found: {grib_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {ref_path}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(ref_path)) as ref:
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs
        aoi_bounds = ref.bounds

    with rasterio.open(str(grib_path)) as src:
        window = from_bounds(
            aoi_bounds.left, aoi_bounds.bottom,
            aoi_bounds.right, aoi_bounds.top,
            transform=src.transform
        )
        win_transform = window_transform(window, src.transform)
        
        num_bands = src.count
        num_timesteps = num_bands // 2

        for k in range(num_timesteps):
            u_band_idx = 2 * k + 1
            v_band_idx = 2 * k + 2

            if u_band_idx > num_bands or v_band_idx > num_bands:
                break

            u_src = src.read(u_band_idx, window=window).astype("float32")
            v_src = src.read(v_band_idx, window=window).astype("float32")

            u_dst = np.empty(dst_shape, dtype="float32")
            v_dst = np.empty(dst_shape, dtype="float32")

            reproject(
                source=u_src,
                destination=u_dst,
                src_transform=win_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

            reproject(
                source=v_src,
                destination=v_dst,
                src_transform=win_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

            speed_dst = np.sqrt(u_dst**2 + v_dst**2)
            direction_to_dst = (np.degrees(np.arctan2(u_dst, v_dst)) + 360.0) % 360.0
            
            time_step_dir = output_dir / f"wind_t{k}"
            time_step_dir.mkdir(parents=True, exist_ok=True)

            save_like_reference(str(time_step_dir / "wind_u_resampled.tif"), u_dst, str(ref_path))
            save_like_reference(str(time_step_dir / "wind_v_resampled.tif"), v_dst, str(ref_path))
            save_like_reference(str(time_step_dir / "wind_speed_resampled.tif"), speed_dst, str(ref_path))
            save_like_reference(str(time_step_dir / "wind_direction_to_resampled.tif"), direction_to_dst, str(ref_path))

            neighbor_angles = {
                "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
                "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
            }

            for name, ang in neighbor_angles.items():
                delta = np.radians(((ang - direction_to_dst + 540.0) % 360.0) - 180.0)
                v_proj = speed_dst * np.cos(delta)
                kw_field = np.exp(0.1783 * v_proj).astype("float32")
                save_like_reference(str(time_step_dir / f"kw_{name}.tif"), kw_field, str(ref_path))

    print(f"Wind data processing complete in {output_dir}")

if __name__ == "__main__":
    process_wind_data()
