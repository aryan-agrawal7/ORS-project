#!/usr/bin/env python3
"""
Google Earth Engine script to generate LSSVM training data.

Exports the five driving factors used in Xu et al. (2022):
  1. Slope       (degrees)        — from SRTM 30 m DEM
  2. Aspect      (degrees, 0–360) — from SRTM 30 m DEM
  3. Elevation   (metres)         — from SRTM 30 m DEM
  4. NDVI        (normalised)     — from Landsat 8/9 cloud-free composite
  5. Rel. Humidity (%)            — from ERA5-Land hourly reanalysis

Fire / non-fire point labels are sampled from the MODIS MCD64A1 burned-area
product for the same region and time period.

Outputs
-------
  1. Five single-band GeoTIFFs (one per feature) clipped to the study AOI.
  2. A CSV file with columns:
       slope, aspect, elevation, ndvi, humidity, label
     where label = +1 (fire) or -1 (non-fire).

Prerequisites
-------------
  pip install earthengine-api
  earthengine authenticate        # one-time browser login
  earthengine set-project <YOUR_CLOUD_PROJECT>

Usage
-----
  python gee_export.py                          # defaults — Xichang, Sichuan
  python gee_export.py --lat 27.85 --lon 102.25 --buf 25000 --year 2020

Known-good fire locations
-------------------------
  SCU Lightning Complex 2020:   --lat 37.4  --lon -121.6  --year 2020
  Creek Fire 2020 (Sierra NF):  --lat 37.2  --lon -119.25 --year 2020
  August Complex 2020:          --lat 39.9  --lon -122.7  --year 2020
  Bandipur, India 2019:         --lat 11.66 --lon 76.63   --buf 25000 --year 2019
  NSW, Australia 2020:          --lat -36.0 --lon 149.5   --year 2020
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import ee


# ╔══════════════════════════════════════════════════════════════╗
# ║  Configuration — edit these or pass via CLI flags            ║
# ╚══════════════════════════════════════════════════════════════╝

DEFAULT_LAT     = 27.85        # Study area center latitude  (Xichang, Sichuan — paper site)
DEFAULT_LON     = 102.25       # Study area center longitude
DEFAULT_BUFFER  = 25_000       # Buffer radius in metres (25 km → ~50 km square region)
DEFAULT_YEAR    = 2020         # Year for fire events & imagery
DEFAULT_N_NOFIRE= 1000         # Number of non-fire sample points (fire points = ALL)
DEFAULT_SCALE   = 30           # Export resolution in metres
DEFAULT_FOLDER  = "lssvm_fire" # Google Drive export folder

# ERA5-Land band for relative humidity (derived from dewpoint & temperature)
# ERA5-Land provides dewpoint_temperature_2m and temperature_2m; we compute RH.
ERA5_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"


# ╔══════════════════════════════════════════════════════════════╗
# ║  Helpers                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def _step(msg: str, t0: float) -> float:
    """Print a progress step with elapsed time since last step."""
    now = time.time()
    elapsed = now - t0
    print(f"  ✓ {msg}  ({elapsed:.1f}s)")
    return now

def _spin(msg: str):
    """Print a step-in-progress indicator (no newline yet)."""
    print(f"  ⏳ {msg} ... ", end="", flush=True)

def _spin_done(t0: float):
    """Finish a spinning indicator with elapsed time."""
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

def compute_relative_humidity(era5_img: ee.Image) -> ee.Image:
    """
    Compute relative humidity (%) from ERA5-Land 2-m temperature and
    2-m dewpoint temperature using the Magnus formula:

        es(T) = 6.112 * exp(17.67 * T / (T + 243.5))
        RH    = 100 * es(Td) / es(T)

    ERA5-Land stores temperatures in Kelvin.
    """
    t2m  = era5_img.select("temperature_2m").subtract(273.15)            # → °C
    td2m = era5_img.select("dewpoint_temperature_2m").subtract(273.15)   # → °C

    es_t  = t2m.multiply(17.67).divide(t2m.add(243.5)).exp().multiply(6.112)
    es_td = td2m.multiply(17.67).divide(td2m.add(243.5)).exp().multiply(6.112)

    rh = es_td.divide(es_t).multiply(100).clamp(0, 100)
    return rh.rename("humidity").toFloat()


def cloud_mask_l8(img: ee.Image) -> ee.Image:
    """Apply QA_PIXEL cloud mask for Landsat 8/9 SR Collection 2."""
    qa = img.select("QA_PIXEL")
    cloud_bit  = 1 << 3   # cloud
    shadow_bit = 1 << 4   # cloud shadow
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return img.updateMask(mask)


def _count_burned_pixels(aoi: ee.Geometry, date_start: str, date_end: str) -> int:
    """Count MODIS MCD64A1 burned pixels in a given AOI (at native 500 m)."""
    burned = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(date_start, date_end)
        .select("BurnDate")
    )
    burned_mask = burned.max().gt(0)
    count = burned_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=500,
        maxPixels=int(1e8),
    ).get("BurnDate")
    return int(ee.Number(count).getInfo())


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main export logic                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def run_export(
    lat: float       = DEFAULT_LAT,
    lon: float        = DEFAULT_LON,
    buffer_m: int     = DEFAULT_BUFFER,
    year: int         = DEFAULT_YEAR,
    n_nofire: int     = DEFAULT_N_NOFIRE,
    scale: int        = DEFAULT_SCALE,
    drive_folder: str = DEFAULT_FOLDER,
):
    total_t0 = time.time()

    # ── Initialise Earth Engine ────────────────────────────────
    _spin("Initialising Earth Engine")
    t0 = time.time()
    ee.Initialize()
    _spin_done(t0)

    center     = ee.Geometry.Point([lon, lat])
    date_start = f"{year}-01-01"
    date_end   = f"{year}-12-31"

    # ────────────────────────────────────────────────────────────
    #  Step 0.  Verify burned pixels exist (CHEAP — 500 m MODIS)
    #
    #  MODIS MCD64A1 is 500 m.  A tiny buffer (e.g. 1 km) may
    #  contain zero burned pixels even in a fire-prone region.
    #  We check first at the requested buffer, then automatically
    #  try larger radii if nothing is found, and finally offer
    #  known-good presets the user can pick from.
    # ────────────────────────────────────────────────────────────
    print("[0/6] Pre-checking burned area (MODIS MCD64A1 @ 500 m) ...")
    t0 = time.time()

    # Enforce a minimum buffer of 5 km (10 MODIS pixels across)
    if buffer_m < 5000:
        print(f"  ⚠️  Buffer {buffer_m} m is very small for 500 m MODIS data.")
        print(f"     Auto-expanding buffer from {buffer_m} m → 5000 m.")
        buffer_m = 5000

    aoi = center.buffer(buffer_m).bounds()

    _spin(f"Counting burned pixels in {buffer_m} m buffer")
    st = time.time()
    n_burned = _count_burned_pixels(aoi, date_start, date_end)
    _spin_done(st)
    print(f"  → Burned MODIS pixels (500 m): {n_burned}")

    # If none found, auto-expand progressively
    if n_burned == 0:
        expand_radii = [10_000, 25_000, 50_000, 100_000]
        for r in expand_radii:
            if r <= buffer_m:
                continue
            _spin(f"No burns found — trying {r // 1000} km buffer")
            st = time.time()
            test_aoi = center.buffer(r).bounds()
            n_burned = _count_burned_pixels(test_aoi, date_start, date_end)
            _spin_done(st)
            print(f"  → {r // 1000} km buffer: {n_burned} burned pixels")
            if n_burned > 0:
                buffer_m = r
                aoi = test_aoi
                print(f"  ✅ Found burned area! Buffer expanded to {r // 1000} km.")
                break

    if n_burned == 0:
        area_km2 = (2 * buffer_m / 1000) ** 2
        print(f"\n  ❌  No MODIS burned pixels found anywhere within "
              f"{buffer_m // 1000} km of ({lat}, {lon}) for {year}.")
        print(f"     Searched area: ~{area_km2:.0f} km².")
        print(f"\n  Known-good fire locations you can use:")
        print(f"    SCU Lightning Complex 2020 (California):")
        print(f"      --lat 37.4  --lon -121.6  --year 2020")
        print(f"    Creek Fire 2020 (Sierra Nevada):")
        print(f"      --lat 37.2  --lon -119.25 --year 2020")
        print(f"    August Complex 2020 (N. California):")
        print(f"      --lat 39.9  --lon -122.7  --year 2020")
        print(f"    Xichang, Sichuan 2020 (paper study site):")
        print(f"      --lat 27.85 --lon 102.25  --year 2020")
        print(f"    Bandipur, India 2019:")
        print(f"      --lat 11.66 --lon 76.63   --buf 25000 --year 2019")
        print(f"    NSW, Australia 2020:")
        print(f"      --lat -36.0 --lon 149.5   --year 2020")
        sys.exit(1)

    t0 = _step(f"Burned-area pre-check passed ({n_burned} pixels)", t0)

    print(f"\n  📍 AOI center: ({lat}, {lon}), buffer: {buffer_m} m")
    print(f"  📅 Date range: {date_start} → {date_end}\n")

    # ────────────────────────────────────────────────────────────
    #  Step 1–3. Elevation, Slope, Aspect  (SRTM 30 m)
    # ────────────────────────────────────────────────────────────
    print("[1/6] Terrain layers (SRTM 30 m) ...")
    t0 = time.time()
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select("elevation").clip(aoi)
    slope     = ee.Terrain.slope(srtm).clip(aoi)
    aspect    = ee.Terrain.aspect(srtm).clip(aoi)
    t0 = _step("DEM / Slope / Aspect defined", t0)

    # ────────────────────────────────────────────────────────────
    #  Step 4. NDVI  (Landsat 8/9 Collection 2, Level 2 SR)
    # ────────────────────────────────────────────────────────────
    print("[2/6] NDVI composite (Landsat 8/9) ...")
    t0 = time.time()
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .map(cloud_mask_l8)
    )

    def scale_sr(img):
        sr = img.select(["SR_B4", "SR_B5"]).multiply(0.0000275).add(-0.2)
        return sr.copyProperties(img, img.propertyNames())

    l8_sr   = l8.map(scale_sr)
    ndvi    = l8_sr.median().normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi").clip(aoi)
    t0 = _step("NDVI composite defined", t0)

    # ────────────────────────────────────────────────────────────
    #  Step 5. Relative Humidity  (ERA5-Land MONTHLY aggregates)
    # ────────────────────────────────────────────────────────────
    print("[3/6] Relative humidity (ERA5-Land monthly) ...")
    t0 = time.time()

    era5_full = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .select(["temperature_2m", "dewpoint_temperature_2m"])
    )

    months = ee.List.sequence(1, 12)

    def monthly_mean_rh(month):
        month = ee.Number(month)
        m_start = ee.Date.fromYMD(year, month, 1)
        m_end   = m_start.advance(1, "month")
        monthly = era5_full.filterDate(m_start, m_end)
        mean_img = monthly.mean()
        return compute_relative_humidity(mean_img).set("month", month)

    rh_monthly = ee.ImageCollection(months.map(monthly_mean_rh))
    humidity   = rh_monthly.select("humidity").mean().clip(aoi)

    t0 = _step("Relative humidity defined (12 monthly composites)", t0)

    # ────────────────────────────────────────────────────────────
    #  Stack all five bands into one image for sampling
    # ────────────────────────────────────────────────────────────
    feature_stack = (
        slope.rename("slope")
        .addBands(aspect.rename("aspect"))
        .addBands(elevation.rename("elevation"))
        .addBands(ndvi)
        .addBands(humidity)
    ).toFloat()

    # ────────────────────────────────────────────────────────────
    #  Step 6. Fire / Non-fire labels  (MODIS MCD64A1 Burned Area)
    # ────────────────────────────────────────────────────────────
    print("[4/6] Burned-area mask (MODIS MCD64A1) ...")
    t0 = time.time()

    burned = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(date_start, date_end)
        .select("BurnDate")
    )
    burned_max = burned.max()

    # Build a clean integer label band: 1 = burned, 0 = not burned.
    # Keep everything at the MODIS native 500 m so the label aligns
    # perfectly with the pixel grid — no fractional-mask issues.
    burned_binary = burned_max.gt(0).toInt().rename("label")

    # For export / visualisation
    burned_mask = burned_binary.selfMask().clip(aoi).rename("burned")

    # Vegetation mask — exclude water / barren via NDVI > 0.1
    veg_mask = ndvi.gt(0.1)

    t0 = _step("Burned-area mask defined", t0)

    # ────────────────────────────────────────────────────────────
    #  Extract training points
    #
    #  Strategy: add the integer label band to the feature stack,
    #  then sample at the MODIS native 500 m scale.  GEE will
    #  automatically resample the 30 m features (SRTM, Landsat)
    #  to 500 m at each sample point.  This guarantees every
    #  burned MODIS pixel becomes a training row.
    # ────────────────────────────────────────────────────────────
    print("[5/6] Extracting training points (server-side) ...")
    t0 = time.time()

    sample_scale = 500   # MODIS native resolution

    # ── Extract ALL fire pixels (label == 1) ──
    # Mask feature_stack to burned AND vegetated pixels
    _spin("Extracting all fire pixels")
    st = time.time()
    fire_image = feature_stack.updateMask(burned_binary).updateMask(veg_mask)
    fire_points = fire_image.sample(
        region=aoi,
        scale=sample_scale,
        geometries=True,
    )
    actual_fire = fire_points.size().getInfo()
    _spin_done(st)
    print(f"    → {actual_fire} fire pixels extracted")

    # ── Extract matched non-fire pixels ──
    # unmask(0) fills any MODIS no-data gaps with 0 (= not burned),
    # so every vegetated non-burned pixel is eligible.
    n_want_nofire = max(actual_fire, n_nofire)
    _spin(f"Sampling {n_want_nofire} non-fire pixels")
    st = time.time()
    not_burned = burned_binary.unmask(0).eq(0)          # 1 where NOT burned
    nofire_image = feature_stack.updateMask(not_burned).updateMask(veg_mask)
    nofire_points = nofire_image.sample(
        region=aoi,
        scale=sample_scale,
        numPixels=n_want_nofire,
        seed=99,
        geometries=True,
    ).limit(n_want_nofire)
    actual_nofire = nofire_points.size().getInfo()
    _spin_done(st)
    print(f"    → {actual_nofire} non-fire pixels extracted")

    # Re-code labels: fire → +1, non-fire → −1  (paper convention)
    fire_points   = fire_points.map(lambda f: f.set("label", 1))
    nofire_points = nofire_points.map(lambda f: f.set("label", -1))

    training_fc = fire_points.merge(nofire_points)
    t0 = _step(f"Extracted {actual_fire} fire + {actual_nofire} non-fire points", t0)

    if actual_fire == 0:
        print("\n  ⚠️  WARNING: Pre-check found burned pixels but extraction returned 0.")
        print("     This may happen if all burned pixels are over water/barren land (NDVI<0.1).")
        print("     Try a larger --buf or a different region.\n")

    # ════════════════════════════════════════════════════════════
    #  EXPORTS  (all to Google Drive)
    # ════════════════════════════════════════════════════════════

    if(input("Do you wish to export these results? (yes/no)") == "yes"):

        print("[6/6] Submitting export tasks to Google Drive ...")
        t0 = time.time()

        export_tasks = []

        # (A) Individual raster layers
        for band_name, image in [
            ("slope",     slope.rename("slope")),
            ("aspect",    aspect.rename("aspect")),
            ("elevation", elevation.rename("elevation")),
            ("ndvi",      ndvi),
            ("humidity",  humidity),
        ]:
            task = ee.batch.Export.image.toDrive(
                image=image.toFloat(),
                description=f"export_{band_name}",
                folder=drive_folder,
                fileNamePrefix=band_name,
                region=aoi,
                scale=scale,
                crs="EPSG:4326",
                maxPixels=int(1e9),
            )
            task.start()
            export_tasks.append((band_name, task))
            print(f"  → Raster export started: {band_name}.tif")

        # (B) Training CSV
        task_csv = ee.batch.Export.table.toDrive(
            collection=training_fc,
            description="export_training_csv",
            folder=drive_folder,
            fileNamePrefix="training_samples",
            fileFormat="CSV",
            selectors=["slope", "aspect", "elevation", "ndvi", "humidity", "label"],
        )
        task_csv.start()
        export_tasks.append(("training_csv", task_csv))
        print("  → CSV export started: training_samples.csv")

        # (C) Also export the burned-area mask raster (useful for visualisation)
        burned_export = burned_binary.clip(aoi).rename("burned").toByte()
        task_burned = ee.batch.Export.image.toDrive(
            image=burned_export,
            description="export_burned_mask",
            folder=drive_folder,
            fileNamePrefix="burned_mask",
            region=aoi,
            scale=scale,
            crs="EPSG:4326",
            maxPixels=int(1e9),
        )
        task_burned.start()
        export_tasks.append(("burned_mask", task_burned))
        print("  → Raster export started: burned_mask.tif")

        _step(f"All {len(export_tasks)} export tasks submitted", t0)

        total_elapsed = time.time() - total_t0
        print(f"\n{'='*60}")
        print(f"  ✅ All {len(export_tasks)} export tasks submitted to Google Drive.")
        print(f"  📁 Folder:  '{drive_folder}'")
        print(f"  🔗 Monitor: https://code.earthengine.google.com/tasks")
        print(f"  ⏱️  Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")
        print("\nOnce complete, download the folder and run:\n"
            "  python gee_data_loader.py --data-dir ./lssvm_fire\n"
            "to load the data into the simulation.\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CLI entry point                                             ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export LSSVM fire-spread training data from Google Earth Engine."
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT,
                        help=f"Center latitude (default {DEFAULT_LAT})")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON,
                        help=f"Center longitude (default {DEFAULT_LON})")
    parser.add_argument("--buf", type=int, default=DEFAULT_BUFFER,
                        help=f"Buffer radius in metres (default {DEFAULT_BUFFER})")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                        help=f"Year for imagery & fire data (default {DEFAULT_YEAR})")
    parser.add_argument("--n-nofire", type=int, default=DEFAULT_N_NOFIRE,
                        help=f"Max non-fire sample points (default {DEFAULT_N_NOFIRE})")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"Export resolution in metres (default {DEFAULT_SCALE})")
    parser.add_argument("--folder", type=str, default=DEFAULT_FOLDER,
                        help=f"Google Drive folder name (default '{DEFAULT_FOLDER}')")

    args = parser.parse_args()

    try:
        run_export(
            lat=args.lat,
            lon=args.lon,
            buffer_m=args.buf,
            year=args.year,
            n_nofire=args.n_nofire,
            scale=args.scale,
            drive_folder=args.folder,
        )
    except ee.EEException as e:
        print(f"\n[ERROR] Earth Engine error: {e}")
        print("  Make sure you have run:  earthengine authenticate")
        print("  And set a cloud project: earthengine set-project <PROJECT_ID>")
        sys.exit(1)
