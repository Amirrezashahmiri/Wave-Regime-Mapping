"""
cmems_download_points.py
------------------------
Batch-download CMEMS wave data for multiple fixed stations.

• Copernicus-Marine Toolbox ≥ 2.0 required
    pip install --upgrade copernicusmarine

Running:
    python cmems_download_points.py
"""

from __future__ import annotations
import pathlib
import copernicusmarine as cm

# ==============================================================
# 0.  ⤵️  PUT YOUR CREDENTIALS HERE (run once, then keep / comment)
# ==============================================================
USERNAME = ""         #  <- change to your CMEMS username
PASSWORD = ""  #  <- change to your CMEMS password

cm.login(USERNAME, PASSWORD)   # creates ~/.copernicusmarine-credentials
# ----------------------------------------------------------------
# After the very first successful run you may comment the 2 lines
# above; the toolbox will read the stored credentials automatically.
# ----------------------------------------------------------------


# ==============================================================
# 1.  Product options that never change inside the main loop
# ==============================================================
COMMON: dict = {
    "dataset_id": "cmems_mod_glo_wav_my_0.2deg_PT3H-i",
    "dataset_version": "202411",
    "variables": [
        "VHM0", "VHM0_SW1", "VHM0_SW2", "VHM0_WW",
        "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW",
        "VPED", "VSDX", "VSDY",
        "VTM01_SW1", "VTM01_SW2", "VTM01_WW",
        "VTM02", "VTM10", "VTPK",
    ],
    "coordinates_selection_method": "nearest",   # best for single points
    "netcdf_compression_level": 1,
    "disable_progress_bar": True,
}


# ==============================================================
# 2.  Station catalogue – edit or add stations only down here
# ==============================================================
STATIONS: dict[str, dict] = {
    "Barka":         dict(lon=58.0862,    lat=23.7700,
                          segments=[("2018-05-16T03:00:00", "2018-10-15T06:00:00")]),
    "Duqum":         dict(lon=57.8052167, lat=19.7719833,
                          segments=[("2019-02-15T09:00:00", "2019-04-15T06:00:00")]),
    "Fahal":         dict(lon=58.4991667, lat=23.6593667, segments=[
                          ("2017-11-23T12:00:00", "2017-12-24T09:00:00"),
                          ("2017-12-25T15:00:00", "2018-04-03T09:00:00"),
                          ("2018-04-10T06:00:00", "2018-07-02T09:00:00"),
                          ("2018-10-23T09:00:00", "2019-01-13T03:00:00")]),
    "Ghubrah":       dict(lon=58.4067500, lat=23.6212833,
                          segments=[("2018-03-06T09:00:00", "2018-05-15T06:00:00")]),
    "Masirah":       dict(lon=58.6789, lat=20.1596833333333,
                          segments=[("2019-02-17T12:00:00", "2019-04-16T03:00:00")]),                     
    "Inshore_Suwayq":dict(lon=57.44221,   lat=23.94223,  segments=[
                          ("2021-10-07T06:00:00", "2022-01-27T09:00:00"),
                          ("2022-07-27T03:00:00", "2022-10-12T03:00:00")]),
    "Quriyat_North": dict(lon=58.9260,    lat=23.2957,
                          segments=[("2018-11-08T09:00:00", "2019-01-10T09:00:00")]),
    "Quriyat_South": dict(lon=58.925924,  lat=23.281023, segments=[
                          ("2021-07-08T09:00:00", "2021-10-12T03:00:00"),
                          ("2021-10-20T03:00:00", "2022-01-19T06:00:00"),
                          ("2022-02-10T03:00:00", "2022-03-31T12:00:00"),
                          ("2022-05-23T03:00:00", "2022-07-26T00:00:00")]),
    "Shywaimiya":    dict(lon=55.5511833, lat=17.8568333,
                          segments=[("2019-02-13T12:00:00", "2019-04-14T12:00:00")]),
    "Taqah":         dict(lon=54.3444833, lat=17.0157167,
                          segments=[("2019-02-11T12:00:00", "2019-04-11T06:00:00")]),
    "Raqqat_Suwayq": dict(lon=57.49815,   lat=24.03608,  segments=[
                          ("2021-01-18T03:00:00", "2021-09-30T03:00:00"),
                          ("2021-10-07T09:00:00", "2022-01-27T03:00:00"),
                          ("2022-05-11T03:00:00", "2022-10-14T06:00:00")]),
    "Wudam_North":   dict(lon=57.61904,   lat=23.87524,
                          segments=[("2021-03-18T12:00:00", "2021-06-16T03:00:00")]),
    "Wudam_South":   dict(lon=57.59747,   lat=23.81999,
                          segments=[("2021-03-18T12:00:00", "2021-06-16T03:00:00")]),
    "Sawadi":        dict(lon=57.77933,   lat=23.80496,
                          segments=[("2022-01-27T06:00:00", "2022-05-11T03:00:00")]),
}


# ==============================================================
# 3.  Destination folder
# ==============================================================
OUTDIR = pathlib.Path(__file__).with_name("cmems_downloads")
OUTDIR.mkdir(exist_ok=True)


# ==============================================================
# 4.  Main loop
# ==============================================================
for site, meta in STATIONS.items():
    lon, lat = meta["lon"], meta["lat"]

    for idx, (t0, t1) in enumerate(meta["segments"], start=1):
        print(f"⏬  {site:<15} | segment {idx:<2} | {t0} ➜ {t1}")

        response = cm.subset(
            minimum_longitude = lon,
            maximum_longitude = lon,
            minimum_latitude  = lat,
            maximum_latitude  = lat,
            start_datetime    = t0,
            end_datetime      = t1,
            output_directory  = OUTDIR,
            output_filename   = f"{site}_segment_{idx}.nc",
            **COMMON,
        )

        print(f"     ↳ saved → {response.file_path}\n")

print("🎉  All downloads finished.")
