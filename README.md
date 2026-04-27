# DDH to Zarr Parallel Processor

This script provides a high-performance, parallelized workflow for extracting data from **DDH (Diagnostic par Domaines Horizontaux)** files and converting them into a modern, cloud-optimized **Zarr** format. It leverages **Dask** for distributed computing and **xarray** for multi-dimensional data management.

---

##  Features

* **Parallel Metadata Extraction:** Uses `dask.bag` to quickly parse validity times and dimensions from large sets of DDH files.
* **Batched Processing:** Implements batching logic to prevent memory overflow and reduce scheduler overhead when dealing with thousands of files.
* **External Tool Integration:** Seamlessly wraps specialized CLI utilities (`ddhr` and `lfac`) via Python’s `subprocess` module.
* **Multi-dimensional Output:** Produces a structured Zarr store indexed by `time`, `level`, and `domain`, with each meteorological variable ("article") as a separate dataset variable.

---

##  Prerequisites

### System Utilities
The script relies on the following external binaries from ddhtoolbox (https://github.com/Hirlam/ddhtoolbox):
* **`ddhr`**: Used for extracting metadata (time, levels, domains) from DDH files.
* **`lfac`**: Used for extracting specific data "articles" from the files.

### Python Dependencies
Ensure you have the following Python packages installed:

```bash
pip install xarray dask zarr numpy
```

---

## Configuration

The script contains a dedicated **USER CONFIG** section at the top. Modify these variables to suit your environment:

| Variable | Description |
| :--- | :--- |
| `INPUT_DIR` | Path to the directory containing your DDH files. |
| `PATTERN` | Glob pattern to match specific files (e.g., `DHFDLDEOD*s`). |
| `OUTPUT_FILE` | Destination path for the `.zarr` output. |
| `ARTICLE_LIST` | Path to a text file containing the names of variables to extract (one per line). |
| `NWORKER` | Number of parallel Dask workers to spin up. |
| `BATCH_SIZE` | Number of files to process per Dask chunk. |
| `MEMLIMIT` | Memory limit allocated per Dask worker (e.g., `'32GB'`). |

---

## Usage

1.  **Prepare your Article List:** Create a file (e.g., `ddh_article_list2`) listing the DDH variables you wish to extract:
    ```text
    PPP
    VUU0
    VUU1
    VCT0
    VCT1
    ```
    etc. Check available DDH articles in each file using:
    ```bash
    lfalaf (yourddhfile)
    ```
    etc. Check available DDH articles in each file using
2.  **Configure the Script:** Edit the `USER CONFIG` section in the Python script as described above.
3.  **Run the Script:**
    ```bash
    python ddh_to_zarr.py
    ```

---

## Workflow Logic

The conversion process follows these steps:

1.  **Metadata Discovery:** The script scans the input directory and uses `ddhr` to determine the number of levels, domains, and the validity timestamps for every file.
2.  **Dask Client Initialization:** A local Dask cluster is started based on your `NWORKER` and `MEMLIMIT` settings.
3.  **Data Extraction:**
    * Files are divided into **Batches**.
    * Workers execute `lfac` calls to read binary data into NumPy arrays.
    * Data is wrapped in `dask.delayed` objects to build a lazy computation graph.
4.  **Xarray Assembly:** A coordinates-aware `DataArray` is constructed, defining the dimensions as `[time, article, level, domain]`.
5.  **Zarr Write:** The lazy dataset is "pushed" into the final Zarr store. The `progress()` bar will track the write operation in real-time.

---

##  Notes

* **Efficiency:** For very large datasets, ensure `BATCH_SIZE` is tuned to your available memory. Small batches increase overhead, while very large batches may lead to worker OOM (Out of Memory) errors.

* **Unit conversion:** This version of the script does *not* convert DDH articles to intensive qunatities.
