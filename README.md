# ddh2nc

Combine a large no.~ of DDH output files to single NetCDF dataset. Can read
multiple domains. Currently only for vertical profiles. Runs in parallel.


## Prerequisites
  - Python 4.10+
  - ddh toolbox [https://github.com/UMR-CNRM/ddhtoolbox]
    - ensure tools `lfac` and `ddhr` are on `$PATH`

## Installation
  - In your preferred python  environment: `pip install -r requirements.txt`

## Usage
  - On Atos:
    - Edit `config.yaml`
      - `input_directory`: where the DDH files are stored
      - `output_file`: target for NetCDF file
      - `ddh_article_list`:
      - `num_threads`:  for parallel execution (defaults to 1)

    - begin interactive job eg., `ecinteractive -c32 -m 16G -t 2:00:00`
    - `./ddh2nc.py`
