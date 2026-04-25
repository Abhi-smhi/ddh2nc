import subprocess
import xarray as xr
import numpy as np
import glob as glob
import dask.array as da
import dask
from dask.distributed import Client, get_client

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                           USER CONFIG STARTS HERE                          │
# └────────────────────────────────────────────────────────────────────────────┘
INPUT_DIR       = "/ec/res4/scratch/swe7088/deode/osm_pgd_ddh_CY49t2_HARMONIE_AROME_LES_input_Paris_200m_linear_20230820/archive/2023/08/20/12/mbr000/"
PATTERN         = 'DHFDLDEOD*s'
OUTPUT_FILE     = "/perm/swe7088/dask_test_out.nc"
ARTICLE_LIST    = "ddh_article_list2"
NWORKER         = 1
MEMLIMIT        = '2GB'
# ┌────────────────────────────────────────────────────────────────────────────┐
# │                           USER CONFIG ENDS HERE                            │
# └────────────────────────────────────────────────────────────────────────────┘

def read_file_T(fasta_path):
    '''
    Returns list of time (numpy datetime64), no. of levels (int) and domains (int)
    '''
    # 1. run ddhtoolbox
    # 1.1  determine number of  levels, domains and time-steps for file

    result = subprocess.run([ "ddhr", '-b', '-es',
        str(fasta_path)
        ],
        capture_output=True,
        text=True,
        check=True)

    result = result.stdout
    start = result[0:10] + 'T' + result[11:16]
    D = np.fromstring(result[17:], sep='\n')[0] # Duration in s
    time = np.datetime64(start, 's') + int(D)

    return time

def read_file_nd(fasta_path):
    '''
    Returns list of time (numpy datetime64), no. of levels (int) and domains (int)
    '''
    # 1. run ddhtoolbox
    # 1.1  determine number of  levels, domains and time-steps for file

    result = subprocess.run([ "ddhr", '-d', '-n',
        str(fasta_path)
        ],
        capture_output=True,
        text=True,
        check=True)

    Dnd = np.fromstring(result.stdout, sep='\n')
    d = int(Dnd[0]) # no. of domains
    n = int(Dnd[1]) # no. of levels

    return n, d


if __name__ == "__main__":
    xr.set_options(use_new_combine_kwarg_defaults=True) # To set coordinates explicitly

    with open(ARTICLE_LIST) as file:
        articles = [line.rstrip() for line in file]

    # 1. Setup Dask Cluster
    try:
        client = get_client()
        print(f"Using existing client: {client}")
    except ValueError:
        client = Client(n_workers=NWORKER, threads_per_worker=1, memory_limit=MEMLIMIT)
        print(f"Created new dask client: {client}")

    file_list = sorted(glob.glob(INPUT_DIR + PATTERN))[0:1000]

    n_times = len(file_list)
    print(f'Found {n_times} DDH files')

    # -- Parse no. of files and domains

    n_levels, n_domains = read_file_nd(file_list[0])
    print(f'DDH files contain {n_domains} domains, of {n_levels} levels')

    # -- prepare dask delayed functions
    read_time_delayed = dask.delayed(read_file_T)

    # -- assess times for all files
    times = [read_time_delayed(file) for file in file_list]
    time_d = [da.from_delayed(time, shape=(), dtype='datetime64') for time in times]
    time_stack = da.stack(time_d, axis=0)






