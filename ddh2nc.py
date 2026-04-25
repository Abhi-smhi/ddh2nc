import subprocess
import xarray as xr
import numpy as np
import glob as glob
import dask.array as da
import dask
from dask.distributed import Client, get_client, progress

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                           USER CONFIG STARTS HERE                          │
# └────────────────────────────────────────────────────────────────────────────┘
INPUT_DIR       = "/ec/res4/scratch/swe7088/deode/osm_pgd_ddh_CY49t2_HARMONIE_AROME_LES_input_Paris_200m_linear_20230820/archive/2023/08/20/12/mbr000/"
PATTERN         = 'DHFDLDEOD*s'
OUTPUT_FILE     = "/perm/swe7088/dask_test_out.nc"
ARTICLE_LIST    = "ddh_article_list2"
NWORKER         = 2
MEMLIMIT        = '4GB'
# ┌────────────────────────────────────────────────────────────────────────────┐
# │                           USER CONFIG ENDS HERE                            │
# └────────────────────────────────────────────────────────────────────────────┘

def read_file_T(fasta_path):
    '''
    Extracts validity of DDH file

    param:
    fasta_path: (str) DDH file path

    return:
    time: (np.datetime64) Validity
    '''

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
    Extracts no. of domains and levels of DDH file

    param:
    fasta_path: (str) DDH file path

    return:
    n: (int) no. of levels
    d: (int) no. of domains

    '''
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


def read_DDH_data(path, articles, n_levels, n_domains):

    data = np.zeros([len(articles), n_levels, n_domains])
    for i, article in enumerate(articles):
        result_article = subprocess.run([
           "lfac",
           str(path),
           article
        ],
        capture_output=True,
        text=True,
        check=True)

        data[i, :,:] = (np.fromstring(result_article.stdout, sep ='\n')
                .reshape(n_domains,n_levels).transpose())


    return data




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

      # client = Client()
        print(f"Created new dask client: {client}")

    file_list = sorted(glob.glob(INPUT_DIR + PATTERN))[0:500]

    n_times = len(file_list)
    print(f'Found {n_times} DDH files')

    # 2. Parse no. of files and domains

    n_levels, n_domains = read_file_nd(file_list[0])
    print(f'DDH files contain {n_domains} domains, of {n_levels} levels')

    # -- prepare dask delayed functions
    read_time_delayed = dask.delayed(read_file_T)
    read_DDH_delayed  = dask.delayed(read_DDH_data)

    # 3. Assemble dask arrays

    # -- assess times for all files
    print(f'[Lazy] Assebling dask array for validites')
    times_list = [read_time_delayed(file) for file in file_list]
    # ---- dask array
    time_da = [da.from_delayed(time, shape=(), dtype='datetime64')
            for time in times_list]
    time_stack = da.stack(time_da, axis=0)

    print(f'[Lazy] Assebling dask array for DDH data')
    data_list = [read_DDH_delayed(path=file, articles=articles,
        n_domains=n_domains, n_levels=n_levels)
            for file in file_list]
    # ---- dask array
    data_da = [da.from_delayed(data,
        shape=(len(articles), n_levels, n_domains),
        dtype='float64')
        for data in data_list]
    da_stack = da.stack(data_da, axis=0)


    # --  Assemble the xarray (dask) dataset
    data_vars = {}
    print(f'[Lazy] Assembling xarray dataset')
    for i, article in enumerate(articles):
        data_vars[article] = (['time', 'level', 'domain'], da_stack[:, i,:,:])

    ds_da = xr.Dataset(
            data_vars=data_vars,
            coords = {
                'time': time_stack,
                'level': np.arange(n_levels) +1,
                'domain': np.arange(n_domains) +1,
                }
            )
    print(f'[Lazy] Chunking dataset')
    ds_da = ds_da.chunk({'time':'auto','level':-1, 'domain':-1})
    print(f'[Lazy] Preparing to write')
    jobs = ds_da.to_netcdf(OUTPUT_FILE, mode='w', engine='h5netcdf', compute=False)

    print(f'Compute and writing to {OUTPUT_FILE}')
    futures = client.compute(jobs)
    progress(futures)

    print(f'Job done!\nClosing dask client')
    client.close()

