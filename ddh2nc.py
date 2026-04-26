import subprocess
import xarray as xr
import numpy as np
import glob as glob
import dask.array as da
import dask.bag as db
import dask
from dask.distributed import Client, get_client, progress

# ┌────────────────────────────────────────────────────────────────────────────┐
# │                           USER CONFIG STARTS HERE                          │
# └────────────────────────────────────────────────────────────────────────────┘
INPUT_DIR       = "/ec/res4/scratch/swe7088/deode/osm_pgd_ddh_CY49t2_HARMONIE_AROME_LES_input_Paris_200m_linear_20230820/archive/2023/08/20/12/mbr000/"
PATTERN         = 'DHFDLDEOD*s'
OUTPUT_FILE     = "/perm/swe7088/dask_test_out.zarr"
ARTICLE_LIST    = "ddh_article_list2"
NWORKER         = 4
BATCH_SIZE      = 100
MEMLIMIT        = '16GB'
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

    try:
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
    except:
        return np.datetime64('NaT')

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


def read_batch(files, articles, n_levels, n_domains):
    data_batch = []
    for f in files:
        data = read_DDH_data(f, articles, n_levels, n_domains)
        data_batch.append(data)

    return np.stack(data_batch, axis=0)


if __name__ == "__main__":
    xr.set_options(use_new_combine_kwarg_defaults=True) # To set coordinates explicitly

    with open(ARTICLE_LIST) as file:
        articles = [line.rstrip() for line in file]

    file_list = sorted(glob.glob(INPUT_DIR + PATTERN))[0:150]

    n_times = len(file_list)
    print(f'\n\n-------------  INFO  -------------')
    print(f'Found {n_times} DDH files')

    # 2. Parse no. of files and domains

    n_levels, n_domains = read_file_nd(file_list[0])
    print(f'DDH files contain {n_domains} domains, of {n_levels} levels')
    print(f'\----------------------------------')


    with Client(n_workers=NWORKER, threads_per_worker=1, memory_limit=MEMLIMIT) as client:
        print(f'\n\nDask client: {client}\n\n')

        files_bag = db.from_sequence(file_list, npartitions=NWORKER*4)

        print('[Dask] Computing DDH validities')
        processed_time = files_bag.map(read_file_T)
        persist_times = client.persist(processed_time)
        progress(persist_times)
        actual_times = np.array(persist_times.compute())

        print('Reading DDH data')

        # ---  saves the scheduler from pickling the entire list for every woker
        articles_future = client.scatter(articles, broadcast=True)
        file_batches = [file_list[i:i+BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)]

        lazy_batches = []

        for batch in file_batches:
            current_shape = (len(batch), len(articles), n_levels, n_domains)

            d_part  = dask.delayed(read_batch)(batch, articles_future, n_levels, n_domains)

            b_array = da.from_delayed(d_part, shape=current_shape, dtype='float64')

            lazy_batches.append(b_array)

        da_stack = da.concatenate(lazy_batches, axis=0)
        print(f'[Dask] Reading data in {len(file_batches)} batches')
        persist_data = client.persist(da_stack)
        progress(persist_data)


