import subprocess
import tempfile
import os
import xarray as xr
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm


xr.set_options(use_new_combine_kwarg_defaults=True) # To set coordinates explicitly

def parse_ddh_attributes(file_path):
    """
    Extracts ddh attributs from .doc file, returns dict
    """
    attributes = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Remove '#' and split by the first colon
                content = line.lstrip('#').strip()
                if '=' in content:
                    key, value = content.split('=', 1)
                    #if key in ['TITRE','UNITE']:
                    attributes[key.strip().lower()] = value.strip()
            else:
                # Break early if attributes are only at the top
                # or continue if they are scattered
                continue
    return attributes


def read_file_Tnd(fasta_path):
    '''
    Returns list of time (numpy datetime64), no. of levels (int) and domains (int)
    '''
    # 1. run ddhtoolbox
    # 1.1  determine number of  levels, domains and time-steps for file
    result = subprocess.run([ "ddhr", '-b', '-es', '-d', '-n',
        str(fasta_path)
        ],
        capture_output=True,
        text=True,
        check=True)

    result = result.stdout
    date = result[0:16]
    date = date[0:10] + 'T' + date[11::]

    Dnd = np.fromstring(result[17:], sep = '\n')
    D = int(Dnd[0]) # Duration in s
    d = int(Dnd[1]) # no. of domains
    n = int(Dnd[2]) # no. of levels
    time = np.datetime64(date, 's') + D

    return [time, n, d]

def extract_article(fasta_path, article, d):
    '''
    Parses DDH file for given article, returns numpy array of size (levels, domains)

    arguments:
    fasta_path: DDH file name
    article: article name
    d: no. of domains
    '''
    result_article = subprocess.run([
       "lfac",
       str(fasta_path),
       article
    ],
    capture_output=True,
    text=True,
    check=True)

    data = np.fromstring(result_article.stdout, sep ='\n')
    n = int(data.shape[0]/d)
    return  data.reshape(d,n).transpose()

def worker_ddh(chunk_list, worker_id, articles = None):
    """
    Worker: Runs Unix tools ddhr and lfac on DDH file list, extracts
    articles, and returns a single xarray Dataset.
    """

    ds_list = []
    time, n, d = read_file_Tnd(chunk_list[0])

    pbar = tqdm.tqdm(chunk_list, desc=f"Worker {worker_id}", position=worker_id + 1, leave=False)

    for file in pbar:
        ds = xr.Dataset()

        time, n, d = read_file_Tnd(file)
        for article in articles:
            data_read = extract_article(file, article, d)

            #  store data as DataArray for article
            da = xr.DataArray(data = data_read,
                    dims = [ 'levels', 'domain' ],
                    coords={'levels':np.arange(n)+1,
                        'domain' : np.arange(d)+1,
                        'time': time
                        },
                    name = article
                    )

            #  assemble article to DataSet
            ds[article] = da

        ds.attrs['Number of Domains'] = d
        ds.attrs['Number of levels'] = n
        ds_list.append(ds.expand_dims('time'))

    # combine to dataset for this worker
    return xr.combine_by_coords(ds_list)


def dir_pipeline(dir_path, output_nc, articles):

    """
    Main process:
    @input dir_path: directory where DDH files are stored (DHFDL*s)
    @input output_nc: absolute path to write netcdf output
    @input articles: list of DDH articles for extraction
    """

    base_dir = Path(dir_path)
    # 1.  List  comprehension to resolve absolute paths (serial)
    fasta_files = [f.resolve() for f in base_dir.glob('DHFDL*s')]

    print(f"{len(articles)} articles")

    ds_list = []

    # 1. Determine number of workers
    num_workers = os.cpu_count()
    num_workers = 4
    num_workers = num_workers or 1

    # 2. Split files into chunks (OMP-style)
    chunks = [list(c) for c in np.array_split(fasta_files, num_workers)]

    # 3.  Split chunks among  threads for processing (parallel)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(worker_ddh, chunk,i, articles)
            for i, chunk in enumerate(chunks)
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            result = future.result()
            ds_list.append(result)


    # 3. Assemble final DataSet
    full_ds = xr.combine_by_coords(ds_list)
    return full_ds

    # 4. Export
    full_ds.to_netcdf(output_nc)
    print(f"NetCDF saved to {output_nc}")


with open('listFull2') as file:
    articles = [line.rstrip() for line in file]

DS = dir_pipeline(dir_path = 'data',articles=articles, output_nc = 'out.nc')
