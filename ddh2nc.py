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

def worker_ddh(fasta_path, articles = None):
    """
    Worker: Runs Unix tools ddhr and lfac on DDH file fasta_path, extracts
    articles, and returns a single xarray Dataset.
    """

    print(f"PID [{os.getpid()}]: file {fasta_path.name} ")

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
    #print(f'Time = {time}, d = {d}, n = {n}')

    ds = xr.Dataset()
    for article in articles:

        #  Extract each article, reads from stdout
        result_article = subprocess.run([
           "lfac",
           str(fasta_path),
           article
        ],
        capture_output=True,
        text=True,
        check=True)

        # article data
        data_read = np.fromstring(result_article.stdout, sep ='\n')
        data_read = data_read.reshape(d,n).transpose()

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

    return ds.expand_dims('time')


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

    # 2.  Split file list among  threads for processing (parallel)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(worker_ddh,f, articles)
            for f in fasta_files
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            result = future.result()
            ds_list.append(result)

    # 3. Assemble final DataSet
    full_ds = xr.combine_by_coords(ds_list)
    return full_ds


    # 4. Export
    full_ds.to_netcdf(output_nc)
    print(f"NetCDF saved to {output_nc}")



with open('list') as file:
    articles = [line.rstrip() for line in file]

DS = dir_pipeline(dir_path = 'data',articles=articles, output_nc = 'out.nc')
