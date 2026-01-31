import subprocess
import os
import xarray as xr
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import sys
import yaml

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

    DataArray_list = []
    time_steps = len(chunk_list)
    time, n, d = read_file_Tnd(chunk_list[0])

   #pbar = tqdm.tqdm(chunk_list, desc=f"Worker {worker_id}", position=worker_id + 1, leave=False)
    pbar = chunk_list

    # 1. Initialize DataArrays, one for each article
    for  article in (articles):
        n, d = extract_article(chunk_list[0], article, d).shape # testing
        da = xr.DataArray(data = np.zeros([time_steps,n,d]),
                dims = [ 'time', 'levels', 'domain', ],
                coords={'levels':np.arange(n)+1,
                    'domain' : np.arange(d)+1,
                    },
                name = article
                )
       #print(f'Appending article {da.name}')
        DataArray_list.append(da)

    time_list = []
    for i, file in enumerate(pbar):

        print(f'PID [{os.getpid()}]: files {i} of {len(chunk_list)}')

        time, n, d = read_file_Tnd(file)
        time_list.append(time)
        for index, article in enumerate(articles):
            da_data = extract_article(file, article, d)

            # 4.2.1 Update each DataArray in master list
            da_loc = DataArray_list[index]
            da_loc.data[i,:,:] = da_data

    ds = xr.Dataset()

    for index, article in enumerate(articles):
        ds[article] = DataArray_list[index]

    ds.attrs['Number of Domains'] = d
    ds.attrs['Number of levels'] = n
    ds.coords['time'] = time_list
    return ds

def dir_list(dir_path):

    base_dir = Path(dir_path)
    print(f'Scanning dir {dir_path} ...')
    fasta_files = [f.resolve() for f in base_dir.glob('DHFDL*s')]
    print(f'Found {len(fasta_files)} DDH files')

    return fasta_files

def files_pipeline(fasta_files, articles, num_workers):

    """
    Main process:
    @input dir_path: directory where DDH files are stored (DHFDL*s)
    @input articles: list of DDH articles for extraction
    """

    print(f"{len(articles)} articles to be extracted")

    ds_list = []

    num_workers = num_workers or 1

    # 2. Split files into chunks (OMP-style)
    chunks = [list(c) for c in np.array_split(fasta_files, num_workers)]

    # 3.  Split chunks among  threads for processing (parallel)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(worker_ddh, chunk,i, articles)
            for i, chunk in enumerate(chunks)
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            result = future.result()
            ds_list.append(result)


    # 3. Assemble final DataSet
    full_ds = xr.concat(ds_list, dim='time')
    return full_ds

def load_config(config_path):
    """Loads and validates the YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, 'r') as file:
        # 1. Read the raw text from the file
        raw_config = file.read()

        # 2. Expand environment variables in the string
        expanded_config = os.path.expandvars(raw_config)
        try:
            config = yaml.safe_load(expanded_config)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            sys.exit(1)

    return config

def parse_config():
    """Parses config.yaml, returns list [input_dir, output_file, article_list, threads]"""

    config_file = "config.yaml"
    cfg = load_config(config_file)

    # Extracting variables with defaults for safety
    input_dir = cfg.get('input_directory')
    output_file = cfg.get('output_file')
    article_list = cfg.get('article_list')
    threads = cfg.get('num_threads', 1) # Default to 1 if not specified

    # Simple validation
    required_fields = ['input_directory', 'output_file', 'article_list']
    for field in required_fields:
        if not cfg.get(field):
            print(f"Error: Missing required field '{field}' in {config_file}")
            sys.exit(1)

    print("\n\n--- Configuration Loaded ---")
    print(f"Directory: {input_dir}")
    print(f"Output:    {output_file}")
    print(f"Articles:  {article_list}")
    print(f"Threads:   {threads}")
    print("----------------------------")

    return [input_dir, output_file, article_list, threads]



if __name__ == "__main__":
    input_dir, output_file, article_list, threads =  parse_config()

    with open(article_list) as file:
        articles = [line.rstrip() for line in file]

    file_list = dir_list(input_dir)
    DS = files_pipeline(file_list, articles=articles, num_workers = threads)
    print(DS)
    print(f'Writing NetCDF file to {output_file}')
    DS.to_netcdf(output_file)

