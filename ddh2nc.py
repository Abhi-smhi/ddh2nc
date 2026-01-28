import tarfile
import subprocess
import tempfile
import os
import xarray as xr
import numpy as np
from pathlib import Path
import tqdm


xr.set_options(use_new_combine_kwarg_defaults=True) # To set coordinates explicitly

def tar_to_ds(tar_path, articles, output_nc):
    DataArray_list = []

    with tarfile.open(tar_path, "r:*") as tar:
        # Filter for .fa files
        fasta_members = [m for m in tar.getmembers() if m.name.endswith("s")]

        time_steps = len(fasta_members)
        print(f'Tarball:  number of time-steps {time_steps}')

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            print(f'Created tmp directory {td}')

            # Initialize DataArray list
            mem1 = fasta_members[0]
            fasta_path1 = td/os.path.basename(mem1.name)

            with tar.extractfile(mem1) as f_in, open(fasta_path1, "wb") as f_out:
                f_out.write(f_in.read())

            # 2. Run ddhtoolbox

            # 2.1  Determine number of  levels, domains and time-steps in given tarball
            result = subprocess.run([ "ddhr", "-d", "-n",
                str(fasta_path1)
                ],
                capture_output=True,
                text=True,
                check=True)

            nd = np.fromstring(result.stdout, sep ='\n')
            d = int(nd[0])
            n = int(nd[1])

            print(f'Tarball:  number of domains {d}, levels {n} and {time_steps} time steps')

            #subprocess.run([ "ddhi", "-llist",
           #    str(fasta_path1)
           #    ],
           #    check=True)

           # 2.2 Determine base time
            result = subprocess.run([ "ddhr", "-b",
                str(fasta_path1)
                ],
                capture_output=True,
                text=True,
                check=True)

            date = result.stdout.split('\n')[0]
            date = date[0:10] + 'T' + date[11::]
            baseTime = np.datetime64(date, 's')
            print(baseTime)

            for  article in (articles):
                da = xr.DataArray(data = np.zeros([time_steps,n,d]),
                        dims = [ 'time', 'levels', 'domain', ],
                        coords={'levels':np.arange(n)+1,
                            'domain' : np.arange(d)+1,
                            },
                        name = article
                        )
                # read in attribures for article
               #docFile =  td / f"{fasta_path1.stem}.tmp.{article}.doc"
               #doc  = parse_ddh_attributes(docFile)
               #da.attrs = doc # store attributes
                print(f'Appending article {da.name}')
                DataArray_list.append(da)

            time_list = []
            # Process remaining (all) files
            print(f'Extracting to {tmpdir}')
            for tx, member in (enumerate(tqdm.tqdm(fasta_members))):
                # 1. Extract FASTA to a temp file
                fasta_path = td / os.path.basename(member.name)

                with tar.extractfile(member) as f_in, open(fasta_path, "wb") as f_out:
                    f_out.write(f_in.read())

                # 2.1 Extract base date, convert to numpy date format
                result = subprocess.run([ "ddhr", "-es", str(fasta_path) ], capture_output=True, text=True, check=True)
                time_list.append(baseTime + np.int64(np.fromstring(result.stdout, sep ='\n')[0]))

                # 3. Read in articles from dta files, save to DataArrays
                data = np.ones([n,d])
                for index, article in enumerate(articles):
                    # 2.2 Extract each article
                    result_article = subprocess.run([
                       "lfac",
                       str(fasta_path),
                       article
                    ],
                    capture_output=True,
                    text=True,
                    check=True)

                   # article data
                   #docFile =  td / f"{fasta_path.stem}.tmp.{article}.doc"
                    data = np.fromstring(result_article.stdout, sep ='\n')
                    da_data = data.reshape(d,n).transpose()

                    # non indexed coordinates: pressure and height
                    #da = da.assign_coords(pressure=(['levels','domain'], data[:,0].reshape(d,n).transpose()))
                    #da = da.assign_coords(height=(['levels','domain'], data[:,1].reshape(d,n).transpose()))

                    da_loc=DataArray_list[index]
                    da_loc.data[tx,:,:] = da_data

                os.remove(str(fasta_path))

        if(not os.path.exists(tmpdir)):
            print('\nTemporary dir removed succesfully')

        print(f'{len(DataArray_list)} articles extracted for {d} domains')

# 4. Assemble into an xarray Dataset
    ds = xr.Dataset()
    for index, article in enumerate(articles):
        ds[article] = DataArray_list[index]

    ds.attrs['Number of Domains'] = d
    ds.attrs['Number of levels'] = n
    ds.coords['time'] = time_list

    return ds

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

with open('list') as file:
    articles = [line.rstrip() for line in file]

DS = tar_to_ds(tar_path = 'data/test.tar.gz', articles= articles, output_nc = 'test.nc')
#DS = tar_to_ds(tar_path = 'data/ddh_files.tar.gz', articles= articles, output_nc = 'test.nc')
