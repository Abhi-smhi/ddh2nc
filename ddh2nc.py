import tarfile
import subprocess
import tempfile
import os
import xarray as xr
import numpy as np
from astropy.time import Time as jdTime
from pathlib import Path
import tqdm


xr.set_options(use_new_combine_kwarg_defaults=True) # To set coordinates explicitly
    
def process_tar_to_netcdf(tar_path, articles, output_nc):
    data_all = []

    with tarfile.open(tar_path, "r:*") as tar:
        # Filter for .fa files
        print(f'Tarball: {tar.name}')
        fasta_members = [m for m in tar.getmembers() if m.name.endswith("s")]

        time_steps = len(fasta_members)
        print(f'Tarball:  number of time-steps {time_steps}')


        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            print(f'Created tmp directory {td}')
            
            time_list = []
            # Process remaining (all) files
            print(f'Extracting to {tmpdir}')
            for tx, member in (enumerate(tqdm.tqdm(fasta_members))):
                # 1. Extract FASTA to a temp file
                fasta_path = td / os.path.basename(member.name)
                with tar.extractfile(member) as f_in, open(fasta_path, "wb") as f_out:
                    f_out.write(f_in.read())
                
                # 2. Run ddhtoolbox
                result = subprocess.run([ "ddhr",
                    "-d",
                    "-n",
                    str(fasta_path)
                    ],
                    capture_output=True,
                    text=True,
                    check=True)

                # Determine number of  levels, domains and time-steps in given tarball
                nd = np.fromstring(result.stdout, sep ='\n')
                d = int(nd[0])
                n = int(nd[1])

                # 2.1 Extract julian time, convert to numpy datetime64, append to timelist
                result = subprocess.run([ "ddhr", "-bjul", str(fasta_path) ], capture_output=True, text=True, check=True)
                jd = np.fromstring(result.stdout, sep ='\n')
                jd = jdTime(jd[0], format = 'jd')
                jd = jd.to_value(format='datetime64')
                time_list.append(jd)

                # 2.2 Extract all data given in article list
                subprocess.run([
                   "ddhi", 
                   "-llist",
                   "-1VP",  # Column 1: Vertical pressure level
                   "-2VZ",  # Column 2: Vertical height level
                            # Column 3: contains article data
                    str(fasta_path), 
                ], 
                stdout = subprocess.DEVNULL,
                check=True)

                # 3. Read in articles from dta files, save to DataArrays
                data = np.ones([n,d])
                for index, article in enumerate(articles):
                   # article data
                    datFile =  td / f"{fasta_path.stem}.tmp.{article}.dta"
                    docFile =  td / f"{fasta_path.stem}.tmp.{article}.doc"
                    data = np.loadtxt(datFile) 
                    da_data = data[:,2].reshape(d,n).transpose()
                    
                    # non indexed coordinates: pressure and height
                    #da = da.assign_coords(pressure=(['levels','domain'], data[:,0].reshape(d,n).transpose()))
                    #da = da.assign_coords(height=(['levels','domain'], data[:,1].reshape(d,n).transpose()))

                    # If data_all is empty initialize the Datarrays
                    if (len(data_all) < len(articles)):
                        da = xr.DataArray(data = np.zeros([time_steps,n,d]),
                                dims = [ 'time', 'levels', 'domain', ],
                                coords={'levels':np.arange(n)+1,
                                    'domain' : np.arange(d)+1,
                                    #'time' : [jd]
                                    },
                                name = article
                                )
                        # read in attribures for article
                        doc  = parse_ddh_attributes(docFile)
                        da.attrs = doc # store attributes
                        da[tx]=da_data
                        print(f'Appending article {da.name}')
                        data_all.append(da)
                    elif(da[tx].coords['levels'].shape[0] != n or 
                            da[tx].coords['domain'].shape[0] != d):
                        print('ERROR: CHANGE in levels/domain')
                        return
                    else:
                        da[tx]=da_data

        if(not os.path.exists(tmpdir)):
            print('\nTemporary dir removed succesfully')

        print(f'{len(data_all)} articles extracted for {d} domains')

    # 4. Assemble into an xarray Dataset
        ds = xr.Dataset()
        for index, article in enumerate(articles):
            ds[article] = data_all[index]
    
    ds.attrs['Number of Domains'] = d
    ds.attrs['Number of levels'] = n
    ds.coords['time'] = np.array(time_list)

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

#DS = process_tar_to_netcdf(tar_path = 'data/test.tar.gz', articles= articles, output_nc = 'test.nc')
DS = process_tar_to_netcdf(tar_path = 'data/ddh_files.tar.gz', articles= articles, output_nc = 'test.nc')
