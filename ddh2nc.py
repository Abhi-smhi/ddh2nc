import tarfile
import subprocess
import tempfile
import os
import xarray as xr
import numpy as np
from astropy.time import Time as jdTime
from pathlib import Path

xr.set_options(use_new_combine_kwarg_defaults=True) # To set coordinates explicitly
    
def process_tar_to_netcdf(tar_path, articles, output_nc):
    data_all = []

    print('Extracting tarball to tmpdir')

    with tarfile.open(tar_path, "r:*") as tar:
        # Filter for .fa files
        fasta_members = [m for m in tar.getmembers()]# if m.name.endswith("s")]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            print(f'Created tmp directory {td}')
            
            member = fasta_members[0]
            fasta_path = td / os.path.basename(member.name)
            with tar.extractfile(member) as f_in, open(fasta_path, "wb") as f_out:
                f_out.write(f_in.read())
            
            #  Run ddhtoolbox for file 1
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
            t  = len(fasta_members)

            print(f'Tarball details: number of domains = {d}, number of levels = {n}, number of time-steps {t}')

            for article in articles:
                init = np.empty([t, n, d])
                init[:] = np.nan
                da = xr.DataArray(data = init,
                        dims = ['time', 'levels', 'domain'],
                        coords={'levels':np.arange(n)+1,
                            'domain' : np.arange(d)+1
                            },
                        name = article
                        )

                # non indexed coordinates: pressure and height

                da.assign_coords(pressure=(['time', 'levels','domain'], np.empty([t,n,d])))
                da.assign_coords(height=(['time', 'levels','domain'], np.empty([t,n,d])))

                # Run ddhi, get attributes for DDH articles
                subprocess.run([
                   "ddhi", 
                   "-llist",
                    str(fasta_path), 
                ], 
                stdout = subprocess.DEVNULL,
                check=True)

                # read in attribures for article
                docFile =  td / f"{fasta_path.stem}.tmp.{article}.doc"
                doc  = parse_ddh_attributes(docFile)
                da.attrs = doc # store 
                data_all.append(da)


            time_list = []
            
            # Process remaining (all) files
            for member in fasta_members:
                # 1. Extract FASTA to a temp file
                fasta_path = td / os.path.basename(member.name)
              # print(f"Extracting member {member} to {td}" )
                with tar.extractfile(member) as f_in, open(fasta_path, "wb") as f_out:
                    f_out.write(f_in.read())
                
                # 2. Run ddhtoolbox

                # 2.1 Extract julian time, convert to numpy datetime64, append to timelist
                result = subprocess.run([ "ddhr", "-bjul", str(fasta_path) ], capture_output=True, text=True, check=True)
                jd = np.fromstring(result.stdout, sep ='\n')
                jd = jdTime(jd[0], format = 'jd')
                time_list.append(jd.to_value(format='datetime64'))

                
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
                for index, article in enumerate(articles):
                    datFile =  td / f"{fasta_path.stem}.tmp.{article}.dta"
                    data = np.loadtxt(datFile) 

                    da_local = data_all[index]
                    if da_local.name != article:
                        print(f'Error: {da_local.name} does not match {article}')
                        return
                #return 
               
            print(time_list)
    
        if(not os.path.exists(tmpdir)):
            print('\nTemporary dir removed succesfully')

        print(f'{len(data_all)} articles extracted for {d} domains')
        
        return 

    # 4. Assemble into an xarray Dataset
    # Adjust dims/coords based on your actual data shapes
  # ds = xr.Dataset(
  #     data_vars={
  #         "metric_a": (("sample", "index"), np.array(all_data_1)),
  #         "metric_b": (("sample", "index"), np.array(all_data_2)),
  #     },
  #     coords={
  #         "sample": sample_names,
  #         "index": np.arange(all_data_1[0].shape[0])
  #     }
  # )

  # # 5. Save to NetCDF
  # ds.to_netcdf(output_nc)
  # print(f"Successfully created {output_nc}")

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
                    if key not in ['FICHIER', 'ORIGINE', 'DATE']:
                        attributes[key.strip().lower()] = value.strip()
            else:
                # Break early if attributes are only at the top
                # or continue if they are scattered
                continue 
    return attributes

import numpy as np




# Integration example:
# attrs = parse_ddh_attributes(doc_file_path)
# ds.attrs.update(attrs)  # Add directly to your xarray Dataset metadata


with open('list') as file:
    articles = [line.rstrip() for line in file]

#process_tar_to_netcdf(tar_path = 'data/test.tar.gz', articles= articles, output_nc = 'test.nc')
process_tar_to_netcdf(tar_path = 'data/ddh_files.tar.gz', articles= articles, output_nc = 'test.nc')
