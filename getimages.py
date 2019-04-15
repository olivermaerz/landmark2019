from urllib.request import urlretrieve
import tarfile
from io import BytesIO
import os


# loop through the 500 files to download all tar files
for i in range(0, 500):
  number_string = str(i).zfill(3)
  url = 'https://s3.amazonaws.com/google-landmark/train/images_{}.tar'.format(number_string)
  print("Downloading: {}".format(url))
  file_tmp = urlretrieve(url, filename=None)[0]
  # needs to seek in file so open saved temp file
  tar_ref = tarfile.open(file_tmp)
  print("Untaring downloaded file")
  tar_ref.extractall('images/')  
  tar_ref.close()
  print('Removing downloaded tar file')
  os.remove(file_tmp)
