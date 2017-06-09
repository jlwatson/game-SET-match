from os import walk

for (dirpath, dirnames, filenames) in walk('./'):
    for filename in filenames:
      if filename.endswith(".JPG"):
        first_part = filename.split('.')[0]
        f= open(first_part + '_labels.txt',"w+")
    