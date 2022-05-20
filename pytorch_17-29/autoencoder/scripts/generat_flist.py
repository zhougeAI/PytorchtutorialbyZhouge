import glob
import argparse
import os

parser = argparse.ArgumentParser(description='generate flist parameters')
parser.add_argument('--datapath',type=str,help='not use')
parser.add_argument('--flistpath',type=str)
args = parser.parse_args()


def generate_flist(data_path,flist_path):
    assert os.path.exists(data_path),'Directory of datapath doesn\'t exist.'
    assert os.path.exists(flist_path), 'Directory of flistpath doesn\'t exist.'
    data_path_list = []
    for name in glob.glob(data_path + '/*'):
        data_path_list.append(name)
    with open(flist_path,'w') as f:
        for item in data_path_list:
            f.write(item + '\n')
        f.close()
    print("Saving flist file successful at " + flist_path + '.')


if __name__ == '__main__':
    generate_flist(args.datapath,args.flistpath)
