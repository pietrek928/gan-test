from os.path import isfile
from subprocess import run
from urllib.request import urlretrieve

from dirs import get_path


def clone(repo_path):
    run(f'git clone "https://github.com/{repo_path}"')


def download(url, fname):
    if not isfile(get_path(fname)):
        print(f'Downloading {url}...')
        urlretrieve(url, get_path(fname))


def download_imagenet_1024():
    download('https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
             'vqgan_imagenet_f16_1024.yaml')
    download('https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
             'vqgan_imagenet_f16_1024.ckpt')


def download_imagenet_16384():
    download('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
             'vqgan_imagenet_f16_16384.yaml')
    download('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
             'vqgan_imagenet_f16_16384.ckpt')


def download_all():
    download_imagenet_1024()
    download_imagenet_16384()
