import argparse
from torchvision import datasets
import numpy as np
import os
import pyapr


def convert_mnist_dataset(dataset, rootdir, converter, aprfile):

    counts = np.zeros(10, dtype=int)

    for dd in dataset:
        img, target = dd

        img = np.array(img).astype(np.float32)

        apr = pyapr.APR()
        parts = pyapr.FloatParticles()
        converter.get_apr(apr, img)
        parts.sample_image(apr, img)

        counts[target] += 1
        fname = str(counts[target]) + '.apr'
        fpath = os.path.join(rootdir, str(target), fname)

        aprfile.open(fpath, 'WRITE')
        aprfile.write_apr(apr)
        aprfile.write_particles('particles', parts)
        aprfile.close()


def main(loc):

    root = os.path.join(loc, 'mnist_apr')
    os.mkdir(root)
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')
    os.mkdir(traindir)
    os.mkdir(testdir)

    for i in range(10):
        os.mkdir(os.path.join(traindir, str(i)))
        os.mkdir(os.path.join(testdir, str(i)))

    pars = pyapr.APRParameters()
    pars.Ip_th = 5
    pars.sigma_th = 10
    pars.sigma_th_max = 2
    pars.gradient_smoothing = 1
    pars.rel_error = 0.1
    pars.auto_parameters = False

    converter = pyapr.converter.FloatConverter()
    converter.set_parameters(pars)
    converter.set_verbose(False)

    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    traindata = datasets.MNIST('../data', train=True, download=True, transform=None)
    convert_mnist_dataset(traindata, traindir, converter, aprfile)

    testdata = datasets.MNIST('../data', train=False, download=True, transform=None)
    convert_mnist_dataset(testdata, testdir, converter, aprfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default='')
    args = parser.parse_args()

    loc = args.location

    if loc:
        main(loc)
    else:
        print('Please give a target directory for output file structure using the --location argument')