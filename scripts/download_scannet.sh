#!/bin/bash
mkdir load
cd load

# taken from https://github.com/autonomousvision/monosdf/blob/main/scripts/download_dataset.sh
# scannet
wget https://s3.eu-central-1.amazonaws.com/avg-projects/monosdf/data/scannet.tar
tar -xf scannet.tar
rm -rf scannet.tar