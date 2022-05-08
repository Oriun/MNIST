#!/bin/bash
if [ -d data ]
then rm -rf data
fi
cp -r ../datasets ./data
gzip -dkr data
rm data/*.gz