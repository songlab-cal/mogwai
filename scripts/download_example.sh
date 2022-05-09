#!/bin/bash
mkdir -p data

wget -P data/test/4rb6Y -q -nc https://files.ipd.uw.edu/krypton/4rb6Y.i90c75.a3m
wget -P data/test/4rb6Y -q -nc https://files.ipd.uw.edu/krypton/4rb6Y.pdb
wget -P data/test/4rb6Y -q -nc https://files.ipd.uw.edu/krypton/4rb6Y.cf

wget -P data/test/ http://s3.amazonaws.com/songlabdata/proteindata/mogwai/3er7_1_A.npz
