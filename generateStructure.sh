#!/bin/bash

# Check if folder name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

FOLDER=$1

mkdir -p $FOLDER
cd $FOLDER
mkdir build lib 
touch CMakeLists.txt
cd lib
touch pass.cpp
cd ..
touch test1.c
touch test2.c
touch test3.c
touch generateCFG.sh
chmod +x generateCFG.sh
touch runPass.sh
chmod +x runPass.sh

echo "Project structure generated in $FOLDER"