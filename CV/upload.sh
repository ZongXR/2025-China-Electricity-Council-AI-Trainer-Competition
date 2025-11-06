#!/bin/bash
mkdir -p save_result
mv output/result.json save_result
tar -cvf save_result.tar save_result
python ./upload.py