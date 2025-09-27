#!/usr/bin/bash
bash -c " \
if ! [ -d ./venv ]; then python -m venv venv; fi
source ./venv/bin/activate
clear
python ./chat.py
"
