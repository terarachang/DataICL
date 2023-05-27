#!/usr/bin/env bash

# gdrive_download
gURL='https://drive.google.com/file/d/1gKueGgRjVKWZ5RXE9PBVyCD5dvbLYdRk/view'
# match more than 26 word characters  
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

ggURL='https://drive.google.com/uc?export=download'
zfile='Dicl.zip'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
echo -e "Downloading from "$gURL"...\n"
eval $cmd


unzip $zfile
rm $zfile
