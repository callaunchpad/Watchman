#!/bin/bash

echo "Bash script developed with the purpose of :"
echo " - to show how to use calibration files and projection code by the following example:"
echo " - to visualize provided tracking results by overlaying circles and target names to SALSA frame sequence"
echo "http://tev.fbk.eu/salsa"

if [ "$#" -ne 2 ]; then
    echo "From http://tev.fbk.eu/salsa :"
    echo "- Download Part 1, PosterSession, Video CAM1"
    echo "- Extract CAM1 frames as described in README"
    echo "   ffmpeg -i salsa_ps_cam1.avi cam1/%08d.jpg"
    echo "- Download C++ code and compile project.cpp"
    echo "- Download CAM1 calibration file cam1.calib" 
    echo "- Download HJS-PF tracking result file and extract SALSA.PS.tracking"
    echo "Then, run ./test.sh SALSA.PS.tracking cam1"
    echo "Note that this will overwrite images in cam1"
    echo "Note that you need linux os and imagemagic for this to work"
    exit
fi


while IFS='' read -r line || [[ -n "$line" ]]; do

    token=($line)

    image="$2/${token[0]}.jpg"
    draw=""

    if ((${token[0]} % 20 != 0 )); then
        continue
    fi

    echo ${token[0]}

    echo "" > "outputs/output${token[0]}.csv"

    for i in `seq 1 ${token[1]}`; do

	name=${token[3*${i}-1]}
	X=${token[3*${i}+0]}
	Y=${token[3*${i}+1]}
	
	ret="$(./project $2.calib $X $Y 0.0)"
	xy=(${ret:8})
    
    echo "${name}, $xy" >> "outputs/output${token[0]}.csv"
	# draw="$draw text $xy $name ellipse $xy 5,5 0,360"
    done

    # convert $image -fill white -stroke black -draw "$draw" $image
    # echo $image
    
done < "$1"
