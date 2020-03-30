#xhost +local:root
docker run -it --volume /tmp/.X11-unix:/tmp/.X11-unix:ro --gpus all --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --env NVIDIA_VISIBLE_DEVICES=all -p 5000:5000  centernet_dkw
#xhost -local:root
#docker run -p 5000:5000 --gpus all --name template simplenet
