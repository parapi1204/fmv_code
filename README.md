```bash
cd path/to/fmv_code
docker run --name physicsnemo2606 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
           --runtime nvidia -v ${PWD}:/workspace \
           -it nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
```
