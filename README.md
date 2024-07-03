# Patch Model

## 설치방법 (Linux)

directory 생성
```
cd ./directory
git clone https://github.com/konyul/PatchModel.git
```
docker image 생성

```
docker pull kyparkk/mmsegmentation:latest
```

docker container 생성

```
docker run -it --gpus all --shm-size=8g -v ${오염 데이터 경로}:${원하는 경로} -w ${디렉토리 경로} --name ${container 명} kyparkk/mmsegmentation:latest /bin/bash
```

다음 명령어 입력

```
cd PatchModel
pip install -v -e.
```

훈련 모델 실행 (예시)

```
python3 tools/train.py configs/patchnet/patchnet_hyundae_512x512.py
```
