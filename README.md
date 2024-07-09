# Patch Model

## 설치방법 (Linux)

directory 생성
```
mkdir $[디렉토리명}
cd $[디렉토리명}
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

docker container 접속

```
docker exec -it ${컨테이너명} /bin/bash
```

다음 명령어 입력

```
cd PatchModel
pip install -v -e.
```

데이터 생성

```
mkdir -p data/hyundae
```

폴더 구조

```bash
data/hyundae
|— img_dir
|    |— train
|    |— val
|— ann_dir 
|    |— train
|    |— val
```

훈련 모델 실행 (예시)

```
python3 tools/train.py configs/patchnet/patchnet_hyundae_512x512.py
```
