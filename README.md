# prism

## clone repository

```
git clone https://github.com/hurcy/prism
```

## validation data

```
cp condition_occurrence.csv data/train
cp measurement.csv data/train
cp person.csv data/train
```
## run

필요한 라이브러리 설치 (최초 한번만)
```
pip install -r docker/src/requirements.txt
```

로컬 환경에서 모델 실행
```
python docker/src/train.py ./data/
```

## docker preparation

```
./build.sh; ./zip.sh
```

이후 생성되는 압축파일을 업로드하면 됩니다.
