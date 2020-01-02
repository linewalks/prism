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

## docker build

```
./build.sh
```

## docker run

```
./run.sh
```

## model development

```
./dev.sh
```

주최측의 샘플파일로 테스트할때는 dev.sh 에서 `-e DEV_DATA=dev` 제거하고 실행


## docker export

```
./zip.sh
```
