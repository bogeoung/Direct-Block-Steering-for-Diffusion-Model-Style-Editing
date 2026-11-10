# DBsteer 실행 및 재현 요구사항

이 저장소는 DBsteer의 핵심 과정만 포함한다. 즉, styled prompt와 neutral
prompt에서 transformer block의 raw update를 수집하고, 두 조건의 평균 차이를
하나의 direction으로 만든 뒤 SDXL denoising 과정에 주입한다.

## 이 저장소만으로 가능한 작업

다음 DBsteer 실험은 실행할 수 있다.

1. neutral/style prompt 쌍으로부터 raw block-update 통계 추출
2. block별 style--neutral difference-of-means direction 계산
3. 하나의 고정 direction을 모든 denoising step과 spatial position에 주입
4. 동일 seed의 unsteered base image와 DBsteer image 생성

핵심 실행 흐름은 다음과 같다.

```text
extract_direction.py
  -> photo.pt, <style>.pt
  -> generate_dbsteer.py
  -> DBsteer-generated images
```

## 별도로 준비해야 하는 항목

다음 항목은 이 저장소에 포함하지 않는다.

| 항목 | 필요 이유 | 준비 방법 |
| --- | --- | --- |
| Python 환경 | 코드 실행 | `requirements.txt`의 패키지 설치 |
| CUDA GPU와 PyTorch | SDXL 추론 | CUDA 환경에 맞는 PyTorch 설치 |
| SDXL base 1.0 가중치 | 이미지 생성 모델 | Hugging Face에서 별도 확보 |
| SDXL fp16-fix VAE | fp16 생성 안정화 | Hugging Face에서 별도 확보 |
| SDXL-Unbox | hook-enabled SDXL runtime | 별도 clone/install 후 경로 지정 |

SDXL-Unbox는 이 저장소에 포함되지 않는다. 설치 후 아래처럼 경로를 설정해야 한다.

```bash
export SDXL_UNBOX_PATH=/absolute/path/to/sdxl-unbox
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

SDXL-Unbox는 외부 공개 코드이며, 본 저장소는 해당 코드·SAE checkpoint·가중치를
배포하거나 저작권 대상으로 주장하지 않는다. 자세한 출처와 라이선스는
`THIRD_PARTY_NOTICES.md`를 참고한다.

## 포함하지 않은 기능

다음 기능은 저작권 등록용 clean source의 범위에서 의도적으로 제외했다.

- SAE 기반 steering recipe: `signed`, `dmsae`, `unbox`, `louvre`, `uniform`, `clamp`
- StyleAligned, style-prompt 등 외부 비교 방법
- SAE checkpoint 및 SAE latent feature 추출
- CSD, LPIPS, DINOv2, CLIP 기반 정량 평가
- 비교 표, Pareto plot, qualitative comparison 등 논문 figure 생성
- 생성 이미지, 추출 feature, metric CSV, 로그 및 모델 가중치

따라서 이 저장소는 DBsteer의 핵심 생성 파이프라인은 재현할 수 있지만,
논문의 전체 benchmark나 다른 방법과의 정량 비교를 단독으로 재현하지는 않는다.

## 최소 실행 예시

```bash
python scripts/extract_direction.py \
  --conditions photo watercolor \
  --output-dir outputs/directions

python scripts/generate_dbsteer.py \
  --style watercolor \
  --direction-dir outputs/directions \
  --output-dir outputs/generation \
  --alpha 0.5 \
  --save-base
```
