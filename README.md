# Road Surface Explorer

A Streamlit research demonstration by team SE-2403: Alua Sagyndyk, Shugyla Akhit and Bekbolat Yergaliuly.

The app uses the genuine EfficientNetB0 epoch-9 checkpoint selected by validation macro-F1. It predicts five Roads v2 appearance labels: Dry, Wet, Snow, Ice and Slush. Upload a JPG/PNG and select a road region.

## Run

Use Python 3.12, install requirements.txt, then run:

```sh
python -m streamlit run app.py
```

For Streamlit Community Cloud, choose this repository, branch main, entrypoint app.py and Python 3.12. The model path resolves relative to app.py and requires no script arguments or API keys.

## Measured study and limitations

On the frozen 1,274-image Roads test subset, the source checkpoint achieved 97.88% accuracy and 0.9792 macro-F1. Validation selected the application model before test evaluation. Original scene/day independence and geographic transfer remain unverified. These measurements do not establish accuracy on arbitrary uploaded internet photographs. Scores are uncalibrated and do not measure tyre grip or accident risk.

Dataset: Roads v2, Roboflow Universe, listed CC BY 4.0: https://universe.roboflow.com/roads-pjwmf/roads-q0nh0/dataset/2 . Original capture provenance is not independently verified. Complete study materials are maintained separately from this deployment repository.

## Deployment export

The included model.keras stores the original learned model weights without optimizer state. deployment_receipt.json records the parent checkpoint hash, all-weight identity check and exact prediction parity on five validation examples. The full research archive retains the original training checkpoint and reported measurements.
