# Spotify Mood Recommender

- LinkedIn: Huy Doan - https://www.linkedin.com/in/huy-doan-407b10355/

## About The Project

Welcome to the Spotify Mood & Trend-Based Recommendation App! 🎵

This project is a music recommendation system that infers song mood from audio features and recommends tracks based on emotional relevance, popularity, and long-term trends. It uses a trained scikit-learn pipeline for mood inference and time-series chart data to surface trending songs.

## Built With

- pandas, numpy
- scikit-learn, lightgbm
- streamlit

## Getting Started

### Installation Steps

1) Create and activate a virtual environment.
2) Install dependencies:

```
pip install -r requirements.txt
```

3) Ensure artifacts and data are available:

- Models: `artifacts/models/<model>/`
- Dataset B: `data/processed/filtered_countries.csv`

### Run The UI (no FastAPI)

```
python -m streamlit run app/app.py
```

Optional env vars:
- `MODEL_ROOT` (default: `artifacts/models`)
- `DATA_B_PATH` (default: `data/processed/filtered_countries.csv`)
- `DF_LATEST_CACHE_DIR` (default: `data/processed`)

First run may be slow because it builds cached `df_latest_<model>.csv`.

### Usage (API & Web App)

Hiện repo chưa có API HTTP sẵn (không có FastAPI/Flask server). Web App chạy bằng Streamlit: chọn `Model`, thử dự đoán mood ở phần **Model playground** (nhập các audio features), và dùng phần **Recommend** để lấy Top K gợi ý theo `mood_idx`, `country`, `mode`, `min_conf`, v.v. Nếu bạn muốn tích hợp API, có thể bọc các hàm nội bộ như `predict_proba_df` (trong `src.models.registry`) và `recommend_df` (trong `src.recommender.recommend`) thành endpoint REST theo nhu cầu.

## Training (optional)

```
python -m src.engine.train --model all --run-id mood_5models
```

## Contributing

Contributions are welcome. Open an issue or submit a PR.

## License

Add your license here.

## Contact

<your-email>
