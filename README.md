# 🔥 Prometheus - Fire Predictive Mapping

![Prometheus Banner](./images/prometheus_banner.png) <!-- Replace with your actual image path -->

A collaborative project by **Jacopo Minniti**, **Marina Levay**, and **Anhelina Nevmerzhytska**.

---

## 🧭 Overview

**Prometheus** is a machine learning project designed to assist firefighters and national agencies in **predicting the spread of wildfires**. By leveraging **satellite imagery**, **weather data**, and **terrain information**, our goal is to generate **probability maps** indicating where fires are likely to spread next.

---

## 🧰 Repository Structure & Tools

All scripts are meant to be run sequentially, as they build on each other.

### Data Collection

| File | Description |
|------|-------------|
| `1_fire_features_api.ipynb` | Collects wildfire events using the **NASA FIRMS API**, bounded by coordinates. |
| `1_weather_features_api.py` | Gathers relevant **climate features**, such as wind speed, direction, and vegetation index. |

---

### Dataset Preparation (PyTorch)

| File | Description |
|------|-------------|
| `2_cluster_generator.py` | Groups fire events into **clusters** of nearby fire spots. |
| `3_tensor_generator.py` | Converts clustered data into **PyTorch tensors** for model input/output. |

---

### Model Training

| File | Description |
|------|-------------|
| `4_model.py` | Trains a **ConvLSTM model** using custom hyperparameters and epochs. |

---

### Prediction & Evaluation

| File | Description |
|------|-------------|
| `5_visualize.py` | Generates **accuracy metrics** and plots for evaluating model performance. |
| `visualize_fire_events.py` | Maps **clustered fire events** within a selected geographical boundary. |

---

## 📊 Sample Output

### Clustered Fire Events

---

## 🚀 Future Work

- Testing with a new model architecture, Attention Swin U-Net, based on literature review
- Balancing the tradeoffs of sourcing a dataset from longer time periods and geographical regions while maintaining high feature resolution to avoid sparse data. 

---

## 📄 License

[MIT License](./LICENSE)

---

## 🤝 Acknowledgments

**NASA FIRMS**, **NOAA**, and **Copernicus** for providing open-source access to environmental data used in this project.

**Minerva University** for initiating the AI Sustainability Lab. 
