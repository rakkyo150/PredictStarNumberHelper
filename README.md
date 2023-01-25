![Build](https://github.com/rakkyo150/PredictStarNumberHelper/actions/workflows/main.yml/badge.svg)

# PredictStarNumberHelper
This creates a learned model for [PredictStarNumber](https://github.com/rakkyo150/PredictStarNumber)

## Relevant Link

Training Data : https://github.com/rakkyo150/RankedMapData <br>
Model : https://github.com/rakkyo150/PredictStarNumberHelper <br>
Mod : https://github.com/rakkyo150/PredictStarNumberMod <br>
Chrome Extension : https://github.com/rakkyo150/PredictStarNumberExtension <br>

```mermaid
flowchart
    First(RankedMapData) -- Training Data --> Second(PredictStarNumberHelper)
    Second -- Learned Model --> Third(PredictStarNumber)
    Second -- Learned Model --> PredictStarNumberMod
    Third <-- REST API --> PredictStarNumberExtension
```

## Describe
![Describe](describe.png)

## Correlation Matrix
![Correlation Matrix](correlation.png)

## Model Evaluation
![Model Evaluation](modelEvaluation.png)
