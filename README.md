# PISA

## Introduction
This repository is the reporisity of **Prompt-Based Time Series Forecasting: A New Task and Dataset** (currently under submission). PISA is a large-scale dataset including three real-world forecasting scenarios (three sub-sets) with 311,932 data instances in total. It is designed to support and facilitate the novel PromptCast task proposed in the paper. 

## Updates

***2022/06/10***
This repo is open for review.



## Numerical Time Series Forecasting vs. PromptCast
![](resources/concept.png)

> Exisiting numerical-based forecasting VS. Prompt-based forecasting

### ***PromptCast Evaluation Metrics***
- RMSE
- MAE
- Missing Rate: whether the numerical forecasting target can be decoded (via string parsing) from the generated output prompts.


## PISA Dataset
### ***Forecasting Scenarios***
The proposed PISA dataset contrains three real-world forecasting scenarios:
- CT: city temperature forecasting
- ECL: electricity consumption forecasting
- SG: humana mobility visitor flow forecasting

![](resources/statistics.png)

> Details of three sub-sets
> 
<br></br>

### ***Folder Structure (see [Dataset](Dataset/README.md))***
```text
Dataset
|── PISA-Prompt
    │── CT
        │-- train_x_prompt.txt
        │-- train_y_prompt.txt
        │-- val_x_prompt.txt
        │-- val_y_prompt.txt
        │-- test_x_prompt.txt
        │-- test_y_prompt.txt
    │── ECL
        │-- train_x_prompt.txt
        │-- train_y_prompt.txt
        │-- val_x_prompt.txt
        │-- val_y_prompt.txt
        │-- test_x_prompt.txt
        │-- test_y_prompt.txt  
    │── SG
        │-- train_x_prompt.txt
        │-- train_y_prompt.txt
        │-- val_x_prompt.txt
        │-- val_y_prompt.txt
        │-- test_x_prompt.txt
        │-- test_y_prompt.txt   
```

## Benchmark Results
Please check [Benchmark](Benchmark/README.md) folder for the implementations of benchmarked methods.
<br></br>

![](resources/result_1.png)

> RMSE and MAE performance
> 
<br></br>

![](resources/result_2.png)

> Missing Rate results
> 
<br></br>

![](resources/result_3.png)

> Results under train-from-scratch and cross-scenario zero-shot settings
> 
<br></br>

## RoadMap
- [x] GitHub repo open for reviewing  
- [ ] Paper release
- [ ] Full dataset release in this repo
- [ ] Full dataset release in HuggingFace Dataset page 
- [ ] Leaderboard Website Launch
- [ ]  ...
- [ ]  ...
