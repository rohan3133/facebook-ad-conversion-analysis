# Facebook Ad Conversion Analysis (R)

Classifying Facebook ads as **Low**, **Average** or **High** converting and identifying which ad features drive conversions, using Naive Bayes and Random Forest models.

Team project by **Sakshi Prabhu** and **Rohan Satishkumar**.

## Overview

| | |
|---|---|
| **Data** | 1,143 ads, 11 variables (`data/facebook.csv`) |
| **Target** | `Approved_Conversion` bucketed into Low (<= 2), Average (3-9), High (>= 10) |
| **Models** | Naive Bayes, Random Forest (11,430 trees), Random Forest with class-balanced sampling |
| **Evaluation** | Accuracy and Adjusted Rand Index on a held-out test set (65/35 split) |
| **Language** | R |

**Variables:** `ad_id`, `xyz_campaign_id`, `fb_campaign_id`, `age` (age bracket), `gender`, `interest` (interest category code), `Impressions`, `Clicks`, `Spent`, `Total_Conversion` (people who enquired after seeing the ad), `Approved_Conversion` (people who bought after seeing the ad).

## Approach

1. Explored the data and visualized the heavily skewed distribution of approved conversions (log transform applied).
2. Dropped ID columns and engineered **click-through rate (CTR = Clicks / Impressions)**.
3. Examined correlations between demographics, spend, reach, engagement and conversions.
4. Built the Low / Average / High target and trained Naive Bayes and Random Forest classifiers, plus a Random Forest with balanced class sampling to address the rarity of high-converting ads.
5. Compared models and ranked feature importance.

## Findings

- Spend, impressions and clicks were strongly associated with approved conversions; CTR added a moderate engagement signal.
- Age and gender showed little correlation with conversion, so behavioral and engagement metrics were more useful than static demographic targeting.
- Most ads convert very little, which motivates bucketing conversions into levels and balancing classes.
- The full write-up, model comparison and recommendations are in [`report/`](report).

## Repository structure

```
facebook-ad-conversion-analysis/
├── data/facebook.csv
├── scripts/facebook_ad_conversion.R
└── report/
    ├── Facebook_Ad_Conversion_Report.pdf
    └── Facebook_Ad_Conversion_Presentation.pdf
```

## How to run

1. Install R packages: `corrplot`, `naivebayes`, `randomForest`, `mclust`, `RColorBrewer`.
2. Set the working directory to the repository root and run:

```r
source("scripts/facebook_ad_conversion.R")
```

## Authors

Sakshi Prabhu and Rohan Satishkumar, MS Marketing Analytics, Illinois Institute of Technology
