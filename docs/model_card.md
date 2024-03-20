# Model Details

This RandomForestClassifier model is trained to predict if an individual's income exceeds $50K/year, leveraging scikit-learn. It aims to analyze factors affecting income levels.

## Intended Use

The model is intended to be used for predicting income levels based on demographic and employment-related features. It is created as part of a udacity nano-degree project and is not intended for real-world use.

## Training Data

Census dataset with features including age, workclass, education, and others is used as independent variables. The target variable is salary (<=50K, >50K).

## Evaluation Data

The model is evaluated on a 20% split test dataset, ensuring unbiased assessment on unseen data.

## Metrics

Model performance evaluated on precision, recall, and F-beta score for overall and feature-specific slices:

### Overall

Precision: 0.7186
Recall: 0.6344
F-beta: 0.6739

### Feature-specific Slices

**Workclass:**
0.0: Precision: 0.7209, Recall: 0.6370, F-beta: 0.6764
1.0: Precision: 0.6286, Recall: 0.5366, F-beta: 0.5789

**Education:**
0.0: Precision: 0.7186, Recall: 0.6308, F-beta: 0.6719
1.0: Precision: 0.7183, Recall: 0.7083, F-beta: 0.7133

**Marital-Status:**
0.0: Precision: 0.7187, Recall: 0.6277, F-beta: 0.6701
1.0: Precision: 0.7177, Recall: 0.7120, F-beta: 0.7149

**Occupation:**
0.0: Precision: 0.7186, Recall: 0.6344, F-beta: 0.6739
1.0: Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000

**Relationship:**
0.0: Precision: 0.7118, Recall: 0.6560, F-beta: 0.6827
1.0: Precision: 0.7227, Recall: 0.6224, F-beta: 0.6688

**Race:**
0.0: Precision: 0.7187, Recall: 0.6230, F-beta: 0.6674
1.0: Precision: 0.7179, Recall: 0.7925, F-beta: 0.7534

**Sex:**
0.0: Precision: 0.7161, Recall: 0.6474, F-beta: 0.6800
1.0: Precision: 0.7500, Recall: 0.5132, F-beta: 0.6094

**Native-Country:**
0.0: Precision: 0.7212, Recall: 0.6326, F-beta: 0.6740
1.0: Precision: 0.6667, Recall: 0.6769, F-beta: 0.6718

## Ethical Considerations

The inclusion of sensitive features such as race and sex necessitates careful consideration to avoid perpetuating biases. Users must critically assess and mitigate potential biases.

## Caveats and Recommendations

Utilize this model as part of a broader decision-making framework, supplemented by expert judgment.
