# XAI Music Genre Classification — Results Summary

## Key Findings

### 1. Musicological Alignment
- Mean Spearman ρ = -0.360
- Best: hiphop
- Worst: country

### 2. Faithfulness
- Mean confidence drop at 50% removal: 0.664

### 3. Inter-method Agreement (SHAP vs LIME)
- Mean Pearson r: 0.097

### 4. Spurious Correlation
- Mean silence/signal attribution ratio: 0.766

## Conclusion
The model partially learns musicologically meaningful features but also
attends to recording artifacts, particularly in genres with older recordings.