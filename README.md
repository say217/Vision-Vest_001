# Vision-Vest
 


<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/75674691-01ce-4a5a-b96d-24fea2e7497c" alt="Image 1" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/7a754af8-aa44-4588-86ac-67a71fd99464" alt="Image 2" width="400"/></td>
  </tr>

</table>


The enhanced stock predictor is a sophisticated deep learning system designed for forecasting stock prices. It leverages a hybrid model combining Conv1D layers, bidirectional Gated Recurrent Units (BiGRU), and a scaled dot-product attention mechanism. This document provides a comprehensive, paragraph-based explanation of the workflow, emphasizing feature engineering, technical indicators, data processing, and the EnhancedGRU model’s architecture and functionality. The goal is to give a clear understanding of the system’s components, their roles, and how they contribute to robust stock price predictions, without relying on mathematical equations.

---

## Workflow Overview

The workflow starts with the user entering a stock ticker symbol, such as AAPL or TSLA, with AAPL as the default if no input is provided. Historical stock data is retrieved from Yahoo Finance using the `yfinance` library, spanning from January 1, 2022, to the current date, August 27, 2025. To optimize performance, the data is cached locally in a pickle file, allowing quick access for subsequent runs without re-downloading. 

The raw data undergoes extensive preprocessing, including the addition of various technical indicators to capture market patterns, handling missing or invalid values, and normalizing the data to a consistent scale. The processed data is then organized into 40-day sequences, split into 80% for training and 20% for testing, and loaded into PyTorch DataLoaders for efficient batch processing. The EnhancedGRU model is trained using the Adam optimizer, Huber loss for robust error handling, mixed precision for faster GPU computation, gradient clipping to stabilize training, and early stopping to prevent overfitting. After training, the model evaluates performance on the test set using metrics such as root mean squared error (RMSE), mean absolute error (MAE), R-squared (R²), and directional accuracy. It also generates recursive forecasts for the next four days by default, feeding predicted values back into the model. Results are visualized through plots showing historical prices, predictions, and forecast confidence intervals, along with feature distribution histograms and correlation heatmaps for deeper insights.


# Diagram Workflow
```mermaid
graph TD
    A[User Input: Ticker Symbol] --> B[DataHandler]
    
    subgraph Data Acquisition
        B -->|yfinance| C[Download Data]
        C -->|Cache| D[Local .pkl File]
        C --> E[Raw OHLCV Data]
    end
    
    subgraph Preprocessing
        E --> F[Add Technical Indicators]
        F -->|SMA_10, RSI, MACD, etc.| G[Feature Enrichment]
        G --> H[Handle Missing Values]
        H -->|Forward Fill, Mean Imputation| I[Scaled Features]
        I -->|MinMaxScaler| J[Sequence Creation]
        J -->|40-day Windows| K[Train/Test Split]
        K --> L[PyTorch DataLoader]
    end
    
    subgraph Model Architecture: EnhancedGRU
        L --> M[Conv1D Layers]
        M -->|Kernel 3, 5; ReLU| N[Concatenated Features]
        N --> O[Bidirectional GRU]
        O -->|Hidden Size 128| P[Attention Mechanism]
        P -->|Scaled Dot-Product| Q[LayerNorm + Dropout]
        Q -->|0.15 Dropout| R[Linear Output]
        R --> S[Predicted Close Price]
    end
    
    subgraph Training
        L --> T[Trainer]
        T -->|Adam, Huber Loss| U[Training Loop]
        U -->|Mixed Precision, Grad Clipping| V[Early Stopping]
        V -->|Save Best Model| W[Model Checkpoint]
    end
    
    subgraph Evaluation
        W --> X[Evaluate on Test Set]
        X -->|RMSE, MAE, R², Directional Acc| Y[Performance Metrics]
    end
    
    subgraph Forecasting
        W --> Z[Recursive Prediction]
        Z -->|Last 40-day Window| AA[4-Day Forecast]
        AA -->|Inverse Transform| AB[Future Prices]
    end
    
    subgraph Visualization
        Y --> AC[Prediction Plot]
        AB --> AD[Forecast Plot with Confidence]
        I --> AE[Feature Histograms]
        I --> AF[Correlation Heatmap]
        AC -->|Dark Theme| AG[Matplotlib/Seaborn]
        AD --> AG
        AE --> AG
        AF --> AG
    end
    
    style A fill:#1c2526,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style B fill:#2c3e50,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style C fill:#2c3e50,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style D fill:#2c3e50,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style E fill:#2c3e50,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style F fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style G fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style H fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style I fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style J fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style K fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style L fill:#34495e,stroke:#00f7ff,stroke-width:2px,color:#ffffff
    style M fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style N fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style O fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style P fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style Q fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style R fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style S fill:#4b6584,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style T fill:#636e72,stroke:#ffff00,stroke-width:2px,color:#ffffff
    style U fill:#636e72,stroke:#ffff00,stroke-width:2px,color:#ffffff
    style V fill:#636e72,stroke:#ffff00,stroke-width:2px,color:#ffffff
    style W fill:#636e72,stroke:#ffff00,stroke-width:2px,color:#ffffff
    style X fill:#2d3436,stroke:#00ffab,stroke-width:2px,color:#ffffff
    style Y fill:#2d3436,stroke:#00ffab,stroke-width:2px,color:#ffffff
    style Z fill:#2d3436,stroke:#ff6f61,stroke-width:2px,color:#ffffff
    style AA fill:#2d3436,stroke:#ff6f61,stroke-width:2px,color:#ffffff
    style AB fill:#2d3436,stroke:#ff6f61,stroke-width:2px,color:#ffffff
    style AC fill:#353b48,stroke:#ffd700,stroke-width:2px,color:#ffffff
    style AD fill:#353b48,stroke:#ffd700,stroke-width:2px,color:#ffffff
    style AE fill:#353b48,stroke:#ffd700,stroke-width:2px,color:#ffffff
    style AF fill:#353b48,stroke:#ffd700,stroke-width:2px,color:#ffffff
    style AG fill:#353b48,stroke:#ffd700,stroke-width:2px,color:#ffffff
```



