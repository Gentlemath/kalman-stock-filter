import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from filters import *

class FilterTester:
    def __init__(self, data: pd.DataFrame, train_test_split = 0.8):
        self.data = data
        self.train_size = int(len(data) * train_test_split)
        self.train_data = data[:self.train_size]
        self.test_data = data[self.train_size:]

    def _calculate_matrics(self, predictions: np.array, metrics: List[str]) -> Dict[str, float]:
        actual = self.test_data["Close"].values
        error = actual - predictions
        metrics_value = {}
        if 'mse' in metrics:
            metrics_value['mes'] = np.mean(error**2)
        if 'mae' in metrics:
            metrics_value['mae'] = np.mean(np.abs(error))
        if 'rmse' in metrics:
            metrics_value['rmes'] = np.sqrt(metrics_value.get('mse', np.mean(error**2)))
        return metrics_value
    
    def eval_filters(self, filters: Dict[str, BaseFilter],
                     metrics: List[str] = ['mes', 'mae', 'rmes']) -> pd.DataFrame:
                    # Dict[str, "BaseFilter"] (string) when: BaseFilter is defined later in the file, 
                    # or there are circular imports
                    # or you're inside a TYPE_CHECKING block and want to avoid runtime errors.

        results = []
        for name, filter_obj in filters.items():
            print(f"Testing {name} ...")
            predictions = self._test_filter(filter_obj)
            eval_metrics = self._calculate_matrics(predictions, metrics)
            results.append({'filter': name, **eval_metrics})

        return pd.DataFrame(results)
    
    def _test_filter(self, filter_obj: "BaseFilter") -> np.array:
        filter_obj.initialize(self.train_data['Close'].values)

        predictions = filter_obj.run(self.test_data['Close'].values)

        return np.array(predictions)

    def plot_results(self, predictions: Dict[str, np.ndarray]):
        """Plot actual vs predicted prices for each filter"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.test_data.index, self.test_data['Close'], label='Actual', linewidth=2)  
        
        for name, pred in predictions.items():
            plt.plot(self.test_data.index, pred, '--', label=name)
            
        plt.title('Stock Price Prediction Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
