
if __name__ == "__main__":
    from data_loader import *
    from tester import *
    from filters import *

    data = load_data()
    tester = FilterTester(data)

    filters = {'Bayesian': BayesianFilter(),
               'Kalman': KalmanFilterImpl(),
                'EKF' : ExtendedKalman()
                }
    
    results = tester.eval_filters(filters)

    predictions = {}
    for name, filter_obj in filters.items():
        filter_obj.initialize(tester.train_data['Close'].values)
        preds = []
        for price in tester.test_data['Close'].values:
            preds.append(filter_obj.run(price)[0])
        predictions[name] = preds
    # Plot results
    tester.plot_results(predictions)   ## We can only see three lines in this study, since in our setting EKF is the same as Kalman.
