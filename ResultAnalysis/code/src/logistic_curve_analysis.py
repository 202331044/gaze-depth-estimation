from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import plot_utils as pu

class LogisticCurveRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, initial_guess=None):
        self.initial_guess = initial_guess
        self.params_ = None
    
    def logistic(self, x, L, x0, k):
        return L / (1 + np.exp(-k * (x - x0)))
    
    def fit(self, X, y):
        X = X.ravel()
        y = y.ravel()

        if self.initial_guess is None:
            initial_guess = [max(y), np.median(X), 1]
        else:
            initial_guess = self.initial_guess
        
        popt, pcov = curve_fit(self.logistic, X, y, p0=initial_guess,  maxfev=10000)
        self.params_ = popt

        return self
    
    def predict(self, X):
        if self.params_ is None:
            raise ValueError("No model fitting.")
        X = X.ravel()
        return self.logistic(X, *self.params_)

def logistci_curve(x, y):
    lr = LogisticCurveRegressor()
    lr.fit(x, y)
    return lr
    
def run_logistic_curve(train_sbj_idx, test_sbj_idx, train_y, test_y,
                       train_distance, test_distance,
                       train_diameter, test_diameter,
                       file_path = None, is_save = False, 
                       is_scaled = False, is_general = False):

    x_min, x_max = None, None
    x_tick_min, x_tick_max, x_tick_step = None, None, None
    y_min, y_max = 5, 65
    y_tick_min, y_tick_max, y_tick_step = 5, 65, 5
    z_min, z_max = None, None
    z_tick_min, z_tick_max, z_tick_step = None, None, None

    dpi = 600
    sbj = train_sbj_idx[0][0]
    graph_path = file_path/"graph"
    text_path = file_path/"text"

    if is_scaled:
        x_min, x_max = -0.1, 1.1
        x_tick_min, x_tick_max, x_tick_step = -0.1, 1.2, 0.2
        x1_range = np.linspace(0, 1, 100)
        x2_range = np.linspace(0, 1, 100)
        name = "general_scaled_nonlinear_model" if is_general else f"sbj{sbj}_scaled_nonlinear_model"
    else:
        name = "general_nonlinear_model" if is_general else f"sbj{sbj}_nonlinear_model"

    lm1 = logistci_curve(train_distance, train_y)
    lm2 = logistci_curve(train_diameter, train_y)
    lm3 = logistci_curve(train_distance, train_diameter)
                
    _y_pred1 = lm1.predict(x1_range)
    _y_pred2 = lm2.predict(x2_range)
    _y_pred3 = lm3.predict(x1_range)
        
    L_opt1, x0_opt1, k_opt1 = lm1.params_
    L_opt2, x0_opt2, k_opt2 = lm2.params_
    L_opt3, x0_opt3, k_opt3 = lm3.params_
        
    #1
    fig, ax = plt.subplots()

    pu.set_axis_scale(ax, x_min, x_max, x_tick_min, x_tick_max, x_tick_step,
                      y_min, y_max, y_tick_min, y_tick_max, y_tick_step,
                      z_min, z_max, z_tick_min, z_tick_max, z_tick_step)

    xlabel, ylabel, title = 'DPI distance', 'depth fixation', 'DPI distance - depth fixation'
    equation_text = pu.format_logistic_equation(L_opt1, x0_opt1, k_opt1)
    pu.plot_model_fit_2d(ax, test_distance, test_y, x1_range, _y_pred1,
                        xlabel, ylabel, title, equation_text)
        
    if is_save:
        graph_path.mkdir(parents=True, exist_ok=True)
        filename = graph_path/f"{name}_distance_depth.jpg"
        fig.savefig(filename, dpi = dpi)

    plt.close(fig)
        
    #2
    fig, ax = plt.subplots()

    pu.set_axis_scale(ax, x_min, x_max, x_tick_min, x_tick_max, x_tick_step,
                      y_min, y_max, y_tick_min, y_tick_max, y_tick_step,
                      z_min, z_max, z_tick_min, z_tick_max, z_tick_step)

    xlabel, ylabel, title = 'pupil size', 'depth fixation', 'pupil size - depth fixation'
    equation_text = pu.format_logistic_equation(L_opt2, x0_opt2, k_opt2)
    pu.plot_model_fit_2d(ax, test_diameter, test_y, x2_range, _y_pred2,
                         xlabel, ylabel, title, equation_text)
        
    if is_save:
        filename = graph_path/f"{name}_diameter_depth.jpg"
        fig.savefig(filename, dpi = dpi)

    plt.close(fig)
        
    #3
    fig, ax = plt.subplots()

    pu.set_axis_scale(ax, x_min, x_max, x_tick_min, x_tick_max, x_tick_step,
                      x_min, x_max, x_tick_min, x_tick_max, x_tick_step,
                      z_min, z_max, z_tick_min, z_tick_max, z_tick_step)

    xlabel, ylabel, title = 'DPI distance', 'pupil size', 'DPI distance - pupil size'
    equation_text = pu.format_logistic_equation(L_opt3, x0_opt3, k_opt3)
    pu.plot_model_fit_2d(ax, test_distance, test_diameter, x1_range, _y_pred3,
                        xlabel, ylabel, title, equation_text)    

    if is_save:
        filename = graph_path/f"{name}_distance_diameter.jpg"
        fig.savefig(filename, dpi = dpi)

    plt.close(fig)
    
    lm1_pred = lm1.predict(test_distance)
    lm2_pred = lm2.predict(test_diameter)
    lm3_pred = lm3.predict(test_distance)
        
    mse1 = mean_squared_error(test_y, lm1_pred) 
    mse2 = mean_squared_error(test_y, lm2_pred) 
    mse3 = mean_squared_error(test_diameter, lm3_pred) 
        
    r1 = r2_score(test_y, lm1_pred)
    r2 = r2_score(test_y, lm2_pred)
    r3 = r2_score(test_diameter,lm3_pred)
       
    data = pd.DataFrame({'DPI_depth':      [round(r1, 2), round(mse1, 2), round(np.sqrt(mse1), 2),
                                            round(L_opt1, 3), round(x0_opt1, 3), round(k_opt1, 3)],                                                
                        'diameter_depth':  [round(r2, 2), round(mse2, 2), round(np.sqrt(mse2), 2),
                                            round(L_opt2, 3), round(x0_opt2, 3), round(k_opt2, 3)],                                     
                        'DPI_diameter':    [round(r3, 2), round(mse3, 2), round(np.sqrt(mse3), 2),
                                            round(L_opt3, 3), round(x0_opt3, 3), round(k_opt3, 3)]},
                        index = ['r2', 'mse', 'rmse', 'L', 'x0', 'K'])
        
    if is_save:
        text_path.mkdir(parents=True, exist_ok=True)
        filename = text_path/f"{name}.txt"
        data.to_csv(filename, sep = '\t')