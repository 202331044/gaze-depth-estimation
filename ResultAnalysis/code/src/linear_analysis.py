from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu

def linear_regression(x, y):
    lr = LinearRegression()
    lr.fit(x, y)
    return lr
    
def run_linear_regression(train_sbj_idx, test_sbj_idx, train_y, test_y,
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
        name = "general_scaled_linear_model" if is_general else f"sbj{sbj}_scaled_linear_model"
    else:
        name = "general_linear_model" if is_general else f"sbj{sbj}_linear_model"

    lr1 = linear_regression(train_distance, train_y)
    lr2 = linear_regression(train_diameter, train_y)
    lr3 = linear_regression(train_distance, train_diameter)

    lr1_pred = lr1.predict(test_distance)
    lr2_pred = lr2.predict(test_diameter)
    lr3_pred = lr3.predict(test_distance)
        
    #1
    fig, ax = plt.subplots()
 
    pu.set_axis_scale(ax, x_min, x_max, x_tick_min, x_tick_max, x_tick_step,
                        y_min, y_max, y_tick_min, y_tick_max, y_tick_step,
                        z_min, z_max, z_tick_min, z_tick_max, z_tick_step)        
    xlabel, ylabel, title = "DPI distance", "depth fixation", "DPI distance - depth fixation"
    equation_text = pu.format_linear_equation(lr1)
    pu.plot_model_fit_2d(ax, test_distance, test_y, test_distance, lr1_pred,
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

    xlabel, ylabel, title = "pupil size", "depth fixation", "pupil size - depth fixation"
    equation_text = pu.format_linear_equation(lr2)  
    pu.plot_model_fit_2d(ax, test_diameter, test_y, test_diameter, lr2_pred,
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

    xlabel, ylabel, title =  "DPI distance", "pupil size", "DPI distance - pupil size"
    equation_text = pu.format_linear_equation(lr3)  
    pu.plot_model_fit_2d(ax, test_distance, test_diameter, test_distance, lr3_pred,
                        xlabel, ylabel, title, equation_text)

    if is_save:
        filename = graph_path/f"{name}_distance_diameter.jpg" 
        fig.savefig(filename, dpi = dpi)

    plt.close(fig)

    #MSE
    mse1 = mean_squared_error(test_y, lr1_pred) 
    mse2 = mean_squared_error(test_y, lr2_pred) 
    mse3 = mean_squared_error(test_diameter, lr3_pred) 
        
    #r2 score
    r1 = r2_score(test_y, lr1_pred)
    r2 = r2_score(test_y, lr2_pred)
    r3 = r2_score(test_diameter, lr3_pred)
        
    data = pd.DataFrame({"DPI_depth":       [round(r1, 2), round(mse1, 2), round(np.sqrt(mse1), 2)],
                         "diameter_depth":  [round(r2, 2), round(mse2, 2), round(np.sqrt(mse2), 2)],
                         "DPI_diameter":    [round(r3, 2), round(mse3, 2), round(np.sqrt(mse3), 2)]},
                          index = ['r2', 'mse', 'rmse'])
    if is_save:
        text_path.mkdir(parents=True, exist_ok=True)
        filename = text_path/f"{name}.txt" 
        data.to_csv(filename, sep = '\t')

                
#다중 선형 회귀
def run_multiple_linear_regression (train_sbj_idx, test_sbj_idx, train_y, test_y, 
                                   train_x, test_x,
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
    title = "General Multiple Linear Regression" if is_general else "Multiple Linear Regression"
    
    if is_scaled:
        x_min, x_max = -0.1, 1.1  
        x_tick_min, x_tick_max, x_tick_step = -0.1, 1.2, 0.2
        y_min, y_max = -0.1, 1.1
        y_tick_min, y_tick_max, y_tick_step = -0.1, 1.2, 0.2
        z_min, z_max = -10, 70
        name = "general_scaled_multiple_linear_model" if is_general else f"sbj{sbj}_scaled_multiple_linear_model"    
    else:
        name = "general_multiple_linear_model" if is_general else f"sbj{sbj}_multiple_linear_model"
    
    poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias= False)
    X_train_poly = poly_features.fit_transform(train_x)
    X_test_poly = poly_features.transform(test_x)
    lr = linear_regression(X_train_poly, train_y)
    lr_pred = lr.predict(X_test_poly)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
    pu.set_axis_scale(ax, x_min, x_max, x_tick_min, x_tick_max, x_tick_step,
                      y_min, y_max, y_tick_min, y_tick_max, y_tick_step,
                      z_min, z_max, z_tick_min, z_tick_max, z_tick_step)

    ax.set_xlabel('DPI distance')
    ax.set_ylabel('pupil size')
    ax.set_zlabel('depth fixation')
    ax.set_title(title) 
    pu.plot_model_fit_3d(poly_features, lr, ax, train_x, test_x, train_y, test_y)

    equation = pu.format_multiple_linear_equation(lr)
        
    if is_save:
        graph_path.mkdir(parents=True, exist_ok=True)
        filename = graph_path/f"{name}.jpg"
        fig.savefig(filename, dpi = dpi)

        text_path.mkdir(parents=True, exist_ok=True)
        filename = text_path/f"{name}.txt" 
            
        data = pd.DataFrame({title: 
                                [round(r2_score(test_y, lr_pred),2),
                                round(mean_squared_error(test_y, lr_pred),2),
                                round(np.sqrt(mean_squared_error(test_y, lr_pred)),2),
                                equation]},
                                index = ['r2', 'mse', 'rmse','equation'])

        data.to_csv(filename, sep = '\t')

        if is_general:                 
            ape = np.abs((lr_pred - test_y) / test_y) * 100
            mape = np.mean(ape)
            mape_std = np.std(ape)
            
            data2 = pd.DataFrame({"general multiple linear regression error": 
                                    [round(mape,2),
                                    round(mape_std, 2), 
                                    round(np.sqrt(mean_squared_error(test_y, lr_pred)), 2)]},
                                    index = ['mape', 'mape std', 'rmse'])
                
            filename = text_path/f"{name}_mape.txt" 
            data2.to_csv(filename, sep = '\t')

    plt.close(fig)