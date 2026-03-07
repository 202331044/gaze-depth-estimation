import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def create_p4_template_heatmap(image_path, save_path = None, is_save = False):
    dpi = 600
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    z = image

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Pixel Intensity')
    ax.set_title('3D Heatmap of p4 template')

    if is_save:
        save_path = save_path/"template_histogram"
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path/"3d_histogram.jpg"
        plt.savefig(filename, dpi = dpi)
    
    plt.show()
    plt.close(fig)

def plot_scatter(sbj_idx, y, distance, diameter, file_path = None, is_save = False):
    y_min, y_max, y_step = 15, 65, 5
    dpi = 600
    sbj = sbj_idx[0][0]

    #1
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(y_min, y_max, y_step))
    ax.set_title("DPI distance - depth fixation")
    ax.set_xlabel("DPI distance")
    ax.set_ylabel("depth fixation")
    ax.scatter(distance, y,  c = 'g', alpha = 0.5)

    if is_save:
        file_path.mkdir(parents=True, exist_ok=True)
        filename = file_path/f"sbj{sbj}_scatter_distance_depth.jpg"
        fig.savefig(filename, dpi = dpi)
        
    plt.close(fig)       

    #2
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(y_min, y_max, y_step))
    ax.set_title("pupil size - depth fixation")
    ax.set_xlabel("pupil size")
    ax.set_ylabel("depth fixation")       
    ax.scatter(diameter, y, c = 'g', alpha = 0.5)
        
    if(is_save):
        filename = file_path/f"sbj{sbj}_scatter_diameter_depth.jpg"
        fig.savefig(filename, dpi = dpi)
   
    plt.close(fig)
            
    #3
    fig, ax = plt.subplots()

    ax.set_title("DPI distance - pupil size")
    ax.set_xlabel("DPI distance")
    ax.set_ylabel("pupil size")
    ax.scatter(distance, diameter, c = 'g', alpha = 0.5)
        
    if is_save:
        filename = file_path/f"sbj{sbj}_scatter_distance_diameter.jpg"
        fig.savefig(filename, dpi = dpi)

    plt.close(fig)

def format_logistic_equation(L, x0, k):
    return f'$y = \\frac{{{L:.3f}}}{{1 + \\exp\\left({-k:.3f}(x - {x0:.3f})\\right)}}$'

def format_linear_equation(lr):
    intercept = lr.intercept_[0]
    coef = lr.coef_[0][0]

    if(intercept > 0):
        equation = f'y = {coef: .3f}x + {intercept: .3f}'
    else:
        equation = f'y = {coef: .3f}x - {-intercept: .3f}'

    return equation

def format_multiple_linear_equation(lr):
    coef = lr.coef_[0]
    intercept = round(lr.intercept_[0], 3)

    a, b, c = coef
    a = round(a, 3)
    b = round(b, 3)
    c = round(c, 3)

    x1 = f'{a} * x1'
    if b >= 0:
        x2 = f' + {b} * x2'
    else:
        x2 = f' {b} * x2'
    if c >= 0:
        x1x2 = f' + {c} * x1 * x2'
    else:
        x1x2 = f' {c} * x1 * x2'

    if intercept >= 0:
        d = f" + {intercept}"
    else:
        d = f" {intercept}"
    
    equation = "y = " + x1 + x2 + x1x2 + d

    return equation

def add_model_text(ax, txt, x=0.05, y=0.9):
    ax.text(x, y, txt, ha='left', va='top', transform=ax.transAxes)

def set_axis_scale(ax, x0, x1, xt0, xt1, xts,
                   y0, y1, yt0, yt1, yts,
                   z0 = None, z1 = None, zt0 = None, zt1 = None, zts = None ):

    if not None in (x0, x1):
        ax.set_xlim(x0, x1)

    if not None in (xt0, xt1, xts):
        ax.set_xticks(np.arange(xt0, xt1, xts))

    if not None in (y0, y1):
        ax.set_ylim(y0, y1)

    if not None in (yt0, yt1, yts):
        ax.set_yticks(np.arange(yt0, yt1, yts))
    
    if not None in (z0, z1):
        ax.set_zlim(z0, z1)

    if not None in (zt0, zt1, zts):
        ax.set_zticks(np.arange(zt0, zt1, zts))

def plot_model_fit_2d(ax, x, y, x2, y_pred, xlabel, ylabel, title, equation_text):
    ax.scatter(x, y, c='g', alpha=0.5)
    ax.plot(x2, y_pred)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    add_model_text(ax, equation_text)

def plot_model_fit_3d(poly_features, lr, ax, train_x, test_x, train_y, test_y):  
    ax.scatter(test_x[:, 0], test_x[:, 1], 
               test_y, color='blue', label='Actual Data', alpha = 0.5)

    X1_range = np.linspace(test_x[:, 0].min(), test_x[:, 0].max(), 10)
    X2_range = np.linspace(test_x[:, 1].min(), test_x[:, 1].max(), 10)  
    X1, X2 = np.meshgrid(X1_range, X2_range)
    X_poly_grid = poly_features.transform(np.c_[X1.ravel(), X2.ravel()])
        
    Y_pred = lr.predict(X_poly_grid).reshape(X1.shape)
    ax.plot_surface(X1, X2, Y_pred, color='red', alpha=0.5, label='Multiple Linear Regression Surface')
