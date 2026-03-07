from pathlib import Path
import numpy as np
import data_loader as dl
import stat_analysis as sa
import linear_analysis as la
import logistic_curve_analysis as lca
import plot_utils as pu

def main():
    is_extra, is_total = False, True
    train_idx = [1, 3, 5]
    filepath = Path("data")

    if is_total:
        total_sbj_number = 8
        subject_list =  [1, 2, 4, 5, 7, 8, 10, 11]
    elif is_extra:
        total_sbj_number = 3
        subject_list =  [2, 4, 8]
        filepath = filepath/"extra"
    else:
        total_sbj_number = 5
        subject_list =  [1, 5, 7, 10, 11]
        filepath = filepath/"main"
       
    keys = {"sbj", "y", "distance", "diameter", "x", 
            "scaled_distance", "scaled_diameter", "scaled_x"}

    train = {k: [] for k in keys}
    test = {k: [] for k in keys}
    all_data = {k: [] for k in keys}

    for idx in subject_list:
        filename = filepath/f"result_sbj{idx}.txt" 
        input_file = dl.read_file(filename)
        _train, _test = dl.make_train_test_data_set(input_file, train_idx)
        _all_data = dl.make_all_data_set(input_file, train_idx)
        
        for key in _train:
            train[key].append(_train[key])
            test[key].append(_test[key])
            all_data[key].append(_all_data[key])
    

    for idx in range(len(all_data['sbj'])):
        all_distance = all_data["distance"][idx]
        all_diameter = all_data["diameter"][idx]
        train_distance = train['distance'][idx]
        train_diameter = train['diameter'][idx]
        test_distance = test['distance'][idx]
        test_diameter = test['diameter'][idx]

        scaler1 = sa.make_scaler(all_distance)
        all_data['scaled_distance'].append(scaler1.transform(all_distance))

        scaler2 = sa.make_scaler(all_diameter)
        all_data['scaled_diameter'].append(scaler2.transform(all_diameter))

        all_data['scaled_x'].append([[dist[0], dia[0]]
                                for dist, dia in zip(all_data['scaled_distance'][idx],
                                                     all_data['scaled_diameter'][idx])])

        scaler3 = sa.make_scaler(train_distance)
        train['scaled_distance'].append(scaler3.transform(train_distance))
        test['scaled_distance'].append(scaler3.transform(test_distance))

        scaler4 = sa.make_scaler(train_diameter)
        train['scaled_diameter'].append(scaler4.transform(train_diameter))   
        test['scaled_diameter'].append(scaler4.transform(test_diameter))
        
        train['scaled_x'].append([[dist[0], dia[0]]
                                   for dist, dia in zip(train['scaled_distance'][idx],
                                                        train['scaled_diameter'][idx])])
        test['scaled_x'].append([[dist[0], dia[0]]
                                   for dist, dia in zip(test['scaled_distance'][idx],
                                                        test['scaled_diameter'][idx])])

    
    train = {k: np.array(v) for k, v in train.items()}
    test = {k: np.array(v) for k, v in test.items()}
    all_data = {k: np.array(v) for k, v in all_data.items()}

    save_path = Path("results")
    if is_extra:
        save_path = save_path/"extra"
    elif not is_total:
        save_path = save_path/"main"

    is_save = True

    1
    print('run spearman')
    file_path = save_path/"spearman"

    for idx in range(len(all_data['sbj'])):
        sa.spearman_analysis(all_data['sbj'][idx], all_data['y'][idx], 
                         all_data['distance'][idx], all_data['diameter'][idx],
                         file_path, is_save)
    
    #2
    print('run scatter')
    file_path = save_path/"scatter"

    for idx in range(len(all_data['sbj'])):
        pu.plot_scatter(all_data['sbj'][idx], all_data['y'][idx], 
                        all_data['distance'][idx], all_data['diameter'][idx],
                        file_path, is_save)

    #Personal model
    #3   
    print('run linear regression')
    file_path = save_path/"personal_model"/"linear_model"/"raw_model"
    
    for idx in range(len(train['sbj'])):
        la.run_linear_regression(train['sbj'][idx], test['sbj'][idx],
                                 train['y'][idx], test['y'][idx],
                                 train['distance'][idx], test['distance'][idx],
                                 train['diameter'][idx], test['diameter'][idx],
                                 file_path, is_save)

    4
    print('run scaled linear regression')
    file_path = save_path/"personal_model"/"linear_model"/"scaled_model"

    for idx in range(len(train['sbj'])):
        la.run_linear_regression(train['sbj'][idx], test['sbj'][idx],
                                 train['y'][idx], test['y'][idx],
                                 train['scaled_distance'][idx], test['scaled_distance'][idx],
                                 train['scaled_diameter'][idx], test['scaled_diameter'][idx],
                                 file_path, is_save, is_scaled = True)

    #5.
    print('run scaled nonlinear regression')
    file_path = save_path/"personal_model"/"nonlinear_model"/"scaled_model"

    for idx in range(len(train['sbj'])):
        lca.run_logistic_curve(train['sbj'][idx], test['sbj'][idx],
                           train['y'][idx], test['y'][idx],
                           train['scaled_distance'][idx], test['scaled_distance'][idx],
                           train['scaled_diameter'][idx], test['scaled_diameter'][idx],
                           file_path, is_save, is_scaled = True)
    
    #6.
    print('run scaled multiple linear regression')
    file_path = save_path/"personal_model"/"multiple_linear_model"/"scaled_model"

    for idx in range(len(train['sbj'])):
        la.run_multiple_linear_regression(train['sbj'][idx], test['sbj'][idx],
                                          train['y'][idx], test['y'][idx],
                                          train['scaled_x'][idx], test['scaled_x'][idx],
                                          file_path, is_save, is_scaled = True)
    
    #General model
    #7.  
    print('run general scaled linear regression')
    file_path = save_path/"general_model"/"linear_model"/"scaled_model"

    la.run_linear_regression(train['sbj'].copy().reshape(-1,1), test['sbj'].copy().reshape(-1,1),
                             train['y'].copy().reshape(-1,1), test['y'].copy().reshape(-1,1),
                             train['scaled_distance'].copy().reshape(-1,1),
                             test['scaled_distance'].copy().reshape(-1,1),
                             train['scaled_diameter'].copy().reshape(-1,1),
                             test['scaled_diameter'].copy().reshape(-1,1),
                             file_path, is_save, is_scaled = True, is_general = True)

    #8.
    print('run general scaled nonlinear regression')
    file_path = save_path/"general_model"/"nonlinear_model"/"scaled_model"
    
    lca.run_logistic_curve(train['sbj'].copy().reshape(-1,1), test['sbj'].copy().reshape(-1,1),
                           train['y'].copy().reshape(-1,1), test['y'].copy().reshape(-1,1),
                           train['scaled_distance'].copy().reshape(-1,1),
                           test['scaled_distance'].copy().reshape(-1,1),
                           train['scaled_diameter'].copy().reshape(-1,1),
                           test['scaled_diameter'].copy().reshape(-1,1),
                           file_path, is_save, is_scaled = True, is_general = True)

    #9.
    print('run general scaled multiple linear regression')
    file_path = save_path/"general_model"/"multiple_linear_model"/"scaled_model"

    la.run_multiple_linear_regression(train['sbj'].copy().reshape(-1, 1),
                                      test['sbj'][idx].copy().reshape(-1, 1),
                                      train['y'].copy().reshape(-1, 1),
                                      test['y'].copy().reshape(-1, 1),
                                      train['scaled_x'].copy().reshape(-1, 2),
                                      test['scaled_x'].copy().reshape(-1, 2),
                                      file_path, is_save, is_scaled = True, is_general = True)

    print('complete!')

main()