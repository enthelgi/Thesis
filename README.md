

## Project Structure
```

Thesis/
├── data/   # Company's data, simulated data and data modification code
├── plasso/   # Python library for Plible Lasso (PL), updated and functional
├── plots/   # Code for generation of plots used in the thesis report 
├── r_code_fs_and_bt/   #r-code
│   ├── images
│   ├── bt_on_continuous.r  # Backtracking applied on simualted data
│   ├── fit_and_select_cli.r    # Same as fit and select but in the format to be used by pyhton stability analysis 
│   ├── fit_and_select.r    # Comparison of PL on company's data with all modifiers vs PL with stable modifiers vs logistic lasso  
│   ├── iFORM.R , iFORM_(2).R   #iFORM library code 
│   ├── iFORM_on_continuous.r   #iFORM applied on simulated data
│   ├── iFORM_on_tgs.r  #iFORM on company's data
│   └── mpl_stable_fit_tgs_and_logcompare.r #Comparison of PL on simulated data vs iFROM vs logistic lasso  
├── results
├── images
├── final_model.py  # L1 logistic regression on stable effects vs random forest, plus evaluation plots
├── get_median_methods_scores.py    # Computes median scores across methods
├── make_simulation_method_comparison_table.py  # Makes latex tables to use for methods comparison of methods on simulated data
├── make_tgs_selected_effects_frequency_table.py    # Makes latex tables for methods comparison of methods on company's data
├── pl_fit_predcv.py    # PL on simulated data (with nested cv)
├── stability_on_simulations.py  # Stability analysis on simulated data
├── stability_on_tgs_mplasso.py # Stability analysis on company's data (using fit_and_select_cli.r)
└── README.md

```
# General Notes

To use this code adust the data path to your local directory.