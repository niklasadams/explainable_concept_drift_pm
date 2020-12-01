import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf
def granger_causality(primary_features, secondary_features, secondary_feature_names, change_points_primary, change_points_secondary, p_value = 0.01):
    '''Given change points and representations of two perspectives, this function calculates the cause-effect relationships.
    A reduced primary and an unreduced secondary perspective along with the
    names of the secondary features have to be provided. This function filters
    the change points of the secondary perspective that precede a change point
    in the primary perspective and test, whether there are granger causal features,
    given the lag between drifts
    args:
        primary_perspective_reduced: Reduced time series retrieved by the previously
          executed dimensionality reduction.
        secondary_features: Feature representation for the secondary perspective.
          Retrieved when constructing the feature representation for the secondary perspective.
        secondary_feature_names: List of the feature names, that is retrieved when
          constructing the features.
        change_points_primary: List of primary change points
        change_points_secondary: List of secondary change points
        p_value: Maximum p-value
    '''
    tmp = np.array(primary_features)
    transpose = tmp.T
    primary_features = transpose.tolist()
    
    tmp = np.array(secondary_features)
    transpose = tmp.T
    secondary_features = transpose.tolist()
    
    
    results = []
    if not isinstance(primary_features[0],list):
        primary_features = [primary_features]
    for cp_1 in change_points_primary:
        for cp_2 in change_points_secondary:
            if cp_2 < cp_1:
                k = cp_1-cp_2
                feature_set = {}
                p = p_value
                for i in range(0,len(secondary_features)):
                    f = secondary_features[i]
                    for f_2 in primary_features:
                        granger_data = pd.DataFrame(f_2)
                        granger_data[secondary_feature_names[i]] = f
                        granger_data = granger_data.dropna()
                        try:
                            gc_res = grangercausalitytests(granger_data, [k], verbose = False)
                            #Increase the margin by 1% (or even less) to account for numeric approximation errors
                            if gc_res[k][0]['params_ftest'][1] < p*1.01:
                                p_feat = primary_features.index(f_2)
                                if p_feat not in feature_set.keys():
                                    feature_set[p_feat] = []
                                if secondary_feature_names[i] not in feature_set[p_feat]:
                                    feature_set[p_feat].append(secondary_feature_names[i])  
                        except ValueError:
                            pass
                results.append((cp_1,cp_2,feature_set,p))
    return results

def draw_ca(res, primary, p_names, secondary, s_names, store_path=""):
    '''Draws the cause-effect relaitonships, if wanted also to a pdf file.
    Parameters
    ----------
    res : list
        Output of cause-effect analysis.
    primary : 
       primary perspective features from time series construction.
    p_names : 
        primary names form time series construction.
    secondary : 
        As for primary
    s_names : 
        As for primary
    store_path : string, optional
        If provided, the produced plots will be stored as a pdf in the provided
        file. The default is "".

    Returns
    -------
    None.

    '''
    tmp = np.array(primary)
    transpose = tmp.T
    primary= transpose.tolist()
    
    tmp = np.array(secondary)
    transpose = tmp.T
    secondary= transpose.tolist()
    if not store_path == "": 
        pdf = matplotlib.backends.backend_pdf.PdfPages(store_path)
    for ca in res:
        if not len(ca[2].keys()) == 0:
            for feature in ca[2].keys():
                fig = plt.figure(figsize=(15, 8))
                outer = gridspec.GridSpec(1, 3, wspace=0.1, hspace=2.5, width_ratios=[3, 1,2])
                causes = [secondary[s_names.index(name)] for name in ca[2][feature]]
                name_causes = ca[2][feature]
                inner_left = gridspec.GridSpecFromSubplotSpec(3, len(name_causes)//3+1,
                    subplot_spec=outer[0], wspace=0.45, hspace=0.5)
                
                cause_df = pd.DataFrame(np.array(causes).T.tolist(), columns = name_causes)
                for i in range(0,len(name_causes)):
                    ax = plt.Subplot(fig, inner_left[i])
                    cause_df[cause_df.columns[i]].plot(color='black', ax=ax)
                    ax.set_title(cause_df.columns[i])
                    ax.axvline(x=ca[1],linewidth=1, color='r')
                    ax.set_xlabel("days")
                    fig.add_subplot(ax)
                
                inner_middle = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[1], wspace=0, hspace=0)  
                ax = plt.Subplot(fig, inner_middle[1])
                ax.arrow(0.1, 0.5, 0.6, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                ax.text(0.05,0.6,"Granger-causal with lag "+str(ca[0]-ca[1]))
                ax.set_axis_off()
                fig.add_subplot(ax)
                
                
                inner_right = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[2], wspace=0.2, hspace=0.4)  
                effect = primary[feature]   
                name_effect = p_names[feature]
                effect_df = pd.DataFrame(np.array(effect).T.tolist(), columns = [name_effect])
                ax = plt.Subplot(fig, inner_right[1])
                effect_df[effect_df.columns[0]].plot(color='black', ax=ax)
                ax.set_title(effect_df.columns[0])
                ax.set_xlabel("days")
                ax.axvline(x=ca[0],linewidth=1, color='r')
                fig.add_subplot(ax)
                if not store_path == "": 
                    pdf.savefig(fig)
                fig.show()
    if not store_path == "": 
        pdf.close()
        print("Output shown in "+str(store_path))
               
                
               
                
               