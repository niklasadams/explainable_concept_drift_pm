from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from datetime import timedelta
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.dfg.utils import dfg_utils
#from pm4py.algo.discovery.dfg.versions import native as dfg_inst
from pm4py.algo.discovery.alpha.data_structures import alpha_classic_abstraction
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py import util as pm_util
from pm4py.objects.log.util import interval_lifecycle
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
import numbers
from sklearn.decomposition import PCA
import numpy as np
def subdivide_log(path,start,end,windowsize):
    '''Subidivides a log for the given time interval and the fiven sublog size
    args:
        path: path to the log
        start_str: String of the starting date
        end_str: String of the end date
        start: Datetime of the starting date
        end: Datetime of the end date
        windowsize: Windowsize/sublogsize in days
    returns:
        list of sublogs
    
    '''
    log = xes_importer.apply(path)
    logs = []
    #Calculate the number of windows
    n = (end - start).days//(windowsize)
    for i in range(0,n):
        logs.append(timestamp_filter.apply_events(log, (start+ timedelta(days=i*windowsize)).strftime("%Y-%m-%d %H:%M:%S"), (start+ timedelta(days=(i+1)*windowsize)).strftime("%Y-%m-%d %H:%M:%S")))
    return logs

def apply_feature_extraction(logs, features):
    '''Applies the feature extraction for a given feature set
    args:
        logs: List of sublogs from timeframe splitting.
        features: list of strings specifying the desired features.
        attribute_set: Can be specified, if all_numeric_data is chosen.
          The attributes in attribute_set will be EXCLUDED and not added to
          the features.
    returns:
        a list of feature_names and a list of feature time series, both in the same order
    '''
    print(len(logs))
    feature_vectors = [[] for i in range(0,len(logs))]
    for feature in features:
        results = []
        if feature == "event_count":
            results = extract_number_of_events(logs)
        if feature == "trace_count":
            results = extract_number_of_traces(logs)
        if feature == "distinct_event_count":
            results = extract_number_of_distinct_events(logs)
        if feature == "direct_follows_relations":
            results = extract_direct_follows_relationships(logs)
        if feature == "direct_follows_relations_count":
            results = extract_number_of_direct_follows_relationships(logs)
        if feature == "performance_direct_follow_relations":
            results = extract_performance_of_direct_follows_relationships(logs)
        if feature == "all_numeric_data":
            results = mine_all_data(logs)
        #if feature == "alpha":
        #    results = extract_alpha(logs)
        if feature == "duration":
            results = extract_event_durations(logs)
        if feature == 'workload':
            results = extract_workload(logs)
        if feature == 'overtime':
            results = extract_overtime(logs)
        if feature == 'heuristics':
            results = extract_heuristics(logs)
        #more features
        for i in range(0,len(results)):
            for result in results[i]:
                feature_vectors[i].append(result)
    #set non existent features to zero
    feature_names__ = []
    for i in range(0,len(logs)):
        for j in range(0,len(feature_vectors[i])):
            feature_names__.append(feature_vectors[i][j][0])
            

    feature_names = list(set(feature_names__))
    
    feature_lists = []
    for i in range(0,len(feature_names)):
        feature_list = []
        for j in range(0,len(logs)):
            existing_features = [k[0] for k in feature_vectors[j]]
            if feature_names[i] in existing_features:
                #find index
                idx = existing_features.index(feature_names[i])
                feature_list.append(feature_vectors[j][idx][1])
            else:
                feature_list.append(0)
        feature_lists.append(feature_list)
    
    
    return feature_names, np.asarray(feature_lists).transpose()

def get_all_attributes(logs):
    '''Returns all attributes of a log, split into numerical and non-numerical
    '''
    activities = []
    not_int_activities = []
    for log in logs:
            #alle Activities mit attributes mit zahlendata herausfinden
            for trace in range(0,len(log)):
                for event in range(0,len(log[trace])):
                    for attribute_key in log[trace][event]:
                        #if isinstance(log[trace][event][attribute_key], numbers.Number):
                        if not attribute_key == "case:concept:name":
                            tmp = (log[trace][event]['concept:name'], attribute_key)
                            if check_int(log[trace][event][attribute_key]):
                                
                                if not tmp in activities:
                                    activities.append(tmp)
                            else:
                                if not tmp in not_int_activities:
                                    not_int_activities.append(tmp)
    return activities, not_int_activities
    

def extract_number_of_events(logs):
    results = [[('event_count',sum([len(logs[i][j]) for j in range(0,len(logs[i]))]))] for i in range(0,len(logs))]
    return results

def extract_number_of_traces(logs):
    results = [[('trace_count',len(logs[i]))] for i in range(0,len(logs))]
    return results

def extract_number_of_distinct_events(logs):
    results = []
    for log in logs:
        all_events = []
        for trace in range(0,len(log)):
            for event in range(0,len(log[trace])):
                all_events.append(log[trace][event]['concept:name'])
        results.append([('distinct_event_count',len(set(all_events)) )])        
        
    return results

def extract_direct_follows_relationships(logs):
    results = []
    for log in logs:
        graph = dfg_discovery.apply(log)
        log_results = []
        for element in set(graph.elements()):
            log_results.append((str(element),graph[element]))
        results.append(log_results)
    return results

def extract_number_of_direct_follows_relationships(logs):
    results = []
    for log in logs:
        graph = dfg_discovery.apply(log)
        results.append([('direct_follows_relations_count',len(set(graph.elements())))])
    return results
  
def avg(data):
    return sum(data)/len(data)

def avg_set(data):
    d = set(data)
    return sum(d)/len(d)
def check_int(s):
    if isinstance(s,str):
        if s[0] in ('-', '+', '~'):
            return s[1:].isdigit()
        return s.isdigit()
    return isinstance(s, numbers.Number)

def replace_boolean_with_int(logs):
    for log in logs:
            for trace in range(0,len(log)):
                for event in range(0,len(log[trace])):
                    for attribute_key in log[trace][event]:
                        if not attribute_key == "case:concept:name":
                            if log[trace][event][attribute_key] == True:
                                log[trace][event][attribute_key] = 1
                            elif log[trace][event][attribute_key] == False:
                                log[trace][event][attribute_key] = 0
                           
    return logs


def mine_all_data(logs, operators = {'count':len,'sum':sum, 'max':max, 'min':min, 'avg_unique':avg_set, 'avg':avg}):
    '''Uses the aggregation functions on all attributes not excluded by attribute_set
    args:
        logs: sublogs of the timeframe splitting
        attribute_set: Attribute set to be excluded
        operators: dictionary of aggregation functions
    returns:
        list of feature tuples
    '''
    #Performance can be improved when changing order of loops
    results = []
    for log in logs:
        curr_res = []
        activities = []
        #store all activities that have attributes with numbers
        for trace in range(0,len(log)):
            for att in log[trace].attributes.keys():
                tmp = ("trace", att)
                if not tmp in activities and not att == "concept:name":
                    if check_int(log[trace].attributes[att]):
                        activities.append(tmp)
            for event in range(0,len(log[trace])):
                for attribute_key in log[trace][event]:
                    if not attribute_key == "case:concept:name" and not attribute_key == "org:resource" :
                        if check_int(log[trace][event][attribute_key]):
                            tmp = (log[trace][event]['concept:name'], attribute_key)
                            if not tmp in activities:
                                activities.append(tmp)
        for operator in operators:
            for activity, attribute in set(activities):
                all_v = []
                for trace in range(0,len(log)):
                    if activity == "trace":
                        all_v.append(int(log[trace].attributes[attribute]))
                    for event in range(0,len(log[trace])):
                        if log[trace][event]['concept:name'] == activity:
                            if isinstance(log[trace][event][attribute],str):
                                log[trace][event][attribute] = log[trace][event][attribute].replace("~","-")
                            if not check_int(log[trace][event][attribute]):
                                all_v.append(0)
                            else:
                                all_v.append(int(log[trace][event][attribute]))
                curr_res.append((activity+":"+attribute+":"+operator, operators[operator](all_v)))
        results.append(curr_res)
   
    return results
      

def extract_alpha(logs):
    '''extracts the alpha relations for all sublog
    args:
        logs: list of sublogs
    returns:
        list of feature tuples
    '''
    print("Alpha is currently not available, since it has to be fitted to new verison of PM4Py!")
    results=[]
# =============================================================================
#     results=[]
#     parameters = {}
#     parameters[pm_util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = 'concept:name'
#     for log in logs:
#         dfg = {k: v for k, v in dfg_inst.apply(log).items() if v > 0}
#         start_activities = dfg_utils.infer_start_activities(dfg)
#         end_activities = dfg_utils.infer_end_activities(dfg)
#         alpha_abstraction = alpha_classic_abstraction.ClassicAlphaAbstraction(start_activities, end_activities, dfg,
#                                                                           activity_key=parameters[
#                                                                               PARAMETER_CONSTANT_ACTIVITY_KEY])
#         causal_rel = list(alpha_abstraction.causal_relation)
#         concurrent_rel = alpha_abstraction.parallel_relation
#         concurrent_rel = list(set([tuple(sorted(t)) for t in concurrent_rel]))
#         log_results = []
#         for element in causal_rel:
#             log_results.append(('causal_'+str(element),1))
#         for element in concurrent_rel:
#             log_results.append(('concurrent_'+str(element),1))
#         results.append(log_results)
#             
# =============================================================================
    
        
    return results
    

    
#direct follows with performance of edges
def extract_performance_of_direct_follows_relationships(logs):
    results = []
    for log in logs:
        graph = dfg_discovery.apply(log)
        graph = dfg_discovery.apply(log, variant="performance")
        log_results = []
        for element in set(graph):
            log_results.append((str(element),graph[element]))
        results.append(log_results)
    return results

def extract_event_durations(logs):
    results = []
    for log in logs:
        duration_lists = {}
        enriched_log = interval_lifecycle.assign_lead_cycle_time(log)
        for case in enriched_log:
            for event in case:
                if event["concept:name"] not in duration_lists.keys():
                    duration_lists[event["concept:name"]] = []
                duration_lists[event["concept:name"]].append(event["@@duration"])    
        log_results = [(key, sum(duration_lists[key])/len(duration_lists[key])) for key in duration_lists.keys()]
        results.append(log_results)
    return results

def extract_heuristics(logs):
    results = []
    for log in logs:
        log_results =[]
        heu_net = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.90})
       #print(heu_net.dependency_matrix)
        for node in heu_net.dependency_matrix:
            #print(heu_net.dependency_matrix[node])
            for target in heu_net.dependency_matrix[node]:
                log_results.append(("Heu_"+node+"_to_"+target,heu_net.dependency_matrix[node][target]))
                
        results.append(log_results)
    return results

def extract_workload(logs):
    results = []
    for log in logs:
        workload = {}
        workload["total"] = 0
        for trace in log:
            for event in trace:
                if 'org:resource' in event.keys():
                    res = event['org:resource']
                    if not res in workload.keys():
                        workload[res] = 0
                    workload[res]+=1
                    workload["total"] += 1
        log_results = [('WL_Resource'+str(res),workload[res]) for res in workload.keys()]
        results.append(log_results)
    return results

def extract_overtime(logs):
    delta = 60 # 1 hour, should be changed for other cases
    results = []
    for log in logs:
        duration_lists = {}
        enriched_log = interval_lifecycle.assign_lead_cycle_time(log)
        for case in enriched_log:
            for event in case:
                if event["@@approx_bh_partial_cycle_time"] > delta:
    
                    if event["concept:name"] not in duration_lists.keys():
                        duration_lists[event["concept:name"]] = 0
                    duration_lists[event["concept:name"]]+=1    
        log_results = [(key, duration_lists[key]) for key in duration_lists.keys()]
        results.append(log_results)
    return results

def __initial_filter(parallel_relation, pair):
    if (pair[0], pair[0]) in parallel_relation or (pair[1], pair[1]) in parallel_relation:
        return False
    return True

def pca_reduction(features_np, dimensions, normalize = False, normalize_function = 'max'):
    '''Reduces a time series of features
    features: Two dimensional array of features
    dimensions: Target dimensionality. Use 'mle' for automated choice of dimensionality
      Automated choice of dimensionality can sometimes fail, a manual choice of dimensionality
      is then needed. If the Feautres are more than the time series is long, the automated
      choice can not be applied. We sole this by first reducing the features to the length
      of the time series and then reducing automatically.
    normalize: Whether the feautres should be normalized before reduction. The
      features should be normalized, if they have very different scales.
    normalize_function: Choose 'max' or 'sum'
    '''
    print(features_np.shape)
    #print(features_np)
    if normalize:
        row_sums = features_np.sum(axis=0) + 0.0001
        if normalize_function == 'max':
            row_sums = features_np.max(axis=0) + 0.0001
        #print(row_sums)
        new_matrix = features_np / row_sums[np.newaxis, : ]
        features_np = new_matrix
        #print(features_np)
    tmp_features = features_np
    if dimensions == 'mle':
        if features_np.shape[1] > features_np.shape[0]:
            pca = PCA(n_components = features_np.shape[0], svd_solver ="full")
            pca.fit(features_np)
            tmp_features = pca.transform(features_np)
    pca = PCA(n_components = dimensions, svd_solver ="full")
    pca.fit(tmp_features)
    reduced_features = pca.transform(tmp_features)
    
    if reduced_features.shape[1]==0:
        pca = PCA(n_components = 1, svd_solver ="full")
        pca.fit(tmp_features)
        reduced_features = pca.transform(tmp_features)
    print("Original features: ", features_np.shape)
    print("Reduced features shape: ", reduced_features.shape)
    return reduced_features