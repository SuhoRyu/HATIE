from scipy.stats import bootstrap
import numpy as np

bg_cst_obj_w = [0.22, 0.17, 0.25, 0.36]
bg_cst_style_w1 = [[0.48, 0.12, 0.4], [0.01, 0.02, 0.97], [0.49, 0.01, 0.5], [0.03, 0.96, 0.01]] 
bg_cst_style_w2 = [0.01, 0.49, 0.01, 0.49]
bg_cst_style_w = [[bg_cst_style_w1[idx][jdx]*bg_cst_style_w2[idx] for jdx in range(3)] for idx in range(4)]

obj_cst_resbg_w1 = [0.23, 0.01, 0.11, 0.65]
obj_cst_resbg_w2 = [0.6, 0.12, 0.9]
obj_cst_attr_w1 = [[0.08, 0.57, 0.35], [0.01, 0.03, 0.96], [0.93, 0.05, 0.02], [0.01, 0.01, 0.98]]
obj_cst_attr_w2 = [0.84, 0.01, 0.02, 0.13]
obj_cst_attr_w3 = [0.7, 0.23, 0.07]

bg_fid_w = [0.87, 0.58]

obj_fid_addrep_w1 = [0.01, 0.92, 0.07]
obj_fid_addrep_w2 = [0.03, 0.71, 0.26]
obj_fid_attr_w = [0.36, 0.57, 0.07, 0, 0]

total_w = [0.3, 0.1, 0.2, 0.4, 0.8]

WEIGHTS = {
    "total": total_w, 

    "obj_add": {
        "bg_cst": bg_cst_obj_w, 
        "obj_fid": [obj_fid_addrep_w1[idx]*obj_fid_addrep_w2[0] for idx in range(3)] + obj_fid_addrep_w2[1:]
    },

    "obj_rep": {
        "bg_cst": bg_cst_obj_w,
        "obj_fid": obj_fid_addrep_w1 + [0, 0]
    },

    "obj_resize": {
        "bg_cst": bg_cst_obj_w,
        "obj_cst": [obj_cst_resbg_w1[idx]*obj_cst_resbg_w2[2] for idx in range(4)] + [1-obj_cst_resbg_w2[2]], 
        "obj_fid": [1, 0, 0, 0, 0] 
    },

    "attr_chg": {
        "bg_cst": bg_cst_obj_w,
        "obj_cst": [[obj_cst_attr_w1[idx][jdx]*obj_cst_attr_w2[idx]*obj_cst_attr_w3[0] for jdx in range(3)] for idx in range(4)] + obj_cst_attr_w3[1:],
        "obj_fid": obj_fid_attr_w
    },

    "bg_chg": {
        "obj_cst": [obj_cst_resbg_w1[idx]*obj_cst_resbg_w2[0] for idx in range(4)] + [obj_cst_resbg_w2[1], 1-obj_cst_resbg_w2[0]-obj_cst_resbg_w2[1]], 
        "bg_fid": [bg_fid_w[0], 1-bg_fid_w[0]] 
    },

    "style_chg": {
        "bg_cst": bg_cst_style_w,  
        "bg_fid": [bg_fid_w[1], 1-bg_fid_w[1]] 
    },

    "obj_remove": {
        "bg_cst": bg_cst_obj_w, 
        "obj_fid": obj_fid_addrep_w1 + [0, 0]
    }
}

def agg_obj_add(scores, weights):
    bg_cst = 0
    for i, scr in enumerate(scores['bg_cst']):
        if scr is not None: bg_cst += scr * weights['bg_cst'][i]
    
    bg_cst = bg_cst**6 
    obj_cst = None
    bg_fid = None

    obj_fid = 0
    for i, scr in enumerate(scores['obj_fid']):
        if scr is not None: obj_fid += scr * weights['obj_fid'][i]
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def agg_obj_rep(scores, weights):
    bg_cst = 0
    for i, scr in enumerate(scores['bg_cst']):
        if scr is not None: bg_cst += scr * weights['bg_cst'][i]
    
    bg_cst = bg_cst**6 
    obj_cst = scores['obj_cst'][4]
    bg_fid = None

    obj_fid = 0
    for i, scr in enumerate(scores['obj_fid']):
        if scr is not None: obj_fid += scr * weights['obj_fid'][i]
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def agg_obj_resize(scores, weights):
    bg_cst = 0
    for i, scr in enumerate(scores['bg_cst']):
        if scr is not None: bg_cst += scr * weights['bg_cst'][i]

    bg_cst = bg_cst**6 
    
    obj_cst = 0
    for i, scr in enumerate(scores['obj_cst']):
        if scr is not None: obj_cst += scr * weights['obj_cst'][i]
    
    bg_fid = None

    obj_fid = 0
    for i, scr in enumerate(scores['obj_fid']):
        if scr is not None: obj_fid += scr * weights['obj_fid'][i]
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def agg_attr_chg(scores, weights):
    bg_cst = 0
    for i, scr in enumerate(scores['bg_cst']):
        if scr is not None: bg_cst += scr * weights['bg_cst'][i]

    bg_cst = bg_cst**6 
    
    obj_cst = 0
    for i, scr in enumerate(scores['obj_cst']):
        if type(scr) is tuple or type(scr) is list:
            for j in range(len(scr)):
                if scr[j] is not None: obj_cst += scr[j] * weights['obj_cst'][i][j]
        else:
            if scr is not None: obj_cst += scr * weights['obj_cst'][i]
   
    bg_fid = None
    
    obj_fid = 0
    for i, scr in enumerate(scores['obj_fid']):
        if scr is not None: obj_fid += scr * weights['obj_fid'][i]
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def f_bgfid(x):
    if x<0.54: return 0.05 * x
    elif x<0.66: return (-3318.33833)*(0.3564*x - 0.6*x**2 + 0.333333*x**3 -  0.0698399) + (0.05*0.54 + 1 - 0.05*0.34)/2
    else: return 0.983 + 0.05 * (x - 0.66)

def agg_bg_chg(scores, weights):
    bg_cst = None
    obj_cst = 0
    for i, scr in enumerate(scores['obj_cst']):
        if scr is not None: 
            for j in range(len(scr)): 
                obj_cst += scr[j] * weights['obj_cst'][i] / len(scr)
    
    bg_fid = 0
    for i, scr in enumerate(scores['bg_fid']):
        if scr is not None: bg_fid += scr * weights['bg_fid'][i]

    bg_fid = f_bgfid(bg_fid)
    
    obj_fid = None
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def agg_style_chg(scores, weights):
    bg_cst = 0
    for i, scr in enumerate(scores['bg_cst']):
        for j in range(len(scr)):
            if scr[j] is not None: bg_cst += scr[j] * weights['bg_cst'][i][j]
    
    obj_cst = None
    bg_fid = 0
    for i, scr in enumerate(scores['bg_fid']):
        if scr is not None: bg_fid += scr * weights['bg_fid'][i]

    bg_fid = f_bgfid(bg_fid)
    
    obj_fid = None
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def agg_obj_remove(scores, weights):
    bg_cst = 0
    for i, scr in enumerate(scores['bg_cst']):
        if scr is not None: bg_cst += scr * weights['bg_cst'][i]
    
    bg_cst = bg_cst**6 
    obj_cst = None
    bg_fid = None

    obj_fid = 0
    for i, scr in enumerate(scores['obj_fid']):
        if scr is not None: obj_fid += scr * weights['obj_fid'][i]
    
    return bg_cst, obj_cst, bg_fid, obj_fid

def agg_score(type, scores):
    if type == 'obj_add':
        return agg_obj_add(scores, WEIGHTS['obj_add'])
    elif type == 'obj_rep':
        return agg_obj_rep(scores, WEIGHTS['obj_rep'])
    elif type == 'obj_resize':
        return agg_obj_resize(scores, WEIGHTS['obj_resize'])
    elif type == 'attr_chg':
        return agg_attr_chg(scores, WEIGHTS['attr_chg'])
    elif type == 'bg_chg':
        return agg_bg_chg(scores, WEIGHTS['bg_chg'])
    elif type == 'style_chg':
        return agg_style_chg(scores, WEIGHTS['style_chg'])
    elif type == 'obj_remove':
        return agg_obj_remove(scores, WEIGHTS['obj_remove'])
    
def merge_scores(scores, img_qual, get_err=False):
    weights = WEIGHTS['total']
    bg_csts = [score['bg_cst'] for score in scores if score['bg_cst'] is not None]
    obj_csts = [score['obj_cst'] for score in scores if score['obj_cst'] is not None]
    bg_fids = [score['bg_fid'] for score in scores if score['bg_fid'] is not None]
    obj_fids = [score['obj_fid'] for score in scores if score['obj_fid'] is not None]

    tot_bg_cst = np.average(bg_csts)
    tot_obj_cst = np.average(obj_csts)
    tot_bg_fid = np.average(bg_fids)
    tot_obj_fid = np.average(obj_fids)

    tot_err, bg_cst_err, obj_cst_err, bg_fid_err, obj_fid_err = None, None, None, None, None
    if get_err:
        assert img_qual[1] is not None, "Image quality error is not available"
        nsample = 10000
        batch = 1000
        bg_cst_err = bootstrap((bg_csts,), np.average, n_resamples=nsample, batch=batch).standard_error
        obj_cst_err = bootstrap((obj_csts,), np.average, n_resamples=nsample, batch=batch).standard_error
        bg_fid_err = bootstrap((bg_fids,), np.average, n_resamples=nsample, batch=batch).standard_error
        obj_fid_err = bootstrap((obj_fids,), np.average, n_resamples=nsample, batch=batch).standard_error
        tot_err = ((bg_cst_err*weights[0]*weights[4])**2 
                   + (obj_cst_err*weights[1]*weights[4])**2 
                   + (obj_fid_err*weights[2]*weights[4])**2 
                   + (bg_fid_err*weights[3]*weights[4])**2
                   + (img_qual[1]*(1-weights[4])))**0.5

    tot_score = (tot_bg_cst*weights[0] + tot_obj_cst*weights[1] + tot_obj_fid*weights[2] + tot_bg_fid*weights[3]) * weights[4] + img_qual[0] * (1-weights[4])

    total_scores = {
            'total': [tot_score, tot_err],
            'bg_cst': [tot_bg_cst, bg_cst_err],
            'obj_cst': [tot_obj_cst, obj_cst_err],
            'bg_fid': [tot_bg_fid, bg_fid_err],
            'obj_fid': [tot_obj_fid, obj_fid_err],
            'img_qual': img_qual
        }

    return total_scores

def query_type_scores(scores, get_err=True, do_remove=False):
    add_scores = np.array([[scr['bg_cst'],scr['obj_fid']] for scr in scores if scr['query_info']['type'] == 'obj_add'])
    add_bg_cst = np.average(add_scores[:,0])
    add_obj_fid = np.average(add_scores[:,1])
    add_score = (add_bg_cst*WEIGHTS["total"][0] + add_obj_fid*WEIGHTS["total"][3]) / (WEIGHTS["total"][0] + WEIGHTS["total"][3])
    add_bg_cst_err, add_obj_fid_err, add_score_err = None, None, None
    if get_err: 
        add_bg_cst_err = bootstrap((add_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
        add_obj_fid_err = bootstrap((add_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
        add_score_err = np.sqrt((add_bg_cst_err*WEIGHTS["total"][0])**2 + (add_obj_fid_err*WEIGHTS["total"][3])**2) / (WEIGHTS["total"][0] + WEIGHTS["total"][3])

    rep_scores = np.array([[scr['bg_cst'],scr['obj_fid']] for scr in scores if scr['query_info']['type'] == 'obj_rep'])
    rep_bg_cst = np.average(rep_scores[:,0])
    rep_obj_fid = np.average(rep_scores[:,1])
    rep_score = (rep_bg_cst*WEIGHTS["total"][0] + rep_obj_fid*WEIGHTS["total"][3]) / (WEIGHTS["total"][0] + WEIGHTS["total"][3])
    rep_bg_cst_err, rep_obj_fid_err, rep_score_err = None, None, None
    if get_err: 
        rep_bg_cst_err = bootstrap((rep_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
        rep_obj_fid_err = bootstrap((rep_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
        rep_score_err = np.sqrt((rep_bg_cst_err*WEIGHTS["total"][0])**2 + (rep_obj_fid_err*WEIGHTS["total"][3])**2) / (WEIGHTS["total"][0] + WEIGHTS["total"][3])
    
    res_scores = np.array([[scr['bg_cst'],scr['obj_cst'],scr['obj_fid']] for scr in scores if scr['query_info']['type'] == 'obj_resize'])
    resize_bg_cst = np.average(res_scores[:,0])
    resize_obj_cst = np.average(res_scores[:,1])
    resize_obj_fid = np.average(res_scores[:,2])
    resize_score = (resize_bg_cst*WEIGHTS["total"][0] + resize_obj_cst*WEIGHTS["total"][1] + resize_obj_fid*WEIGHTS["total"][3]) / (WEIGHTS["total"][0] + WEIGHTS["total"][1] + WEIGHTS["total"][3])
    resize_bg_cst_err, resize_obj_cst_err, resize_obj_fid_err, resize_score_err = None, None, None, None
    if get_err: 
        resize_bg_cst_err = bootstrap((res_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
        resize_obj_cst_err = bootstrap((res_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
        resize_obj_fid_err = bootstrap((res_scores[:,2],), np.average, n_resamples=10000, batch=1000).standard_error
        resize_score_err = np.sqrt((resize_bg_cst_err*WEIGHTS["total"][0])**2 + (resize_obj_cst_err*WEIGHTS["total"][1])**2 + (resize_obj_fid_err*WEIGHTS["total"][3])**2) / (WEIGHTS["total"][0] + WEIGHTS["total"][1] + WEIGHTS["total"][3])
    
    attr_scores = np.array([[scr['bg_cst'],scr['obj_cst'],scr['obj_fid']] for scr in scores if scr['query_info']['type'] == 'attr_chg'])
    attr_bg_cst = np.average(attr_scores[:,0])
    attr_obj_cst = np.average(attr_scores[:,1])
    attr_obj_fid = np.average(attr_scores[:,2])
    attr_score = (attr_bg_cst*WEIGHTS["total"][0] + attr_obj_cst*WEIGHTS["total"][1] + attr_obj_fid*WEIGHTS["total"][3]) / (WEIGHTS["total"][0] + WEIGHTS["total"][1] + WEIGHTS["total"][3])
    attr_bg_cst_err, attr_obj_cst_err, attr_obj_fid_err, attr_score_err = None, None, None, None
    if get_err: 
        attr_bg_cst_err = bootstrap((attr_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
        attr_obj_cst_err = bootstrap((attr_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
        attr_obj_fid_err = bootstrap((attr_scores[:,2],), np.average, n_resamples=10000, batch=1000).standard_error
        attr_score_err = np.sqrt((attr_bg_cst_err*WEIGHTS["total"][0])**2 + (attr_obj_cst_err*WEIGHTS["total"][1])**2 + (attr_obj_fid_err*WEIGHTS["total"][3])**2) / (WEIGHTS["total"][0] + WEIGHTS["total"][1] + WEIGHTS["total"][3])

    bg_scores = np.array([[scr['obj_cst'],scr['bg_fid']] for scr in scores if scr['query_info']['type'] == 'bg_chg'])
    bg_chg_obj_cst = np.average(bg_scores[:,0])
    bg_chg_obj_cst_err, bg_chg_bg_fid_err, bg_chg_score_err = None, None, None
    bg_chg_bg_fid = np.average(bg_scores[:,1])
    bg_chg_score = (bg_chg_obj_cst*WEIGHTS["total"][1] + bg_chg_bg_fid*WEIGHTS["total"][2]) / (WEIGHTS["total"][1] + WEIGHTS["total"][2])
    if get_err: 
        bg_chg_obj_cst_err = bootstrap((bg_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
        bg_chg_bg_fid_err = bootstrap((bg_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
        bg_chg_score_err = np.sqrt((bg_chg_obj_cst_err*WEIGHTS["total"][1])**2 + (bg_chg_bg_fid_err*WEIGHTS["total"][2])**2) / (WEIGHTS["total"][1] + WEIGHTS["total"][2])
    
    style_scores = np.array([[scr['bg_cst'],scr['bg_fid']] for scr in scores if scr['query_info']['type'] == 'style_chg'])
    style_bg_cst = np.average(style_scores[:,0])
    style_bg_fid = np.average(style_scores[:,1])
    style_score = (style_bg_cst*WEIGHTS["total"][0] + style_bg_fid*WEIGHTS["total"][2]) / (WEIGHTS["total"][0] + WEIGHTS["total"][2])
    style_bg_cst_err, style_bg_fid_err, style_score_err = None, None, None
    if get_err: 
        style_bg_cst_err = bootstrap((style_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
        style_bg_fid_err = bootstrap((style_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
        style_score_err = np.sqrt((style_bg_cst_err*WEIGHTS["total"][0])**2 + (style_bg_fid_err*WEIGHTS["total"][2])**2) / (WEIGHTS["total"][0] + WEIGHTS["total"][2])

    qtype_scores = {
        'obj_add': {"total": [add_score,add_score_err], "bg_cst": [add_bg_cst,add_bg_cst_err], "obj_fid": [add_obj_fid,add_obj_fid_err]},
        'obj_rep': {"total": [rep_score,rep_score_err], "bg_cst": [rep_bg_cst, rep_bg_cst_err] , "obj_fid": [rep_obj_fid, rep_obj_fid_err]},
        'obj_resize': {"total": [resize_score, resize_score_err], "bg_cst": [resize_bg_cst, resize_bg_cst_err], "obj_cst": [resize_obj_cst, resize_obj_cst_err], "obj_fid": [resize_obj_fid, resize_obj_fid_err]},
        'attr_chg': {"total": [attr_score, attr_score_err], "bg_cst": [attr_bg_cst, attr_bg_cst_err], "obj_cst": [attr_obj_cst, attr_obj_cst_err], "obj_fid": [attr_obj_fid, attr_obj_fid_err]},
        'bg_chg': {"total": [bg_chg_score, bg_chg_score_err], "obj_cst": [bg_chg_obj_cst, bg_chg_obj_cst_err], "bg_fid": [bg_chg_bg_fid, bg_chg_bg_fid_err]},
        'style_chg': {"total": [style_score, style_score_err], "bg_cst": [style_bg_cst, style_bg_cst_err], "bg_fid": [style_bg_fid, style_bg_fid_err]}
    }

    if do_remove:
        rm_scores = np.array([[scr['bg_cst'],scr['obj_fid']] for scr in scores if scr['query_info']['type'] == 'obj_remove'])
        rm_bg_cst = np.average(rm_scores[:,0])
        rm_obj_fid = np.average(rm_scores[:,1])
        rm_score = (rm_bg_cst*WEIGHTS["total"][0] + rm_obj_fid*WEIGHTS["total"][3]) / (WEIGHTS["total"][0] + WEIGHTS["total"][3])
        rm_bg_cst_err, rm_obj_fid_err, rm_score_err = None, None, None
        if get_err: 
            rm_bg_cst_err = bootstrap((rm_scores[:,0],), np.average, n_resamples=10000, batch=1000).standard_error
            rm_obj_fid_err = bootstrap((rm_scores[:,1],), np.average, n_resamples=10000, batch=1000).standard_error
            rm_score_err = np.sqrt((rm_bg_cst_err*WEIGHTS["total"][0])**2 + (rm_obj_fid_err*WEIGHTS["total"][3])**2) / (WEIGHTS["total"][0] + WEIGHTS["total"][3])
        qtype_scores['obj_remove'] = {"total": [rm_score, rm_score_err], "bg_cst": [rm_bg_cst, rm_bg_cst_err], "obj_fid": [rm_obj_fid, rm_obj_fid_err]}

    return qtype_scores
