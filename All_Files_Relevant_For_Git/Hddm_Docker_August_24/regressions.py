# various regression models
# TO DO: Think about median-splitting subjective value

# Version 0: Fixed effect of sv_pain on the drift rate v with a random intercept
# v_reg = {'model': 'v ~ 1 + sv_pain','link_func': lambda x: x}
# Interpretation: This model examines how the subjective value of pain (sv_pain) influences the drift rate (v), with a fixed effect and a random intercept for participants.
# Hypothesis: The drift rate decreases as the subjective value of pain increases, indicating that higher pain levels make participants more cautious or slower in their decision-making.

# Version 1: Drift rate depends on sv_pain, sv_money, and their interaction
# v_reg = {'model': 'v ~ 1 + sv_pain + sv_money + sv_pain * sv_money','link_func': lambda x: x}
# Interpretation: This model includes both sv_pain and sv_money and their interaction as predictors of the drift rate.
# Hypothesis: The interaction between pain and money influences the drift rate, potentially indicating a trade-off between pain and money when making decisions.

# Version 5: Drift rate depends on combined sv (sv_both) with random intercepts
# v_reg = {'model': 'v ~ 1 + sv_both ','link_func': lambda x: x}
# Interpretation: This model uses a combined subjective value of pain and money (sv_both) to predict the drift rate.
# Hypothesis: A single metric representing the combined value of pain and money affects decision-making speed.

# Version 6: Drift rate depends on sv_pain, sv_money, sv_both with random intercepts
# v_reg = {'model': 'v ~ 1 + sv_pain + sv_money + sv_both','link_func': lambda x: x}
# Interpretation: This model considers the independent contributions of sv_pain, sv_money, and sv_both.
# Hypothesis: Independent and combined effects of pain and money values influence decision-making.

# Version 7: Independent effects of sv_pain, sv_money, sv_both, and their three-way interaction
# v_reg = {'model': 'v ~ 1 + sv_pain + sv_money + sv_both + sv_pain * sv_money * sv_both','link_func': lambda x: x}
# Interpretation: This model includes main effects and three-way interactions among sv_pain, sv_money, and sv_both.
# Hypothesis: Complex interactions among pain and money values significantly affect decision-making.

# Version 9: Interaction of fixduration and sv_pain on the drift rate
# v_reg = {'model': 'v ~ 1 + sv_pain * fixduration ','link_func': lambda x: x}
# Interpretation: This model includes the interaction between the time spent fixating on stimuli (fixduration) and sv_pain.
# Hypothesis: Longer fixation times combined with pain values influence decision-making speed.

# Version 10: Interaction of sv_pain and fixduration on non-decision time (t)
# t_reg = {'model': 't ~ 1 + sv_pain * fixduration ','link_func': lambda x: x}
# Interpretation: This model focuses on how the interaction between fixduration and sv_pain affects non-decision time.
# Hypothesis: The time spent fixating on stimuli combined with pain values influences the time taken before making a decision.

## mapping of acceptance_pair conditions (money>pain, identical, pain>money)

# mapping_m = {             # mapping_m: differences between levels where money > pain and levels where pain > money              
#     'M': 1,
#     'P': 0
# }

# mapping_p = {             # mapping_m: differences between levels where money > pain and levels where pain > money              
#     'P': 1,
#     'M': 0
# }

# mapping_a = {             # mapping_a: differentiates between high money (1) and high pain (0) irrespective of overall value
#     'h_OV_h_money': 1,
#     'low_OV_h_money': 1,
#     'low_OV_h_pain': 0,
#     'h_OV_h_pain': 0
# }
# mapping_b = {             # mapping_b: differentiates between high overall value (1) and low overall value (0) irrespective of money/pain condition
#     'h_OV_h_money': 1,
#     'h_OV_h_pain': 1,
#     'low_OV_h_money': 0,
#     'low_OV_h_pain': 0
# }
# mapping_c = {             # mapping_c: differentiates between high money (1) and high pain  (0) irrespective of absolute money
#     'high_abs_h_money': 1,
#     'low_abs_h_money': 1,
#     'low_abs_h_pain': 0,
#     'high_abs_h_pain': 0
# }
# mapping_d = {             # mapping_d: differentiates between high absolute money (1) and low absolute money  (0) irrespective of money/pain condition
#     'high_abs_h_money': 1,
#     'high_abs_h_pain': 1,
#     'low_abs_h_money': 0,
#     'low_abs_h_pain': 0
# }

# # creating new columns based on the mapping to facilitate stimulus coding analysis, since it only allows coding between 2 options

# data_accept['stimulus_mapped_m'] = data_accept['acceptance_pair'].map(mapping_m)
# data_accept['stimulus_mapped_p'] = data_accept['acceptance_pair'].map(mapping_p)

# data['stimulus_mapped_a'] = data['OV_Money_Pain'].map(mapping_a)
# data['stimulus_mapped_b'] = data['OV_Money_Pain'].map(mapping_b)
# data['stimulus_mapped_c'] = data['Abs_Money_Pain'].map(mapping_c)
# data['stimulus_mapped_d'] = data['Abs_Money_Pain'].map(mapping_d)

#------------------------------------------------------------------------------------------------------------------
# custom-made link functions for regression models to transform z and v, depending on m>p or p>m
#------------------------------------------------------------------------------------------------------------------

# stimulus coding of z, 'stimulus_mapped_m', z ~ (money > pain), -z ~ (pain > money)
def z_stimulus_link_func_m(x, data=data_accept):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_m.loc[x.index]})))
    z_flip = stim-x
    z_flip[stim==0]*=1  # above inverts values we don't want to flip, so we invert them back
    return z_flip

# stimulus coding of z, 'stimulus_mapped_p', z ~ (pain > money), -z ~ (money>pain)
def z_stimulus_link_func_p(x, data=data_accept):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_p.loc[x.index]})))
    z_flip = stim-x
    z_flip[stim==0]*=1  # above inverts values we don't want to flip, so we invert them back
    return z_flip

# stimulus coding of v, 'stimulus_mapped_m', v ~ (money > pain), -v ~ (pain > money)
def v_stimulus_link_func_m(x, data=data_accept):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_m.loc[x.index]})))
    v_flip = stim-x
    v_flip[stim==0]*=1   # above inverts values we don't want to flip, so we invert them back
    return v_flip 

# stimulus coding of v, 'stimulus_mapped_p', v ~ (pain > money), -v ~ (money>pain)
def v_stimulus_link_func_p(x, data=data_accept):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_p.loc[x.index]})))
    v_flip = stim-x
    v_flip[stim==0]*=1   # above inverts values we don't want to flip, so we invert them back
    return v_flip 

#------------------------------------------------------------------------------------------------------------------
# custom-made link functions for regression models to transform z and v, depending on overall value
#------------------------------------------------------------------------------------------------------------------
# stimulus coding of z, 'stimulus_mapped_a', z ~ (money > pain), -z ~ (pain > money)
def z_stimulus_link_func_OV_a(x, data=data):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_a.loc[x.index]})))
    z_flip = stim-x
    z_flip[stim==0]*=1  # above inverts values we don't want to flip, so we invert them back
    return z_flip

# stimulus coding of z, 'stimulus_mapped_a', z ~ (high overall value), -z ~ (low overall value)
def z_stimulus_link_func_OV_b(x, data=data):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_b.loc[x.index]})))
    z_flip = stim-x
    z_flip[stim==0]*=1  # above inverts values we don't want to flip, so we invert them back
    return z_flip

# stimulus coding of v, 'stimulus_mapped_a', v ~ (money > pain), -v ~ (pain > money)
def v_stimulus_link_func_OV_a(x, data=data):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_a.loc[x.index]})))
    v_flip = stim-x
    v_flip[stim==0]*=1   # above inverts values we don't want to flip, so we invert them back
    return v_flip 

# stimulus coding of v, 'stimulus_mapped_b', v ~ (high overall value), -v ~ (low overall value)
def v_stimulus_link_func_OV_b(x, data=data):
    stim = (np.asarray(dmatrix('0 + c(s, [[0], [1]])', {'s':data.stimulus_mapped_b.loc[x.index]})))
    v_flip = stim-x
    v_flip[stim==0]*=1   # above inverts values we don't want to flip, so we invert them back
    return v_flip 
