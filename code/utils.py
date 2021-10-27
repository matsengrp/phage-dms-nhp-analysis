"""
@File: utils.py

@Author: Kevin Sung
"""

# dependencies
import pandas as pd
import numpy as np
import xarray as xr
from Bio.Align import substitution_matrices


#------------------------------------
# Analysis definitions and constants
#====================================

# epitope region definitions
epitope_limits = {
    "CTDN" : [ 526,  593],
    "FP"   : [ 805,  835],
    "SHH"  : [1135, 1170]
}

epitope_names = {
    "CTDN" : "CTD-N'",
    "FP"   : "FP",
    "SHH"  : "SH-H"
}


# sample IDs by groups
vaccinated_pigtail = [
    103, 104, 105, 106, 107, 113,
    114, 115, 116, 117, 118
]

convalescent_rhesus = [
    186, 188, 190, 192, 194, 196,
    198, 200, 202, 204, 206, 208
]

moderna = [
    254, 256, 260, 262, 266, 268, 
    272, 274, 278, 282, 284, 286,
    290, 296, 300
]

conv_60d = [
    49, 51, 52, 59, 65, 66, 69, 
    71, 77, 79, 81, 83
]


# Okabe-Ito colors
oi_black         = '#000000'
oi_orange        = '#E69F00'
oi_skyblue       = '#56B4E9'
oi_bluishgreen   = '#009E73'
oi_yellow        = '#F0E442'
oi_blue          = '#0072B2'
oi_vermillion    = '#D55E00'
oi_reddishpurple = '#CC79A7'


#------------------------------------------
# Helper for dataset queries and selection
#==========================================
def id_coordinate_subset(
    ds,
    where,
    table="sample_table",
    is_equal_to=None,
    is_not_equal_to=None,
    is_greater_than=None,
    is_greater_than_or_equal_to=None,
    is_less_than=None,
    is_less_than_or_equal_to=None,
    is_in=None,
    is_valid=None,
):
    """
    a general function to compute the coordinate dimensions given some conditions.
    """

    if table not in ["sample_table", "peptide_table"]:
        raise ValueError(
            f"{table} is not a valid data table for {ds}\n Available data tables are: 'sample_table' or 'peptide_table'"
        )

    if table == "sample_table":
        metadata = "sample_metadata"
        metadata_features = ds[metadata]
        coord = "sample_id"
    else:
        metadata = "peptide_metadata"
        metadata_features = ds[metadata]
        coord = "peptide_id"

    if where not in metadata_features:
        raise ValueError(
            f"{where} is not in the sample metadata\n Available options are: {metadata_features.values}"
        )

    num_kw_args = [
        0 if arg is None else 1
        for arg in [
            is_equal_to,
            is_not_equal_to,
            is_greater_than,
            is_greater_than_or_equal_to,
            is_less_than,
            is_less_than_or_equal_to,
            is_in,
            is_valid,
        ]
    ]

    if sum(num_kw_args) != 1:
        raise ValueError(
            "You must provide exactly one of the keyword conditional arguments"
        )

    table = ds[table]
    dim = table.loc[{metadata: where}]
    coordinate_ids = ds[coord]

    if is_equal_to is not None:
        return coordinate_ids[dim == is_equal_to].values

    elif is_not_equal_to is not None:
        return coordinate_ids[dim != is_not_equal_to].values

    elif is_greater_than is not None:
        return coordinate_ids[dim > is_greater_than].values

    elif is_greater_than_or_equal_to is not None:
        return coordinate_ids[dim >= is_greater_than_or_equal_to].values

    elif is_less_than is not None:
        return coordinate_ids[dim < is_less_than].values

    elif is_less_than_or_equal_to is not None:
        return coordinate_ids[dim <= is_less_than_or_equal_to].values

    elif is_in is not None:
        return coordinate_ids[dim.isin(is_in)].values

    else:
        return coordinate_ids[dim == dim].values


def sample_id_coordinate_subset(ds, where, **kwargs):
    return id_coordinate_subset(ds, where, "sample_table", **kwargs)


def peptide_id_coordinate_subset(ds, where, **kwargs):
    return id_coordinate_subset(ds, where, "peptide_table", **kwargs)


#----------------------------------------
# Helpers for escape profile comparisons
#========================================

def get_aa_ordered_list():
    """
    return the ordered list of amino acid.
    This convention is based on the BLOSUM
    matrix in biopython and assumed for the
    binned distribution presenting amino
    acid contribution to differential 
    selection at a site
    """
    return list('ARNDCQEGHILKMFPSTWYV')


def get_cost_matrix():
    """
    return the default 40x40 cost matrix based on BLOSUM62
    and assigns maximum cost to transport between
    opposite signed differential selection contributions
    """

    substitution_matrix = substitution_matrices.load('BLOSUM62')
    alphabet_list = get_aa_ordered_list()
    Naa = len(alphabet_list)
    
    # chosen so that range of costs in the 
    # matrix is within an order of magnitude
    nthroot=7.
    
    # maximum cost assigned by the cost matrix
    maxMij = np.exp(np.max(-substitution_matrix)/nthroot)
    

    cost_matrix=[]

    # first 20 rows
    for aa in alphabet_list:
        row = [-x/nthroot for x in substitution_matrix[aa,:][:Naa]]
        cost_row = (np.exp(row)).tolist() + [maxMij for i in range(Naa)]
        cost_matrix.append(cost_row)

    # last 20 rows
    for aa in alphabet_list:
        row = [-x/nthroot for x in substitution_matrix[aa,:][:Naa]]
        cost_row = [maxMij for i in range(Naa)] + (np.exp(row)).tolist()
        cost_matrix.append(cost_row)

    return cost_matrix


def get_loc_escape_data(
    ds,
    sample_ID,
    loc,
    metric,
    normalized=True
):
    """
    return the normalized distribution represented as a list
    for the amino acid pattern of scaled differential
    selection for a specified site and individual
    
    metric: label of the scaled differential selection data in ds
    loc: peptide annotation label for the location
    The individual is specified by a sample annotation label
    in sample_factor (e.g. 'sample_ID') and the corresponding
    value in sfact_val
    """
    my_ds = ds.loc[
                dict(
                    peptide_id=peptide_id_coordinate_subset(ds,where='Loc',is_equal_to=loc),
                    sample_id=sample_id_coordinate_subset(ds,where='sample_ID',is_equal_to=sample_ID)
                )
            ]
    
    diff_sel = my_ds[metric].to_pandas().to_numpy().flatten()
    
    my_df = my_ds.peptide_table.loc[:,['aa_sub']].to_pandas()
    my_df['diff_sel'] = diff_sel
    
    esc_data_neg=[]
    esc_data_pos=[]
    alphabet_list = get_aa_ordered_list()
    for aa in alphabet_list:
        val = my_df[my_df['aa_sub']==aa]['diff_sel'].item()
        if val>0:
            esc_data_neg.append(0)
            esc_data_pos.append(val)
        else:
            esc_data_neg.append(-val)
            esc_data_pos.append(0)
    
    esc_data = esc_data_neg + esc_data_pos
    
    if normalized is False or np.sum(esc_data)==0:
        return esc_data
    else:
        return esc_data/np.sum(esc_data)
