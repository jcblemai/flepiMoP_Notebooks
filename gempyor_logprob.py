import gempyor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob, os, sys, re, shutil
from pathlib import Path
import numba as nb

# import seaborn as sns
import os
import pandas as pd
from gempyor import config, outcomes, seir
import scipy



def run_simulation(snpi_df_in, hnpi_df_in, modinf, p_draw, unique_strings, transition_array, proportion_array, proportion_info, initial_conditions, seeding_data, seeding_amounts,outcomes_parameters, save=False):
    random_id = np.random.randint(0,1e8)
    npi_seir = seir.build_npi_SEIR(
        modinf=modinf, load_ID=False, sim_id2load=None, config=config, 
        bypass_DF=snpi_df_in
    )
    if modinf.npi_config_outcomes:
        npi_outcomes = outcomes.build_outcome_modifiers(
                    modinf=modinf,
                    load_ID=False,
                    sim_id2load=None,
                    config=config,
                    bypass_DF=hnpi_df_in
                )

    # reduce them
    parameters = modinf.parameters.parameters_reduce(p_draw, npi_seir)
            # Parse them
    parsed_parameters = modinf.compartments.parse_parameters(
        parameters, modinf.parameters.pnames, unique_strings
    )

    seeding_data_nbdict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.int64[:])

    for k,v in seeding_data.items():
        seeding_data_nbdict[k] = np.array(v, dtype=np.int64)

    states = seir.steps_SEIR(
        modinf,
        parsed_parameters,
        transition_array,
        proportion_array,
        proportion_info,
        initial_conditions,
        seeding_data_nbdict,
        seeding_amounts,
    )
    seir_out_df = seir.states2Df(modinf, states)
    if save:
        modinf.write_simID(ftype="seir", sim_id=random_id, df=seir_out_df)


    outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
        modinf=modinf,
        sim_id2write=0,
        parameters=outcomes_parameters,
        loaded_values=None,
        npi=npi_outcomes,
        bypass_seir=seir_out_df
    )
    outcomes_df, hpar, hnpi = outcomes.postprocess_and_write(
        sim_id=0,
        modinf=modinf,
        outcomes_df=outcomes_df,
        hpar=hpar_df,
        npi=npi_outcomes,
    )
    # outcomes_df["time"] = outcomes_df["date"] which one should it be ?
    outcomes_df = outcomes_df.set_index("date")
    
    
    if save:
        modinf.write_simID(ftype="hosp", sim_id=random_id, df=outcomes_df)

    return outcomes_df


def compute_likelyhood(model_df, gt, modinf, statistics):
    log_loss = 0
    
    for subpop in modinf.subpop_struct.subpop_names:
        # essential to sort by index (date here)
        gt_s = gt[gt["subpop"]==subpop].sort_index()
        model_df_s = model_df[model_df["subpop"]==subpop].sort_index()

        # places where data and model overlap
        first_date = max(gt_s.index.min(), model_df_s.index.min())
        last_date = min(gt_s.index.max(), model_df_s.index.max())

        gt_s = gt_s.loc[first_date:last_date].drop(["subpop"],axis=1).resample("W-SAT").sum()
        model_df_s = model_df_s.drop(["subpop","time"],axis=1).loc[first_date:last_date].resample("W-SAT").sum() # todo sub subpop here
        for key, value in statistics.items():
            assert model_df_s[key].shape == gt_s[value].shape

            log_loss += sum((model_df_s[key]-gt_s[value])**2)

    return -log_loss


def check_in_bounds(proposal, fitted_params):
    if (proposal < fitted_params["lb"]).any() or (proposal > fitted_params["ub"]).any():
        return False
    return True


def input_proposal(proposal, snpi_df_ref, hnpi_df_ref, fitted_params, ndim):
    snpi_df_mod = snpi_df_ref.copy(deep=True)
    hnpi_df_mod = hnpi_df_ref.copy(deep=True)

    for i in range(ndim):
        if fitted_params["ptype"][i] == "snpi":
            snpi_df_mod.loc[(snpi_df_mod["modifier_name"] == fitted_params["pname"][i]) & (snpi_df_mod["subpop"] == fitted_params["subpop"][i]),"value"] = proposal[i]
        elif fitted_params["ptype"][i] == "hnpi":
            hnpi_df_mod.loc[(hnpi_df_mod["modifier_name"] == fitted_params["pname"][i]) & (hnpi_df_mod["subpop"] == fitted_params["subpop"][i]),"value"] = proposal[i]

    return snpi_df_mod, hnpi_df_mod




def log_prob(proposal, snpi_df_ref, ndim, statistics, fitted_params, gt, hnpi_df_ref, modinf, p_draw, unique_strings, transition_array, proportion_array, proportion_info, initial_conditions, seeding_data, seeding_amounts,outcomes_parameters, save=False):
    if not check_in_bounds(proposal=proposal, fitted_params=fitted_params):
        print("OUT OF BOUND!!")
        return -np.inf
    
    snpi_df_mod, hnpi_df_mod = input_proposal(proposal, snpi_df_ref, hnpi_df_ref, fitted_params, ndim)

    outcomes_df = run_simulation(snpi_df_mod, 
                                hnpi_df_mod,
                                modinf, p_draw, unique_strings, transition_array, proportion_array, proportion_info, initial_conditions, seeding_data, seeding_amounts,outcomes_parameters, save=save)

    llik = compute_likelyhood(model_df=outcomes_df, gt=gt, modinf=modinf, statistics=statistics)
    print(f"llik is {llik}")

    return llik

