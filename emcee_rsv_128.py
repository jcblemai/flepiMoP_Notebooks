import gempyor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob, os, sys, re, shutil
from pathlib import Path

# import seaborn as sns
import matplotlib._color_data as mcd
import pyarrow.parquet as pq
import click
import subprocess
import dask.dataframe as dd
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.backends.backend_pdf import PdfPages

import os
import multiprocessing as mp
import pandas as pd
import pyarrow.parquet as pq
from gempyor import config, model_info, outcomes, seir
from multiprocessing import cpu_count
import emcee

from multiprocessing import Pool
from gempyor_logprob import log_prob, compute_likelyhood, check_in_bounds, run_simulation, input_proposal

# disable  operations using the MKL linear algebra.
os.environ["OMP_NUM_THREADS"] = "1"
data_dir = "RSV_USA/"
os.chdir(data_dir)
data_dir = "."  # necessary for now for time serie parameters !!! fixme 

# %%
config_path = f"config_CACOMD_maternal_v8_emcee.yml"
nwalkers = 128
niter = 100
nsamples = 100 # number of likelyhood eval to write to disk...
thin=5

ncpu = cpu_count()
print(f"found {ncpu} CPUs, using these")


# %%

run_id = config_path.split(".")[0]

config.clear()
config.read(user=False)
config.set_file(config_path)
print(config_path)


seir_modifiers_scenario="none"
outcome_modifiers_scenario="IHRadj"

in_run_id = run_id
out_run_id = in_run_id
in_prefix = f"emcee_{run_id}"

out_prefix = in_prefix

write_csv = False
write_parquet = True
modinf = model_info.ModelInfo(
    config=config,
    nslots=1,
    seir_modifiers_scenario=seir_modifiers_scenario,
    outcome_modifiers_scenario=outcome_modifiers_scenario,
    write_csv=write_csv,
    write_parquet=write_parquet,
    first_sim_index=1,
    in_run_id=in_run_id,
    in_prefix=in_prefix,
    inference_filename_prefix="no",
    inference_filepath_suffix="no",
    out_run_id=out_run_id,
    out_prefix=out_prefix,
    stoch_traj_flag=False,
)

nsubpop = len(modinf.subpop_struct.subpop_names)
subpop_names = modinf.subpop_struct.subpop_names

# %%
# find what to perturb
fitted_params = {
    "ptype":[],
    "pname":[],
    "subpop":[],
    "pdist":[],
    "ub":[],
    "lb":[],
}
ndim=0

print(f"there are {nsubpop} subpop in the config")

print("SEIR MODIFIERS")
for npi in gempyor.config["seir_modifiers"]["modifiers"].get():
    if gempyor.config["seir_modifiers"]["modifiers"][npi]["perturbation"].exists():
        c = config["seir_modifiers"]["modifiers"][npi]
        for sp in modinf.subpop_struct.subpop_names:
            fitted_params["ptype"].append("snpi")
            fitted_params["pname"].append(npi)
            fitted_params["subpop"].append(sp)
            fitted_params["pdist"].append(c["value"].as_random_distribution())
            fitted_params["lb"].append(c["value"]["a"].get())
            fitted_params["ub"].append(c["value"]["b"].get())
            ndim+=1
        print(f" >> {npi} has perturbation, recording")
        
print("OUTCOMES MODIFIERS")   
for npi in gempyor.config["outcome_modifiers"]["modifiers"].get():
    if gempyor.config["outcome_modifiers"]["modifiers"][npi]["perturbation"].exists():
        c = config["outcome_modifiers"]["modifiers"][npi]
        for sp in modinf.subpop_struct.subpop_names:
            fitted_params["ptype"].append("hnpi")
            fitted_params["pname"].append(npi)
            fitted_params["subpop"].append(sp)
            fitted_params["pdist"].append(c["value"].as_random_distribution())
            fitted_params["lb"].append(c["value"]["a"].get())
            fitted_params["ub"].append(c["value"]["b"].get())
            ndim+=1
        print(f" >> {npi} has perturbation, recording")


# TODO: does not support the subpop groups here !!!!!!!
print(f"The dimension of the parameter space is {ndim}!!")

# %%
# Find the ground-truth
gt = pd.read_csv(f"{data_dir}/"+gempyor.config["inference"]["gt_data_path"].get())
# gt
statistics = {}
# # Ingoring agreegation and all, assuming by weekP
for stat in gempyor.config["inference"]["statistics"]:
    statistics[gempyor.config["inference"]["statistics"][stat]["sim_var"].get()] = gempyor.config["inference"]["statistics"][stat]["data_var"].get()
statistics
gt = gempyor.read_df(gempyor.config["inference"]["gt_data_path"].get())
gt["date"]= pd.to_datetime(gt['date'])
gt = gt.set_index("date")

# %%


# %% [markdown]
# ## Create the first gempyor object

# %%

print("MAKING A TEST RUN TO GET SETUP")
(
    unique_strings,
    transition_array,
    proportion_array,
    proportion_info,
) = modinf.compartments.get_transition_array()

outcomes_parameters = outcomes.read_parameters_from_config(modinf)


npi_seir = seir.build_npi_SEIR(
    modinf=modinf, load_ID=False, sim_id2load=None, config=config
)
if modinf.npi_config_outcomes:
    npi_outcomes = outcomes.build_outcome_modifiers(
                modinf=modinf,
                load_ID=False,
                sim_id2load=None,
                config=config,
            )

p_draw = modinf.parameters.parameters_quick_draw(
                n_days=modinf.n_days, nsubpops=modinf.nsubpops
            )

initial_conditions = modinf.initial_conditions.get_from_config(sim_id=0, setup=modinf)
seeding_data, seeding_amounts = modinf.seeding.get_from_config(sim_id=0, setup=modinf)



# reduce them
parameters = modinf.parameters.parameters_reduce(p_draw, npi_seir)
        # Parse them
parsed_parameters = modinf.compartments.parse_parameters(
    parameters, modinf.parameters.pnames, unique_strings
)

states = seir.steps_SEIR(
    modinf,
    parsed_parameters,
    transition_array,
    proportion_array,
    proportion_info,
    initial_conditions,
    seeding_data,
    seeding_amounts,
)

seir_out_df_ref = seir.postprocess_and_write(
    0, modinf, states, p_draw, npi_seir, seeding_data
)
snpi_df_ref = npi_seir.getReductionDF()

outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
    modinf=modinf,
    sim_id2write=0,
    parameters=outcomes_parameters,
    loaded_values=None,
    npi=npi_outcomes,
    bypass_seir=seir_out_df_ref
)
outcomes_df_ref, hpar_ref, hnpi_df_ref = outcomes.postprocess_and_write(
    sim_id=0,
    modinf=modinf,
    outcomes_df=outcomes_df,
    hpar=hpar_df,
    npi=npi_outcomes,
)
outcomes_df_ref["time"] = outcomes_df_ref["date"] #which one should it be ?
modinf.write_simID(ftype="hosp", sim_id=0, df=outcomes_df_ref)
outcomes_df_ref = outcomes_df_ref.set_index("date")
print("TEST RUN IS DONE")


# %%
# need to convert to numba dict to python dict so it is pickable
seeding_data = dict(seeding_data)

# %%

# find the initial point from the chain:
p0 = np.zeros((nwalkers, ndim))
for i in range(ndim):
    p0[:,i] = fitted_params["pdist"][i](nwalkers)
    # DOES not take subpop groups into consideration !!!!!!


print(f"initial run llik {compute_likelyhood(outcomes_df_ref, gt, modinf, statistics)}")


# %%
for i in range(nwalkers):
    assert check_in_bounds(p0[i], fitted_params)

# %%


# %%
filename = f"{run_id}_backend.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool(ncpu) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    args=[snpi_df_ref, ndim, statistics, fitted_params, gt, hnpi_df_ref, modinf, p_draw, unique_strings, transition_array, proportion_array, proportion_info, initial_conditions, seeding_data, seeding_amounts,outcomes_parameters, False], 
                                    pool=pool,
                                    backend=backend, moves=[(emcee.moves.StretchMove(live_dangerously=True), 1)])
    state = sampler.run_mcmc(p0, niter, progress=True, skip_initial_state_check=True)

print("done emcee, doing sampling")
# %%

last_llik = sampler.get_log_prob()[-1,:]
good_slots = last_llik > (last_llik.mean()-1*last_llik.std())
print(f"there are {good_slots.sum()}/{len(good_slots)} good walkers... keeping these")

fig, axes = plt.subplots(ndim+1,2, figsize=(15, (ndim+1)*2))

labels = list(zip(fitted_params["pname"], fitted_params["subpop"]))
samples = sampler.get_chain()

import seaborn as sns
def plot_chain(frompt,axes):
    ax = axes[0]

    ax.plot(np.arange(frompt,frompt+sampler.get_log_prob()[frompt:].shape[0]),
                      sampler.get_log_prob()[frompt:,good_slots], "navy", alpha=.2, lw=1, label="good walkers")
    ax.plot(np.arange(frompt,frompt+sampler.get_log_prob()[frompt:].shape[0]),
            sampler.get_log_prob()[frompt:,~good_slots], "tomato", alpha=.4, lw=1, label="bad walkers")
    ax.set_title("llik")
    #ax.legend()
    sns.despine(ax=ax, trim=False)
    ax.set_xlim(frompt, frompt+sampler.get_log_prob()[frompt:].shape[0])

    #ax.set_xlim(0, len(samples))

    for i in range(ndim):
        ax = axes[i+1]
        ax.plot(np.arange(frompt,frompt+sampler.get_log_prob()[frompt:].shape[0]),
                samples[frompt:,good_slots, i], "navy", alpha=.2, lw=1,)
        ax.plot(np.arange(frompt,frompt+sampler.get_log_prob()[frompt:].shape[0]),
                samples[frompt:, ~good_slots, i], "tomato", alpha=.4, lw=1,)
        #ax.set_xlim(0, len(samples))
        ax.set_title(labels[i])
        #ax.yaxis.set_label_coords(-0.1, 0.5)
        sns.despine(ax=ax, trim=False)
        ax.set_xlim(frompt, frompt+samples[frompt:].shape[0])
        

    axes[-1].set_xlabel("step number");

plot_chain(0,axes[:,0])
plot_chain(3*samples.shape[0]//4,axes[:,1])
fig.tight_layout()

plt.savefig(f"{run_id}_chains.pdf")

good_samples =  sampler.get_chain()[:,good_slots,:]

step_number = -1
exported_samples = np.empty((nsamples,ndim))
for i in range(nsamples):
    exported_samples[i,:] = good_samples[step_number - thin*(i//(good_slots.sum())) ,i%(good_slots.sum()),:] # parentesis around i//(good_slots.sum() are very important

position_arguments = [snpi_df_ref, ndim, statistics, fitted_params, gt, hnpi_df_ref, modinf, p_draw, unique_strings, transition_array, proportion_array, proportion_info, initial_conditions, seeding_data, seeding_amounts,outcomes_parameters, True]

with Pool(ncpu) as pool:
    results = pool.starmap(log_prob, [(sample, *position_arguments) for sample in exported_samples])


results = []
for fn in gempyor.utils.list_filenames(folder="model_output/", filters=[run_id,"hosp.parquet"]):
    if "000000000" not in fn:
        df = gempyor.read_df(fn)
        #raise ValueError
        df["date"] = df["time"]
        df = df.set_index("date")
        results.append(df)

fig, axes = plt.subplots(len(statistics),len(subpop_names), figsize=(5*len(subpop_names), 3*len(statistics)), sharex=True, dpi=300)
for j, subpop in enumerate(modinf.subpop_struct.subpop_names):
        gt_s = gt[gt["subpop"]==subpop].sort_index()
        first_date = max(gt_s.index.min(),results[0].index.min())
        last_date = min(gt_s.index.max(), results[0].index.max())
        gt_s = gt_s.loc[first_date:last_date].drop(["subpop"],axis=1).resample("W-SAT").sum()
        
        for i, (key, value) in enumerate(statistics.items()):
                ax = axes[i,j]
                ax.plot(gt_s[value], color='k', marker='.', lw=1)
                for model_df in results:
                        model_df_s = model_df[model_df["subpop"]==subpop].drop(["subpop","time"],axis=1).loc[first_date:last_date].resample("W-SAT").sum() # todo sub subpop here
                        ax.plot(model_df_s[key],  lw=.9, alpha=.5)
                #if True:
                #        init_df_s = outcomes_df_ref[model_df["subpop"]==subpop].drop(["subpop","time"],axis=1).loc[min(gt_s.index):max(gt_s.index)].resample("W-SAT").sum() # todo sub subpop here
                ax.set_title(f"{value}, {subpop}")
fig.tight_layout()
plt.savefig(f"{run_id}_results.pdf")