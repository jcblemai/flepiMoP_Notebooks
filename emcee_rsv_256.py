import gempyor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob, os, sys, re, shutil
from pathlib import Path
import copy

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
import xarray as xr
from gempyor import config, model_info, outcomes, seir, inference_parameter, logloss, inference

import os
from multiprocessing import cpu_count

# disable  operations using the MKL linear algebra.
os.environ["OMP_NUM_THREADS"] = "1"
import emcee

from multiprocessing import Pool

data_dir = "COVID19_USA/"
config_path= "config_SMH_R18_allBoo_highIE_blk1_emcee.yml"
seir_modifiers_scenario="inference"
outcome_modifiers_scenario="all"

nwalkers = 1600
niter = 200
nsamples = 200 # number of likelyhood eval to write to disk...
thin=5

ncpu = cpu_count()
ncpu=256  # too much ???? TS
print(f"found {ncpu} CPUs, using these")


run_id = config_path.split(".")[0]

config.clear()
config.read(user=False)
config.set_file(data_dir+config_path)
# same but in a Path safe way
config.set_file
print(config_path)


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
    path_prefix=data_dir,
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

inferpar = inference_parameter.InferenceParameters(global_config=config, modinf=modinf)
p0 = inferpar.draw_initial(n_draw=nwalkers)
for i in range(nwalkers):
    assert inferpar.check_in_bound(proposal=p0[i]), "The initial parameter draw is not within the bounds, check the perturbation distributions"

loss = logloss.LogLoss(inference_config=config["inference"], data_dir=data_dir, modinf=modinf)

os.makedirs("COVID19_USA/model_output/emcee_config_SMH_R18_allBoo_highIE_blk1_emcee/hosp/no/", exist_ok=True)
os.makedirs("COVID19_USA/model_output/emcee_config_SMH_R18_allBoo_highIE_blk1_emcee/hpar/no/", exist_ok=True)
os.makedirs("COVID19_USA/model_output/emcee_config_SMH_R18_allBoo_highIE_blk1_emcee/hnpi/no/", exist_ok=True)
static_sim_arguments = gempyor.inference.get_static_arguments(modinf=modinf)

print(loss)
print(inferpar)

test_run = True
if test_run:
    ss = copy.deepcopy(static_sim_arguments)

    ss["snpi_df_in"] = ss["snpi_df_ref"]
    ss["hnpi_df_in"] = ss["hnpi_df_ref"]
    # delete the ref
    del ss["snpi_df_ref"]
    del ss["hnpi_df_ref"]

    hosp = gempyor.inference.simulation_atomic(**ss, modinf=modinf)

    ll_total, logloss, regularizations = loss.compute_logloss(model_df=hosp, modinf=modinf)
    print(f"test run successful 🎉, with logloss={ll_total:.1f} including {regularizations:.1f} for regularization ({regularizations/ll_total*100:.1f}%) ")

gempyor.inference.emcee_logprob(p0[0], modinf, inferpar, loss, static_sim_arguments, save=False)

# %%
filename = f"{run_id}_backend.h5"
backend = emcee.backends.HDFBackend(filename)

if True:
    backend.reset(nwalkers, inferpar.get_dim())
    p0=p0
else:
    p0 = None

with Pool(ncpu) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, 
                                    inferpar.get_dim(), 
                                    gempyor.inference.emcee_logprob,
                                    args=[modinf, inferpar, loss, static_sim_arguments], 
                                    pool=pool,
                                    backend=backend, moves=[(emcee.moves.StretchMove(live_dangerously=True), 1)])
    state = sampler.run_mcmc(p0, niter, progress=True, skip_initial_state_check=True)

print("done emcee, doing sampling")
# %%

last_llik = sampler.get_log_prob()[-1,:]
good_slots = last_llik > (last_llik.mean()-1*last_llik.std())
print(f"there are {good_slots.sum()}/{len(good_slots)} good walkers... keeping these")

fig, axes = plt.subplots(inferpar.get_dim()+1,2, figsize=(15, (inferpar.get_dim()+1)*2))

labels = list(zip(inferpar.pnames, inferpar.subpops))
samples = sampler.get_chain()
p_gt = np.load("parameter_ground_truth.npy")

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

    for i in range(inferpar.get_dim()):
        ax = axes[i+1]
        x_plt = np.arange(frompt,frompt+sampler.get_log_prob()[frompt:].shape[0])
        ax.plot(x_plt,
                samples[frompt:,good_slots, i], "navy", alpha=.2, lw=1,)
        ax.plot(x_plt,
                samples[frompt:, ~good_slots, i], "tomato", alpha=.4, lw=1,)
        ax.plot(x_plt,
                np.repeat(p_gt[i],len(x_plt)), "black", alpha=1, lw=2, ls='-.')
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
exported_samples = np.empty((nsamples,inferpar.get_dim()))
for i in range(nsamples):
    exported_samples[i,:] = good_samples[step_number - thin*(i//(good_slots.sum())) ,i%(good_slots.sum()),:] # parentesis around i//(good_slots.sum() are very important

position_arguments = [modinf, inferpar, loss, static_sim_arguments, True]
shutil.rmtree("model_output/")
with Pool(ncpu) as pool:
    results = pool.starmap(gempyor.inference.emcee_logprob, [(sample, *position_arguments) for sample in exported_samples])



results = []
for fn in gempyor.utils.list_filenames(folder="model_output/", filters=[run_id,"hosp.parquet"]):
        df = gempyor.read_df(fn)
        df = df.set_index("date")
        results.append(df)

fig, axes = plt.subplots(len(subpop_names),len(loss.statistics), figsize=(3*len(loss.statistics), 3*len(subpop_names)), sharex=True)
for j, subpop in enumerate(modinf.subpop_struct.subpop_names):
        gt_s = loss.gt[loss.gt["subpop"]==subpop].sort_index()
        first_date = max(gt_s.index.min(),results[0].index.min())
        last_date = min(gt_s.index.max(), results[0].index.max())
        gt_s = gt_s.loc[first_date:last_date].drop(["subpop"],axis=1).resample("W-SAT").sum()
        
        for i, (stat_name, stat) in enumerate(loss.statistics.items()):
                ax = axes[j,i]
                
                ax.plot(gt_s[stat.data_var], color='k', marker='.', lw=1)
                for model_df in results:
                        model_df_s = model_df[model_df["subpop"]==subpop].drop(["subpop"],axis=1).loc[first_date:last_date].resample("W-SAT").sum() # todo sub subpop here
                        ax.plot(model_df_s[stat.sim_var],  lw=.9, alpha=.5)
                #if True:
                #        init_df_s = outcomes_df_ref[model_df["subpop"]==subpop].drop(["subpop","time"],axis=1).loc[min(gt_s.index):max(gt_s.index)].resample("W-SAT").sum() # todo sub subpop here
                ax.set_title(f"{stat_name}, {subpop}")
fig.tight_layout()
plt.savefig(f"{run_id}_results.pdf")
