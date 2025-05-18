from snakemake.script import snakemake
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()

raw_data = pd.read_pickle(snakemake.input.raw_data)
ap_track_window = pd.read_pickle(snakemake.input.ap_track_window)
ap_derivatives = pd.read_pickle(snakemake.input.ap_derivatives)
ap_window_iloc = pd.read_pickle(snakemake.input.ap_window_iloc)
spikes = pd.read_pickle(snakemake.input.spikes)

# color list to extract colors 
colors = ["tab:blue", "tab:green","tab:orange", "tab:red","tab:cyan", "tab:brown", "tab:pink", "tab:olive"]
track_names = sorted(ap_track_window['track'].unique())
# if error, new colors needs to be added
assert len(track_names) < len(colors)
track_colors = {track_names[i]:colors[i] for i in range(len(track_names))}

# find threshold bounds of templates, which are created by taking the mean of spikes per track
# current filter computes the range [min(template) - (min(template) * 0.3, min(template) + (min(template) * 0.3]
# so it checked if the negative peak is inside this bound 
def find_threshold_bounds(ap_track_window):
    ap_templates = pd.DataFrame()
    thresholds_template = {}
    for track in ap_track_window['track'].unique():
        ap_track_window_sorted = ap_track_window[(ap_track_window['track'] == track)]
        ap_raw = ap_track_window_sorted[['start_iloc','end_iloc']]\
            .progress_apply(extract_raw_values, args=(raw_data,), axis=1, result_type='expand')
        template = ap_raw.mean()
        thresholds_template[track] = [round(min(template) - (min(template) * 0.3),2)]
        thresholds_template[track].append(round(min(template) + (min(template) * 0.3),2))
        data = pd.DataFrame({"track":track, "template" :[template.to_list()]})
        ap_templates = pd.concat([ap_templates,data])

    return ap_templates.reset_index(drop=True), thresholds_template

# extract raw values
def extract_raw_values(row, data:pd.Series, idx_window_start_iloc=0, idx_window_end_iloc=1):
    start = row[idx_window_start_iloc]
    end = row[idx_window_end_iloc]
    raw = data.iloc[start:end]
    return raw.to_list()

# helper function to filter by threshold
def filter_by_threshold(row, thresholds):
    raw = row.raw
    min_raw = min(raw)
    return round(min_raw,2) <= thresholds[row.track][0] and round(min_raw,2) >= thresholds[row.track][1]

# helper function to compute the deviation
def compute_deviation(spike, template):
    deviation = np.linalg.norm(template - spike)
    return deviation

# filte action potentials based on template deviation
def filter_action_potentials(row, df_templates, threshold_percentage=90):
    track = row["track"]
    spike = row["raw"]
    template = ap_templates.loc[ap_templates["track"] == track]["template"].values[0]
    template_deviation = np.linalg.norm(template)  
    deviation = compute_deviation(spike, template)
    deviation_percentage = (deviation / template_deviation) * 100
    if deviation_percentage <= threshold_percentage:
        return True
    else:
        return False
    
# plot tempaltes of tracks
def plot_template(row, ax1, track_colors):
    template = pd.Series(row["template"], index=[round(x, 4) for x in np.arange(0, len(row["template"]) * 0.0001, 0.0001)])
    ax1.plot([round(x, 4) for x in np.arange(0.0001, (len(template) * 0.0001 + 0.0001), 0.0001)], template.values,
         c=track_colors[row.track], alpha=1, linestyle="dashed",
         label="Template of " + row.track)
    ax1.legend(fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=20)
    ax1.set_ylabel(u'${\mu}V$', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=18)
    ax1.yaxis.set_tick_params(labelsize=18)

def ax_plot(ax, *ax_args, **ax_kv_args):
    return ax.plot(*ax_args, **ax_kv_args)

def ax_scatr(ax, *ax_args, **ax_kv_args):
    return ax.scatter(*ax_args, **ax_kv_args)

# plot süikes
def plot_ap_shape(df: pd.DataFrame, raw_data:pd.DataFrame, style="scatter",color_dict=track_colors,type='single', templates=None, template2=None, accent_color="green"):
    if accent_color is None:
        accent_color = color_dict['marker']
    if style == 'plot':
        plot_func = ax_plot
    elif style == 'scatter':
        plot_func = ax_scatr
    else:
        raise Exception(f"Unknown style {style}")
    if type == 'single':
        fig,ax = plt.subplots(1,1)
        waveforms = []
        tracks = []
        legend_labels = []
        for track in df['track'].unique():
            label_args = {'label':f"{track}"}
            for row,r in df[(df['track'] == track)][['start_iloc','end_iloc']].iterrows():
                ax.set_title(row)
                waveforms.append(raw_data.iloc[r[0]:r[1]].to_numpy())
                tracks.append(track) 
                if not snakemake.params.use_bristol_processing:
                    plot_func(ax,np.arange(0, 3, 0.1),raw_data.iloc[r[0]:r[1]].to_numpy(), color=color_dict[track], alpha=0.4,**label_args) # , alpha=0.2
                else:
                    plot_func(ax,np.arange(0, 3, 0.1/2),raw_data.iloc[r[0]:r[1]].to_numpy(), color=color_dict[track], alpha=0.4,**label_args) # , alpha=0.2
                label_args = {}
            if templates is not None:
                ax.plot(templates[track].to_numpy(), color=accent_color, label="template")
            if template2 is not None:
                template_track = template2.loc[template2['track'] == track]
                if not snakemake.params.use_bristol_processing:
                    ax.plot(np.arange(0, 3, 0.1),template_track["template"].values[0], color="blue", label="Template computed" if "Template computed" not in legend_labels else "")
                else:
                    ax.plot(np.arange(0, 3, 0.1/2),template_track["template"].values[0], color="blue", label="Template computed" if "Template computed" not in legend_labels else "")

                legend_labels.append("Template computed")
            ax.set_title(f"{track} action potentials")
            ax.set_title("Waveforms of action potentials")
            ax.legend(loc = 1)
            ax.set_xlabel(f"Time (ms)")
            ax.set_ylabel(f"Voltage")
            fig.tight_layout()
    elif type == 'all':
        fig,ax = plt.subplots(1,1)
        label_args = {'label':"waveforms"}
        for _,r in df[['start_iloc','end_iloc']].iterrows():
            plot_func(ax,raw_data.iloc[r[0]:r[1]].to_numpy(), color=accent_color, **label_args)
            label_args = {}
        if templates is not None:
            for track, template in templates.items():
                ax.plot(template.to_numpy(), color=color_dict[track], label=f"{track} template")
        ax.set_title("all waveforms")
        ax.legend()
        ax.set_xlabel(f"datapoint")
        ax.set_ylabel(f"amplitude")
        fig.tight_layout()
    fig.savefig(snakemake.output.waveform_figure, dpi=300)

ap_templates, thresholds_template = find_threshold_bounds(ap_track_window)

# add column, if spike shold be droped
ap_track_window["drop_min"] = ap_track_window\
        .join(ap_derivatives)\
        .progress_apply(filter_action_potentials, args=(ap_templates,), axis=1, result_type='expand')\

# create list of indices with spikes that should be dropped
# skip fitlering
#drop_index_below_thresholds = ap_track_window.index[ap_track_window['drop_min'] == False].tolist()


# plot templates
fig = plt.figure()
plt.rcParams["font.size"] = 10
ax1 = fig.add_subplot(111)
ap_templates.progress_apply(plot_template, args=(ax1,track_colors), axis = 1 , result_type="expand")
fig.tight_layout()

# plot raw waveforms
templates = None
ap_window_iloc\
    .join(spikes[['track']])\
    .pipe(plot_ap_shape, raw_data, templates=templates, template2=ap_templates, style="plot",)

fig.savefig(snakemake.output.template_figure, dpi=300)
ap_templates.to_pickle(snakemake.output.ap_templates)
ap_track_window.to_pickle(snakemake.output.ap_track_window_m)