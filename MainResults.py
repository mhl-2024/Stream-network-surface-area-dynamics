# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import scipy.stats as stats
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

def Maps(basin,us_bound=None):
    plotdf = basin.copy()
    # Settings for each variable
    settings = {
        "gamma": {
            "bins": [0.30, 0.36, 0.42, 0.48, 0.54],
            "name": r"Surface area elasticity $\gamma$",
            "cmap": "cividis",
            "fmt": "%.2f"
        },
        "CV_A": {
            "bins": [0.30, 0.47, 0.64, 0.81, 0.98],
            "name": "Coefficient of variation of surface area",
            "cmap": "cividis",
            "fmt": "%.2f"
        },
        "beta": {
            "bins": [0.15, 0.25, 0.35, 0.45, 0.55],
            "name": r"Network length elasticity $\beta$",
            "cmap": "viridis",
            "fmt": "%.2f"
        },
        "alpha": {
            "bins": [-0.10, -0.04, 0.02, 0.08, 0.14],
            "name": "Network-averaged stream\nwidth elasticity " + r"$\alpha$",
            "cmap": "viridis",
            "fmt": "%.2f"
        },
        
        "CV_L": {
            "bins": [0.30, 0.47, 0.64, 0.81, 0.98],
            "name": "Coefficient of\nvariation of network length",
            "cmap": "viridis",
            "fmt": "%.2f"
        },
        
        
        "rogh_dg": {
            "bins": [1.00, 3.20, 5.40, 7.60, 9.80],
            "name": "Topographic slope [°]",
            "cmap": "viridis_r",
            "fmt": "%.2f"
        },
        "AI": {
            "bins": [0.21, 0.42, 0.63, 0.84, 1.05],
            "name": "Aridity index = P/PET",
            "cmap": "viridis_r",
            "fmt": "%.2f"
        },
        
        "CV_Q": {
            "bins": [1.10, 2.10, 3.10, 4.10, 5.10],
            "name": "Coefficient of\nvariation of discharge",
            "cmap": "viridis",
            "fmt": "%.2f"
        }
    }
    
    plot_vars = ["gamma","CV_A","beta","alpha","CV_L","rogh_dg","AI","CV_Q"]

    panel_labels = ["a", "b", "c", "d", "e", "f", "g","h"]
    
    fontsize0=24
    fig = plt.figure(figsize=(18, 15))
    axes_positions = [
        [0.03, 0.69, 0.5, 0.3],  # a
        [0.52, 0.69, 0.5, 0.3],  # b

        [0.02, 0.35, 0.30, 0.23],  # c
        [0.35, 0.35, 0.30, 0.23],  # d
        [0.68, 0.35, 0.30, 0.23],  # e

        [0.02, 0.02, 0.30, 0.23],  # f
        [0.35, 0.02, 0.30, 0.23],   # g
        [0.68, 0.02, 0.30, 0.23],  # h
    ]

    axes = [
        fig.add_axes(position)
        for position in axes_positions
    ]

    xmin, ymin, xmax, ymax = plotdf.total_bounds
  
    # Plot each panel
    for ax, plot_var, panel_label in zip(axes,plot_vars,panel_labels):

        cfg = settings[plot_var]

        bins = cfg["bins"]
        cmap = plt.get_cmap(cfg["cmap"])

        norm = mpl.colors.BoundaryNorm(bins,ncolors=cmap.N,extend="both")

        # Map
        plotdf.plot(column=plot_var,ax=ax,cmap=cmap,norm=norm,linewidth=0.1,edgecolor="white")
        
        if us_bound is not None:
            us_bound.plot(ax=ax, color="none",edgecolor="black",linewidth=1.2, zorder=4)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()

        # Panel label
        ax.text(-0.01,1.1,panel_label,transform=ax.transAxes,fontsize=fontsize0 + 5,fontweight="bold",ha="left",va="top",clip_on=False)

        # Colorbar above each map
        cax = inset_axes(
            ax,
            width="70%",
            height="5%",
            loc="upper center",
            bbox_to_anchor=(0, 0.04, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        sm = mpl.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm,cax=cax,orientation="horizontal",ticks=bins,extend="both")

        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        
        cbar.set_label( cfg["name"],fontsize=fontsize0+2,labelpad=8)

        cbar.ax.tick_params(axis="x",labelsize=fontsize0,length=3,width=1,pad=2)
        
        if panel_label not in ['a', 'b']:
            cbar.ax.tick_params(axis="x",labelsize=fontsize0-1,length=3,width=1,pad=2)
        cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(cfg["fmt"]))
        cbar.outline.set_linewidth(1.2)
        
        # if plot_var == "CV_A" or plot_var == "CV_Q":
        #     scalebar = ScaleBar(
        #         dx=1,                  
        #         units="m",
        #         fixed_value=1000,
        #         fixed_units="km",
        #         length_fraction=0.20,  
        #         location="lower left",
        #         scale_loc="bottom",
        #         box_alpha=0.0,          
        #         color="black",
        #         font_properties={"size": fontsize0}
        # )
        #     ax.add_artist(scalebar)

    return fig


def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def binned_plot_sem_equal_num(df, x_var, y_var, n_bins=10,  ax=None):

    data = df[[x_var, y_var]].dropna().copy()
    data['bin'] = pd.qcut(data[x_var], q=n_bins, duplicates='drop')

    grouped = data.groupby('bin')
    x_center = grouped[x_var].mean()

    y_stat = grouped[y_var].mean()

    y_std = grouped[y_var].std()
    n = grouped[y_var].count()
    y_sem = y_std / np.sqrt(n)

    return x_center, y_stat, y_sem    


def binned_plot_combine(basin):
    dfplot0 = basin[basin['beta'] <= 1].copy()

    # Variable labels
    var_names = {
        'CV_A': 'CV$_A$',
        'CV_Q': 'CV$_Q$',
        'CV_L': 'CV$_L$',

        'AI': 'Aridity index',
        'rogh_dg': 'Topographic slope [°]',
        'gamma': 'Surface area\nelasticity $\\gamma$',
        'alpha':
            'Network-averaged\n'
            'width elasticity $\\alpha$',
        'beta':
            'Network length\nelasticity $\\beta$'
    }

    # Panel arrangement
    plot_pairs = [
        ('beta',        'gamma'),   # a
        ('beta',        'alpha'),   # b
        ('alpha', 'gamma'),   # c
        ('rogh_dg',     'gamma'),   #d
        ('rogh_dg',     'CV_A'),      # e
        ('AI',          'CV_A')       # f
    ]

    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']

    fontsize0 = 26
    fig, axes = plt.subplots(2, 3,figsize=(20, 10),gridspec_kw={'wspace': 0.55,'hspace': 0.5})

    axes = axes.flatten()
    for ax, (x_var, y_var), panel_label in zip(axes, plot_pairs, panel_labels):   
        # Spearman correlation
        spearman_corr, p_value = stats.spearmanr(dfplot0[x_var], dfplot0[y_var])

        rho_stars = significance_stars(p_value)

        # Binned statistics
        x_center, y_stat, y_sem = binned_plot_sem_equal_num(dfplot0,x_var,y_var,n_bins=15,ax=None)

        # Original points
        ax.scatter(dfplot0[x_var],dfplot0[y_var],c='lightgrey',s=8,edgecolors='none',alpha=0.55,rasterized=True,zorder=1)

        # Binned means
        ax.scatter(x_center,y_stat,marker='o',c='orange',s=150,edgecolors='black',linewidths=0.8,zorder=3)

        # Axis scales
        if x_var in ['CV_A', 'CV_Q', 'CV_L','AI', 'rogh_dg']:
            ax.set_xscale('log')

        if y_var in ['CV_A', 'CV_Q', 'CV_L']:
            ax.set_yscale('log')

        # 1:1 line for panel A
        if x_var == 'beta' and y_var == 'gamma':

            min_val = max(dfplot0[x_var].min(),dfplot0[y_var].min())
            max_val = min(dfplot0[x_var].max(),dfplot0[y_var].max())

            ax.plot([min_val, max_val],[min_val, max_val],'k--',linewidth=1.5,zorder=2)

            ax.text(0.55, 0.27,'1:1 line', transform=ax.transAxes,fontsize=fontsize0 )
            rho_x, rho_y = 0.50, 0.05
        else:
            rho_x, rho_y = 0.05, 0.05

        # Correlation annotation
        rho_text = f'ρ={spearman_corr:.2f}'

        t_rho = ax.text(rho_x,rho_y,rho_text,fontsize=fontsize0,color='black',transform=ax.transAxes,ha='left',va='bottom')

        fig.canvas.draw()
        bbox = t_rho.get_window_extent(renderer=fig.canvas.get_renderer())
        star_x = ax.transAxes.inverted().transform( (bbox.x1, bbox.y1))[0]

        ax.text(
            star_x,
            rho_y + 0.015,
            rho_stars,
            fontsize=fontsize0,
            fontweight='bold',
            color='black',
            transform=ax.transAxes,
            ha='left',
            va='bottom'
        )

        ax.set_xlabel(var_names[x_var],fontsize=fontsize0 ,labelpad=5)
        ax.set_ylabel(var_names[y_var],fontsize=fontsize0 ,labelpad=5)

        ax.text(
            0.03,
            0.97,
            panel_label,
            transform=ax.transAxes,
            fontsize=fontsize0 +4,
            fontweight='bold',
            ha='left',
            va='top'
        )
        
        if panel_label == 'b':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
      
        ax.tick_params(axis='both', which='major',direction='out', width=1.2,length=5, pad=5,labelsize=fontsize0 )
        ax.tick_params(axis='both',which='minor',direction='out',width=1, length=3)

        for spine in ['bottom', 'left', 'right', 'top']:
            ax.spines[spine].set_linewidth(1.3)

    return fig

def plot_CONUS_SA_map(basin,CONUS_SA,sumA_com_con_meanQ,us_bound=None):

    fontsize0 = 25
    plt.rcParams.update({
        "font.size": fontsize0,
        "axes.labelsize": fontsize0,
        "axes.titlesize": fontsize0,
        "xtick.labelsize": fontsize0,
        "ytick.labelsize": fontsize0,
        "legend.fontsize": fontsize0
    })


    cmap = plt.cm.twilight
    norm = mcolors.BoundaryNorm(np.arange(1, 14), cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(16, 14))

    # Panel A: CONUS surface area ~ day of year
    ax_SA = fig.add_axes([0.17, 0.67, 0.72, 0.33]) 

    # Row 2: maps
    ax_map_max = fig.add_axes([0.04, 0.24, 0.44, 0.35])
    ax_map_min = fig.add_axes([0.52, 0.24, 0.44, 0.35])

    # Row 3: histograms
    ax_hist_max = fig.add_axes([0.08, 0.125, 0.37, 0.10])
    ax_hist_min = fig.add_axes([0.555, 0.125, 0.37, 0.10])

    # Bottom: colorbars
    ax_cbar_max = fig.add_axes([0.08, 0.1, 0.37, 0.014])
    ax_cbar_min = fig.add_axes([0.555, 0.1, 0.37, 0.014])
    
    

    # panel A
    ax_SA.fill_between( CONUS_SA["doy"], CONUS_SA["p10"],CONUS_SA["p90"],color="#8BC4E8",alpha=0.4,label="10%–90% range")
    ax_SA.axhline(sumA_com_con_meanQ,label='sum area_mean flow',color='#800f0f',linestyle='--',linewidth=3)
    ax_SA.plot( CONUS_SA["doy"],CONUS_SA["mean"],label="Mean",linewidth=2.5,color="#0471B7")
    ax_SA.set_xlim(CONUS_SA["doy"].min(), CONUS_SA["doy"].max())
    ax_SA.xaxis.set_major_locator(mdates.MonthLocator())

    ax_SA.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: str(mdates.num2date(x).month)
        )
    )

    ax_SA.tick_params(axis="both",which="major",direction="out",length=5,width=1.5,pad=8,labelsize=fontsize0)
    ax_SA.tick_params(which="minor", direction="out",length=3,width=1.5)

    for spine in ax_SA.spines.values():
        spine.set_linewidth(1.5)

    ax_SA.ticklabel_format(style="sci",axis="y",scilimits=(0, 0))

    ax_SA.set_ylabel("Total stream network\nsurface area [km$^2$]",fontsize=fontsize0)
    ax_SA.set_xlabel("Month",fontsize=fontsize0)

    # Panel label A
    ax_SA.text(
        -0.075,
        1.0,
        "a",
        transform=ax_SA.transAxes,
        fontsize=fontsize0 + 6,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # B–C. Maps, histograms and colorbars
    map_axes = [ax_map_max, ax_map_min]
    hist_axes = [ax_hist_max, ax_hist_min]
    cbar_axes = [ax_cbar_max, ax_cbar_min]

    month_fields = ["maxSAmonth", "minSAmonth"]

    plot_names = [
        "Timing of maximum surface area [month]",
        "Timing of minimum surface area [month]"
    ]

    panel_labels = ["b", "c"]
    xmin, ymin, xmax, ymax = basin.total_bounds
    months = np.arange(1, 13)

    for ax, hax, cax, month_field, plot_name, panel_label in zip(map_axes,hist_axes,cbar_axes,month_fields,plot_names,panel_labels):
        # Map
        basin.plot(column=month_field,
            cmap=cmap,
            norm=norm,
            linewidth=0.1,
            edgecolor="white",
            ax=ax
        )
        if us_bound is not None:
            us_bound.plot(
                ax=ax,
                color="none",
                edgecolor="black",
                linewidth=1.5,
                zorder=4
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        ax.set_zorder(1)

        # Panel labels B and C
        ax.text(
            -0.02,
            1,
            panel_label,
            transform=ax.transAxes,
            fontsize=fontsize0 + 6,
            fontweight="bold",
            va="top",
            ha="left",
            zorder=20
        )
        
        # scale bar
        # if month_field == "maxSAmonth":
        #     scalebar = ScaleBar(
        #         dx=1,
        #         units="m",
        #         fixed_value=1000,
        #         fixed_units="km",
        #         length_fraction=0.20,
        #         location="lower left",
        #         scale_loc="bottom",
        #         box_alpha=0,
        #         frameon=False,
        #         color="black",
        #         font_properties={"size": fontsize0}
        #     )

        #     ax.add_artist(scalebar)

        # Histogram
        vals = basin[month_field].astype(int)

        counts = (
            vals.value_counts()
            .reindex(months, fill_value=0)
            .sort_index()
        )

        freq = counts / counts.sum()
        bar_colors = cmap(norm(months))

        hax.bar(
            months,
            freq.values,
            color=bar_colors,
            edgecolor="black",
            linewidth=1,
            width=0.8
        )

        hax.set_xlim(0.5, 12.5)
        hax.set_xticks([])
        hax.set_ylabel("Frequency", fontsize=fontsize0)

        if month_field == "maxSAmonth":
            hax.set_ylim(0, 0.32)
            hax.set_yticks([0, 0.1, 0.2, 0.3])

        else:
            hax.set_ylim(0, 0.45)
            hax.set_yticks([0, 0.2, 0.4])

        hax.tick_params(
            axis="y",
            labelsize=fontsize0,
            length=4,
            width=1.5,
            direction="out"
        )

        hax.spines["left"].set_linewidth(1.5)

        for spine in ["top", "right", "bottom"]:
            hax.spines[spine].set_visible(False)

        hax.set_facecolor("white")
        hax.patch.set_alpha(1)
        hax.set_zorder(10)

        # Colorbar
        cbar = fig.colorbar(sm,cax=cax,orientation="horizontal",ticks=months + 0.5)

        cbar.set_ticklabels([str(i) for i in months],fontsize=fontsize0)
        cbar.set_label(plot_name,fontsize=fontsize0,labelpad=8)
        cbar.ax.tick_params(labelsize=fontsize0,length=0,pad=8  )
        cbar.outline.set_linewidth(1.5)

        cax.set_facecolor("white")
        cax.set_zorder(10)

    return fig

def plot_GPP_ER_comparison(GPP_ER, basin, us_bound=None):

    fontsize0 = 22
    df = GPP_ER.copy()
    fig = plt.figure(figsize=(16, 12))
    curve_axes = [
        fig.add_axes([0.09, 0.68, 0.39, 0.28]),
        fig.add_axes([0.57, 0.68, 0.39, 0.28])
    ]

    map_axes = [
        fig.add_axes([0.03, 0.2, 0.46, 0.41]),
        fig.add_axes([0.53, 0.2, 0.46, 0.41])
    ]

    hist_axes = [
        fig.add_axes([0.1, 0.108, 0.34, 0.108]),
        fig.add_axes([0.6, 0.108, 0.34, 0.108])
    ]

    # colorbars
    cbar_axes = [
        fig.add_axes([0.1, 0.078, 0.34, 0.016]),
        fig.add_axes([0.6, 0.078, 0.34, 0.016])
    ]

    # A and B
    curve_config = [
        {
            "maavara": "GPP_static",
            "dynamic": "GPP_dyn",
            "ylabel": "Normalized GPP",
            "clight": "#96D274",
            "cdark": "#39A432",
            "panel": "a"
        },
        {
            "maavara": "ER_static",
            "dynamic": "ER_dyn",
            "ylabel": "Normalized ER",
            "clight": "#F9CB80",
            "cdark": "#F49600",
            "panel": "b"
        }
    ]

    x = np.arange(1, 13)

    for ax, cfg in zip(curve_axes, curve_config):

        y1 = df[cfg["maavara"]]
        y2 = df[cfg["dynamic"]]
        
        # normalize fluxes
        y1 = (y1 - y1.min()) / (y1.max() - y1.min())
        y2 = (y2 - y2.min()) / (y2.max() - y2.min())
        ax.plot(x, y2, linewidth=2, c=cfg["cdark"])
        ax.plot(x, y1, linewidth=2, c=cfg["clight"])

        ax.scatter(
            x, y2,
            s=180,
            marker="s",
            label="This study",
            c=cfg["cdark"],
            edgecolors="black",
            linewidths=1.5
        )
        
        ax.scatter(
            x, y1,
            s=200,
            marker="o",
            label="Maavara et al.",
            c=cfg["clight"],
            edgecolors="black",
            linewidths=1.5
        )


        ax.set_xticks(x)
        ax.set_xlabel("Month", fontsize=fontsize0)
        ax.set_ylabel(cfg["ylabel"], fontsize=fontsize0)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize0,
            direction="in",
            length=6,
            width=1.5,    
            bottom=True,
            left=True
        )
        ax.tick_params(axis="x", pad=6)

        ax.legend(
            fontsize=fontsize0,
            frameon=True,
            handletextpad=0.01,
            borderpad=0.1,
            labelspacing=0.25,
            bbox_to_anchor=(0.46, 0.3)
        )

        ax.text(
            0.02, 0.98,
            cfg["panel"],
            transform=ax.transAxes,
            fontsize=fontsize0+5,
            fontweight="bold",
            va="top",
            ha="left"
        )

    # color scale for C and D
    vals_all = np.arange(-6, 6)

    colors = plt.cm.twilight_shifted(np.linspace(0, 1, len(vals_all)))

    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-6.5, 6.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # C and D
    map_config = [
        {
            "field": "DifGPPmon",
            "xlabel": "Differences in GPP peak month [months]",
            "panel": "c",
            "yticks": [0, 0.2, 0.4]
        },
        {
            "field": "DifERmon",
            "xlabel": "Differences in ER peak month [months]",
            "panel": "d",
            "yticks": [0, 0.1, 0.2]
        }
    ]

    for map_ax, hax, cax, cfg in zip(map_axes,hist_axes,cbar_axes, map_config):

        basin.plot(
            column=cfg["field"],
            cmap=cmap,
            norm=norm,
            linewidth=0.1,
            ax=map_ax
        )
        if us_bound is not None:
            us_bound.plot(
                ax=map_ax,
                color="none",
                edgecolor="black",
                linewidth=1.5,
                zorder=4
            )

        map_ax.set_axis_off()

        map_ax.text(
            0.01, 0.99,
            cfg["panel"],
            transform=map_ax.transAxes,
            fontsize=fontsize0+5,
            fontweight="bold",
            va="top",
            ha="left"
        )
        
        # if cfg["panel"]== "d":
        #     scalebar = ScaleBar(
        #         dx=1,                  
        #         units="m",
        #         fixed_value=1000,
        #         fixed_units="km",
        #         length_fraction=0.20,   
        #         location="lower left",
        #         scale_loc="bottom",
        #         box_alpha=0.0,          
        #         color="black",
        #         border_pad=1.2,
        #         font_properties={"size": fontsize0}
        # )
        #     map_ax.add_artist(scalebar)

        vals = basin[cfg["field"]].dropna().astype(int)

        counts = (
            vals.value_counts()
            .reindex(vals_all, fill_value=0)
        )

        freq = counts / counts.sum()

        hax.bar(
            vals_all,
            freq.values,
            color=cmap(norm(vals_all)),
            linewidth=1.5,
            edgecolor="black",
            alpha=0.9,
            width=0.8
        )

        hax.set_xlim(-6.5, 5.5)
        hax.set_xticks([])
        hax.set_yticks(cfg["yticks"])
        hax.set_ylabel("Frequency", fontsize=fontsize0)

        hax.tick_params(
            axis="y",
            labelsize=fontsize0,
            length=4,
            width=1.5
        )

        hax.spines["left"].set_visible(True)
        hax.spines["left"].set_linewidth(1.5)

        for spine in ["top", "right", "bottom"]:
            hax.spines[spine].set_visible(False)

        cbar = fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            ticks=vals_all
        )

        cbar.set_ticks(vals_all[::2])
        cbar.set_label(cfg["xlabel"], fontsize=fontsize0)
        cbar.ax.tick_params(labelsize=fontsize0, length=0,pad=6)
        cbar.outline.set_linewidth(1.5)

    return fig

def main(data_dir=None,us_boundary_path=None,output_dir=None):
    script_dir = Path(__file__).resolve().parent
    data_dir = Path(data_dir) if data_dir is not None else script_dir / "data"
    
    # load the datasets
    basin = gpd.read_file(data_dir / "Basins.gpkg")
    uc = gpd.read_file(data_dir / "UnitCatchments.gpkg")
    conus_sa = pd.read_csv(data_dir / "CONUS_SA_DOY.csv")
    gpp_er = pd.read_csv(data_dir / "Basin_GPP_ER_comparison.csv")
    
    us_bound = None
    if us_boundary_path is not None:
        # us boundary shapefiles (uc_bound) are available at https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
        us_bound = gpd.read_file(us_boundary_path).to_crs(basin.crs)
        
    
    
    #  ***************************Figure 1*********************************
    fig1 = Maps(basin,us_bound=us_bound)
    
    #  ***************************Figure 2*********************************
    fig2=binned_plot_combine(basin)

    #  ***************************Figure 3*********************************
    conus_sa["doy"] = pd.to_datetime(
        "2020-" + conus_sa["day"].astype(str), format="%Y-%j"
    )
    
    # CONUS surface area calculated at annual mean discharge.
    surface_area_mean_q = (
        basin["SA_Qmean"].sum()
        + uc["SAQmeanMn"].sum()
        + uc["SAQmeanSi"].sum()
    )
    
    fig3 = plot_CONUS_SA_map(
        basin, conus_sa, surface_area_mean_q, us_bound=us_bound
    )
    
    #  ***************************Figure 4*********************************
    gpp_er_monthly = pd.DataFrame({"month": range(1, 13)})
    for name in ["GPP_dyn", "ER_dyn", "GPP_static", "ER_static"]:
        variable, representation = name.split("_")
        gpp_er_monthly[name] = [
            gpp_er[f"{representation}_{variable}flux_{month:02d}"].sum()
            for month in gpp_er_monthly["month"]
        ]

    fig4 = plot_GPP_ER_comparison(gpp_er_monthly, basin, us_bound=us_bound)
    
    figures = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")

    return figures
    

# Input datasets include:
# 1) Basins.gpkg
# 2) UnitCatchments.gpkg
# 3) CONUS_SA_DOY.csv
# 4) Basin_GPP_ER_comparison.csv
# 5) U.S. boundary shapefiles, available from:
#    https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
#
# Files 1–4 are provided together with this code.
# File 5 is used only for visualization and is therefore optional.
# By default, the code runs without a U.S. boundary shapefile.

if __name__ == "__main__":
    main()


