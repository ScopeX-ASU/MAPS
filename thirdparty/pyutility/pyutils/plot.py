"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 02:02:14
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 02:02:14
"""

import numpy as np
import scienceplots
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LinearLocator
from mpl_toolkits.axes_grid1 import Divider, Size
try:
    from reportlab.graphics import renderPDF
except Exception:
    print("Cannot import reportlib library, svg2pdf() cannot work.")
from scipy.interpolate import UnivariateSpline as spline
from svglib.svglib import svg2rlg

__all__ = [
    "set_ieee",
    "set_ms",
    "plotGraph",
    "draw_box_plot",
    "autolabel",
    "draw_bar_plot",
    "draw_line_plot",
    "draw_pie_plot",
    "draw_mesh2d_plot",
    "draw_surface3d_plot",
    "batch_plot",
    "pdf_crop",
    "svg2pdf",
    "default_color_cycle_dict",
    "set_axis_formatter",
    "set_axes_size_ratio",
]


def set_ieee():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.sans-serif": ["Helvetica"]})

def set_ms():
    plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})


set_ieee()

default_color_cycle_dict = {
    "red": "#de425b",  # red
    "blue": "#1F77B4",  # blue
    "orange": "#f58055",  # orange
    "yellow": "#f6df7f",  # yellow
    "green": "#2a9a2a",  # green
    "grey": "#979797",  # grey
    "purple": "#AF69C5",  # purple
}
default_color_cycle = [
    "#de425b",  # red
    "#1F77B4",  # blue
    "#f58055",  # orange
    "#f6df7f",  # yellow
    "#2a9a2a",  # green
    "#979797",  # grey
    "#AF69C5",  # purple
]


def plotGraph(X, Y):
    fig = plt.figure()
    return fig


def smooth_line(x, y, smoothness=0):
    assert 0 <= smoothness <= 1, f"[E] Only support smoothness within [0,1]"
    if smoothness < 1e-3:
        return x, y

    # N = len(x)
    # N_new = int(N * (smoothness * 6 + 1))
    # x_smooth = np.linspace(min(x), max(x), N_new)
    # y_smooth = spline(x, y, x_smooth)
    # return x_smooth.tolist(), y_smooth.tolist()
    spl = spline(x, y)
    spl.set_smoothing_factor(smoothness)
    y_smooth = spl(x)
    return x, y_smooth


def draw_box_plot(data, ax, edge_color, fill_color, yrange, linewidth=1, markersize=1, boxwidth=0.5):
    meanpointprops = dict(marker="D", markeredgecolor="black", markerfacecolor="red", markersize=markersize)
    bp = ax.boxplot(data, showmeans=True, patch_artist=True, meanprops=meanpointprops, widths=boxwidth)
    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color, linewidth=linewidth)
    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)
    # for flier in bp['fliers']:
    # 	.set(marker='o', color='#e7298a', alpha=2)

    plt.setp(bp["fliers"], marker="+", markersize=markersize, color="k")
    plt.setp(bp["means"], markersize=markersize)

    # plt.setp(bp['whiskers'], color='k', linestyle='-')

    # plt.setp(bp['fliers'], )
    plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(80)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(80)
    return bp


def autolabel(ax, bars, format="%.2f"):
    """Attach a text label above each bar in *rects*, displaying its height."""
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom
    for rect in bars:
        height = rect.get_height()
        # Fraction of axis height taken up by this rectangle
        p_height = height / y_height

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.85:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.15)
        else:
            label_position = height + (y_height * 0.01)
        ax.annotate(
            format % (height),
            xy=(rect.get_x() + rect.get_width() / 2, label_position),
            xytext=(0, 0),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def draw_bar_plot(data, ax, barwidth, barcolor, label=None, tick_label=None):
    bp = ax.bar(
        data["x"],
        data["y"],
        align="center",
        alpha=0.9,
        width=barwidth,
        color=barcolor,
        label=label,
        tick_label=tick_label,
    )
    return bp


def draw_line_plot(
    data, ax, linewidth, linecolor, label=None, marker=None, markersize=1, linestyle=None, alpha=1
):
    bp = ax.plot(
        data["x"],
        data["y"],
        linewidth=linewidth,
        color=linecolor,
        label=label,
        marker=marker,
        markersize=markersize,
        linestyle=linestyle,
        alpha=alpha,
    )
    return bp


def draw_scatter2d_plot(data, ax, linewidth, linecolor, label=None, marker=None, alpha=1):
    if "area" not in data:
        area = None
    else:
        area = data["area"]
    bp = ax.scatter(
        data["x"],
        data["y"],
        s=area,
        linewidths=linewidth,
        c=linecolor,
        label=label,
        marker=marker,
        alpha=alpha,
    )
    return bp


def draw_errorbar_plot(data, ax, linewidth, linecolor, label=None, alpha=1):
    bp = ax.errorbar(
        data["x"],
        data["y"],
        yerr=data["yerror"],
        linewidth=linewidth,
        color=linecolor,
        capthick=linewidth,
        capsize=linewidth * 2,
        elinewidth=linewidth,
        markersize=2,
        label=label,
        alpha=alpha,
    )
    return bp


def draw_pie_plot(data, ax, colors=None, fontsize=10):
    if colors is None:
        colors = default_color_cycle
    bp = ax.pie(
        data["x"],
        labels=data["y"],
        # autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={"fontsize": fontsize},
        labeldistance=1.05,
    )
    return bp


def draw_mesh2d_plot(data, fig, ax, fontsize=10, cmap=plt.cm.RdYlGn):
    ### x = [M], y = [N], z = [M, N]
    x, y, z = data["x"], data["y"], data["z"]
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    x = np.arange(x[0] - dx / 2, x[-1] + 1.1 * dx / 2, dx)
    y = np.arange(y[0] - dy / 2, y[-1] + 1.1 * dy / 2, dy)

    im = ax.pcolormesh(x, y, z, vmin=np.min(z), vmax=np.max(z), shading="auto", cmap=cmap)
    fig.colorbar(im, ax=ax)
    return im


def draw_surface3d_plot(data, fig, ax, fontsize=10, zrange=[0, 1, 0.1], zformat="%.1f"):
    ### x = [M], y = [N], z = [M, N]
    x, y, z = data["x"], data["y"], data["z"]
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, z, cmap=plt.cm.RdYlGn, linewidth=0, antialiased=False)
    # Customize the z axis.
    dz = zrange[2] / 20
    ax.set_zlim(zrange[0] - dz, zrange[1] + dz)
    ax.set_zticks(np.arange(zrange[0], zrange[1], step=zrange[2]))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter(zformat))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return surf


def pdf_crop(src, dst):
    import logging
    import os

    cmd = f"pdfcrop {src} {dst}"
    os.system(cmd)
    logging.info(f"The cropped pdf is saved in {dst}.")


def svg2pdf(src: str, dst: str) -> None:
    assert src.endswith(".svg"), print(f"[E] Source file must be SVG, but got {src}")
    assert dst.endswith(".pdf"), print(f"[E] Target file must be PDF, but got {dst}")
    drawing = svg2rlg(src)
    renderPDF.drawToFile(drawing, dst)


def set_axis_formatter(ax, axis="x", scale=1000, format="%.1f"):
    def _formatter(x, pos):
        return format % (x / scale)

    if axis == "x":
        ax.xaxis.set_major_formatter(FuncFormatter(_formatter))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_formatter))


def set_axes_size_ratio(axew, axeh, fig=None, ax=None):
    axew = axew * 3.3
    axeh = axeh * 2.5

    # lets use the tight layout function to get a good padding size for our axes labels.
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    fig.tight_layout()
    # obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    # work out what the new  ratio values for padding are, and the new fig size.
    neww = axew + oldw * (1 - r + l)
    newh = axeh + oldh * (1 - t + b)
    newr = r * oldw / neww
    newl = l * oldw / neww
    newt = t * oldh / newh
    newb = b * oldh / newh

    # right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww, newh)


def batch_plot(
    type,
    raw_data=None,
    fig=None,
    ax=None,
    name=None,
    # trace/line settings
    trace_color="#1871bf",
    pie_colors=None,
    xcolor="#000000",
    ycolor="#000000",
    trace_label="",
    trace_marker=None,
    trace_markersize=1,
    xlabel="",
    ylabel="",
    zlabel="",
    tick_label=None,
    xrange=[0, 1, 0.1],
    yrange=[0, 1, 0.1],
    zrange=[0, 1, 0.1],
    xlimit=None,
    ylimit=None,
    zlimit=None,
    xformat="%.1f",
    yformat="%.1f",
    zformat="%.1f",
    xscale="linear",
    yscale="linear",
    zscale="linear",
    linestyle=None,
    fontsize=10,
    barwidth=0.1,
    linewidth=1,
    boxwidth=0.5,
    gridwidth=0.5,
    smoothness=0,
    alpha=1,
    # figure settings
    figsize_pixels=[400, 300],
    dpi=300,
    figscale=[1, 1],
    # legend settings
    legend=False,
    legend_title="",
    legend_loc="upper right",
    legend_ncol=1,
    # science and ieee settings
    ieee=True,
    no_latex=False,
    # color map
    cmap=plt.cm.RdYlGn,
):
    """
    description: batch plot function\\
    @type {str} Support box, bar, line, errorbar.\\
    @raw_data {dict} {'x':np.ndarray, 'y':np.ndarray, 'yerror':np.ndarray or None}.\\
    @fig {Object} figure handler from plt.subplots.\\
    @ax {Object} axis handler from plt.subplots.\\
    @name {Deprecated, Optional}\\
    @trace_color {str, Optional} Trace/line color, hex. Default to "#1871bf".\\
    @pie_colors {list of str, Optional} pie chart color list, hex or word. Default to None.\\
    @xcolor {str, Optional} X-axis label color, hex or word. Default to #000000.\\
    @ycolor {str, Optional} Y-axis label color, hex or word. Default to #000000.\\
    @trace_label {str, Optional} Trace/line label. Default to "".\\
    @trace_marker {str, Optional} Trace/line marker. Default to None.\\
    @xlabel {str, Optional} X-axis label string. Default to None.\\
    @ylabel {str, Optional} Y-axis label string. Default to None.\\
    @zlabel {str, Optional} Z-axis label string. Default to None.\\
    @tick_label {str or list of str, Optional} tick label(s) for bar chart. Default to None.\\
    @xrange {list/tuple, Optional} X-axis range [start, end, step]. Default to [0, 1, 0.1].\\
    @yrange {list/tuple, Optional} Y-axis range [start, end, step]. Default to [0, 1, 0.1].\\
    @zrange {list/tuple, Optional} Z-axis range [start, end, step]. Default to [0, 1, 0.1].\\
    @xlimit {list/tuple, Optional} X-axis limit [start, end]. Default to None.\\
    @ylimit {list/tuple, Optional} Y-axis limit [start, end]. Default to None.\\
    @zlimit {list/tuple, Optional} Z-axis limit [start, end]. Default to None.\\
    @xformat {str, Optional} X-axis tick label format. Default to %.1f.\\
    @yformat {str, Optional} Y-axis tick label format. Default to %.1f.\\
    @zformat {str, Optional} Z-axis tick label format. Default to %.1f.\\
    @xscale {str, Optional} X-axis tick scale. Default to linear. [linear, log]\\
    @yscale {str, Optional} Y-axis tick scale. Default to linear. [linear, log]\\
    @zscale {str, Optional} Z-axis tick scale. Default to linear. [linear, log]\\
    @fontsize {int/float scalar, Optional} axis label, tick label font size. Default to 10.\\
    @barwidth {int/float scalar, Optional} bar width in bar chart. Default to 0.1.\\
    @linewidth {int/float scalar, Optional} line width for all lines. Default to 1.\\
    @gridwidth {int/float scalar, Optional} line width for grids. Default to 0.5.\\
    @smoothness {float scalar, Optional} Smoothness of the line. Valid from [0,1]. Default to 0.\\
    @figsize_pixels {list/tuple, Optional} Figure pixels [width, height]. Default to [400, 300].\\
    @dpi {int, Optional} DPI settings. Default to 300.\\
    @figscale {list/tuple, Optional} Dimension scales compared to ieee single column figure [height, width] / [3.3 inch, 2.5 inch]. Default to [1, 1].\\
    @legend {bool scalar, Optional} Whether turn on legend. Default to False.\\
    @legend_title {str, Optional} Legend title. Default to None.\\
    @legend_loc {str, Optional} Legend location from 'upper right' 'upper left', "lower right", "lower left". Default to 'upper right'.\\
    @legend_ncol {int, Optional} Legend number of columns.\\
    @ieee {bool scalar, Optional} Whether use science and ieee style. Default to True.\\
    return None
    """
    assert type in {
        "box",
        "bar",
        "line",
        "scatter",
        "errorbar",
        "pie",
        "mesh2d",
        "surface3d",
        "none",
        None,
    }, f"[E] Only support box, bar, line, scatter, errorbar, pie, mesh2d, surface3d chart, and none/None"
    if ieee:
        """https://github.com/garrettj403/SciencePlots"""
        plt.style.use(["science", "ieee"])
    if no_latex:
        plt.style.use(["science", "no-latex"])

        # 3.3 inch x 2.5 inch is for single column figure
    if fig is None:
        plt.gca().set_prop_cycle(None)
        if "3d" in type:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
        else:
            fig, ax = plt.subplots()

    width, height = 3.3, 2.5
    trace_color = default_color_cycle_dict.get(trace_color, trace_color)
    xcolor = default_color_cycle_dict.get(xcolor, xcolor)
    ycolor = default_color_cycle_dict.get(ycolor, ycolor)
    if pie_colors is not None:
        pie_colors = [default_color_cycle_dict.get(i, i) for i in pie_colors]
    if type == "box":
        data = [i for i in raw_data.values()]
        bp = draw_box_plot(
            data,
            ax,
            trace_color,
            "white",
            yrange,
            linewidth=linewidth,
            markersize=trace_markersize,
            boxwidth=boxwidth,
        )

        xtl = [int(i) for i in raw_data.keys()]
        ax.set_xticklabels(xtl)
    elif type == "bar":
        data = raw_data
        if gridwidth > 0:
            ax.grid(True, linewidth=gridwidth)
        else:
            ax.grid(False, linewidth=0)
        ax.set_axisbelow(True)
        bp = draw_bar_plot(data, ax, barwidth, trace_color, label=trace_label, tick_label=tick_label)
    elif type == "line":
        data = raw_data
        if smoothness > 1e-2:
            x, y = smooth_line(raw_data["x"], raw_data["y"], smoothness=smoothness)
            data["x"] = x
            data["y"] = y
        if gridwidth > 0:
            ax.grid(True, linewidth=gridwidth)
        else:
            ax.grid(False, linewidth=0)
        ax.set_axisbelow(True)
        bp = draw_line_plot(
            data,
            ax,
            linewidth,
            trace_color,
            label=trace_label,
            marker=trace_marker,
            markersize=trace_markersize,
            linestyle=linestyle,
            alpha=alpha,
        )
    elif type == "scatter":
        data = raw_data
        if smoothness > 1e-2:
            x, y = smooth_line(raw_data["x"], raw_data["y"], smoothness=smoothness)
            data["x"] = x
            data["y"] = y
        if gridwidth > 0:
            ax.grid(True, linewidth=gridwidth)
        else:
            ax.grid(False, linewidth=0)
        ax.set_axisbelow(True)
        bp = draw_scatter2d_plot(
            data, ax, linewidth, trace_color, label=trace_label, marker=trace_marker, alpha=alpha
        )
    elif type == "errorbar":
        data = raw_data
        if smoothness > 1e-2:
            x, y = smooth_line(raw_data["x"], raw_data["y"], smoothness=smoothness)
            data["x"] = x
            data["y"] = y
        if gridwidth > 0:
            ax.grid(True, linewidth=gridwidth)
        else:
            ax.grid(False, linewidth=0)
        ax.set_axisbelow(True)
        bp = draw_errorbar_plot(data, ax, linewidth, trace_color, label=trace_label, alpha=alpha)
    elif type == "pie":
        data = raw_data
        bp = draw_pie_plot(data, ax, colors=pie_colors, fontsize=fontsize)
    elif type == "mesh2d":
        data = raw_data
        bp = draw_mesh2d_plot(data, fig, ax, fontsize=fontsize, cmap=cmap)
    elif type == "surface3d":
        data = raw_data
        bp = draw_surface3d_plot(data, fig, ax, fontsize=fontsize, zrange=zrange, zformat=zformat)
    elif type == "none" or type is None:
        bp = None
    else:
        raise NotImplementedError

    if type not in {"pie", None}:
        # trace/line settings
        ax.xaxis.set_major_formatter(FormatStrFormatter(xformat))
        ax.yaxis.set_major_formatter(FormatStrFormatter(yformat))
        plt.ylabel(ylabel, fontsize=fontsize, fontweight="bold", color=ycolor)
        plt.xlabel(xlabel, fontsize=fontsize, fontweight="bold", color=xcolor)

        [i.set_linewidth(linewidth) for i in ax.spines.values()]
        if xlimit is None:
            xlimit = [xrange[0], xrange[1]]
        if ylimit is None:
            ylimit = [yrange[0], yrange[1]]
        if zlimit is None:
            zlimit = [zrange[0], zrange[1]]
        if type != "mesh2d":
            plt.xticks(np.arange(xrange[0], xrange[1], step=xrange[2]))
            plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))
            # dx = xrange[2] / 20
            # dy = yrange[2] / 20
            # ax.set_xlim([xrange[0]-dx, xrange[1]+dx])
            # ax.set_ylim([yrange[0]-dy, yrange[1]+dy])
            ax.set_xlim(xlimit[0], xlimit[1])
            ax.set_ylim(ylimit[0], ylimit[1])
        else:
            plt.xticks(np.arange(xrange[0], xrange[1], step=xrange[2]))
            plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))
            # dx = xrange[2] / 20
            # dy = yrange[2] / 20
            # ax.set_xlim([xrange[0]-dx, xrange[1]+dy])
            # ax.set_ylim([yrange[0]-dx, yrange[1]+dy])

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(fontsize)
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(fontsize)
        # if(xscale not in {"linear", "log"}):
        #     xscale = "linear"
        # if(yscale not in {"linear", "log"}):
        #     yscale = "linear"
        # if(zscale not in {"linear", "log"}):
        #     zscale = "linear"
        if xscale == "log":
            plt.xscale(xscale)
        if yscale == "log":
            plt.yscale(yscale)

        if tick_label is not None:
            ax.set_xticklabels(tick_label)
        # ax.set_zscale(zscale)
        # plt.zscale(zscale)

    # legend settings
    if legend:
        ax.legend(title=legend_title, loc=legend_loc, ncol=legend_ncol)
    plt.tight_layout()
    # figure settings
    fig.set_size_inches(figscale[0] * width, figscale[1] * height)
    if not ieee:
        DPI = fig.get_dpi()
        fig.set_size_inches(figsize_pixels[0] / float(DPI), figsize_pixels[1] / float(DPI))
    return fig, ax, bp
