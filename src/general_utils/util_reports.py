import base64
import os
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image

from bokeh.models import HoverTool, ColumnDataSource, CustomJS
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup

import random

def generate_random_colors(N):
    color_list = []
    for i in range(N):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        color_list.append(color)
    return color_list

def scatter_plot_bokeh(output_dir, data, label, output_name, labels_name, colors, markers,  sizes, alphas):
    # Set up the Bokeh scatter plot figure
    p = figure(width=800, height=600, toolbar_location="right", tools='pan,box_zoom,reset,save',
               title="UMAP Scatter Plot")

    # Create a list to store the GlyphRenderers for each label
    renderers = []

    # Plot the points for each label using the appropriate color, marker, size, and alpha
    for i in range(len(np.unique(label))):
        mask = (label == i)
        source = ColumnDataSource(data=dict(
            x=data[mask, 0],
            y=data[mask, 1],
            label=label[mask]
        ))
        r = p.scatter('x', 'y', source=source, color=colors[i], marker=markers[i], size=sizes[i], alpha=alphas[i])
        renderers.append(r)

    # Set axis labels
    p.xaxis.axis_label = 'Embedding 1'
    p.yaxis.axis_label = 'Embedding 2'

    # Add a hover tool
    hover = HoverTool()
    hover.tooltips = [("Label", "@label")]
    p.add_tools(hover)

    # Create a CheckboxGroup widget to select which labels to display
    checkbox_group = CheckboxGroup(labels=labels_name, active=list(range(len(labels_name))))

    # Create a CustomJS callback to update the scatter plot based on the checkbox selection
    callback = CustomJS(args=dict(renderers=renderers, checkbox_group=checkbox_group), code="""
        const selected_labels = checkbox_group.active;
        for (let i = 0; i < renderers.length; i++) {
            renderers[i].visible = selected_labels.includes(i);
        }
    """)

    # Attach the callback to the CheckboxGroup
    checkbox_group.js_on_change('active', callback)

    # Arrange the plot and the widget in a column layout
    layout = column(p, checkbox_group)

    # Show the plot
    output_notebook()
    show(layout, notebook_handle=True)

    # Save the plot
    output_file(os.path.join(output_dir, f"{output_name}.html"))
    show(layout)

def scatter_plot(output_dir, data, label, output_name, labels_name, colors, markers, sizes, alphas):
    column_width_pt = 516.0
    pt_to_inch = 1 / 72.27
    column_width_inches = column_width_pt * pt_to_inch
    aspect_ratio = 4 / 3
    sns.set(style="whitegrid", font_scale=1.6,   rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

    fig, ax = plt.subplots()
    # Plot each label separately with the appropriate color, marker, size, and alpha.
    for i in range(len(np.unique(label))):
        mask = (label == i)
        ax.scatter(
            data[mask, 0], data[mask, 1], c=colors[i], edgecolor='none', marker=markers[i], s=sizes[i], alpha=alphas[i],
            label=labels_name[i]
        )
    # Add axis labels and title.
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{output_name}.pdf"), dpi=400, format='pdf')
    # Show the plot.
    plt.show()