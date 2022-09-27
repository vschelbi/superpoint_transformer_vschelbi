import torch
import numpy as np
import os.path as osp
import plotly.graph_objects as go
from superpoint_transformer.data import Data, NAG
from superpoint_transformer.transforms import GridSampling3D, SaveOriginalPosId
from torch_scatter import scatter_mean
from colorhash import ColorHash


# TODO: To go further with ipwidgets :
#  - https://plotly.com/python/figurewidget-app/
#  - https://ipywidgets.readthedocs.io/en/stable/


def rgb_to_plotly_rgb(rgb):
    """Convert torch.Tensor of float RGB values in [0, 1] to
    plotly-friendly RGB format.
    """
    assert isinstance(rgb, torch.Tensor)
    assert rgb.max() <= 1.0
    assert rgb.dim() <= 2
    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)
    return [f"rgb{tuple(x)}" for x in (rgb * 255).int().numpy()]


def int_to_plotly_rgb(x):
    """Convert 1D torch.Tensor of int into plotly-friendly RGB format.
    This operation is deterministic on the int values.
    """
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 1
    assert not x.is_floating_point()
    x = x.cpu().long().numpy()
    palette = np.array([ColorHash(i).rgb for i in range(x.max() + 1)])
    return palette[x]


def hex_to_tensor(h):
    h = h.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return torch.Tensor(rgb) / 255


def feats_to_rgb(feats, normalize=False):
    """Convert features of the format M x N with N>=1 to an M x 3
    tensor with values in [0, 1 for RGB visualization].
    """
    is_normalized = False

    if feats.dim() == 1:
        feats = feats.unsqueeze(1)
    elif feats.dim() > 2:
        raise NotImplementedError

    if feats.shape[1] == 3:
        color = feats

    elif feats.shape[1] == 1:
        # If only 1 feature is found convert to a 3-channel
        # repetition for grayscale visualization.
        color = feats.repeat_interleave(3, 1)

    elif feats.shape[1] == 2:
        # If 2 features are found, add an extra channel.
        color = torch.cat([feats, torch.ones(feats.shape[0], 1)], 1)

    elif feats.shape[1] > 3:
        # If more than 3 features or more are found, project features to
        # a 3-dimensional space using N-simplex PCA. Heuristics for
        # clamping:
        #   - most features live in [0, 1]
        #   - most n-simplex PCA features live in [-0.5, 0.6]
        color = identity_PCA(feats, dim=3, normalize=normalize)
        color = (torch.clamp(color, -0.5, 0.6) + 0.5) / 1.1
        is_normalized = True

    if normalize and not is_normalized:
        # Unit-normalize the features in a hypercube of shared scale
        # for nicer visualizations
        if color.max() != color.min():
            color = color - color.min(dim=0).values.view(1, -1)
        color = color / (color.max(dim=0).values.view(1, -1) + 1e-6)

    return color


def identity_PCA(x, dim=3, normalize=False):
    """Reduce dimension of x based on PCA on the union of the n-simplex.
    This is a way of reducing the dimension of x while treating all
    input dimensions with the same importance, independently of the
    input distribution in x.
    """
    assert x.dim() == 2, f"Expected x.dim()=2, got x.dim()={x.dim()} instead"

    # Create z the union of the N-simplex
    input_dim = x.shape[1]
    z = torch.eye(input_dim)

    # PCA on z
    z_offset = z.mean(axis=0)
    z_centered = z - z_offset
    cov_matrix = z_centered.T.mm(z_centered) / len(z_centered)
    _, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Normalize x if need be
    if normalize:
        high = x.max(dim=0).values
        low = x.min(dim=0).values
        x = (x - low) / (high - low)
        x[x.isnan() | x.isinf()] = 0

    # Apply the PCA on x
    x_reduced = (x - z_offset).mm(eigenvectors[:, -dim:])

    return x_reduced


def visualize_3d(
        input, figsize=800, width=None, height=None, class_names=None,
        class_colors=None, voxel=-1, max_points=100000, pointsize=5,
        error_color=None, super_centre=True, super_edge=False,
        super_number=False, super_edge_attr=False, **kwargs):
    """3D data interactive visualization.

    :param input: Data or NAG object
    :param figsize: figure dimensions will be (figsize, figsize/2) if
      `width` and `height` are not specified
    :param width: figure width
    :param height: figure height
    :param class_names: names for point labels in MMData
    :param class_colors: colors for point labels in MMData
    :param voxel: voxel size to subsample the point cloud to facilitate
      visualization
    :param max_points: maximum number of points displayed to facilitate
      visualization
    :param pointsize: size of points
    :param error_color: color used to identify mis-predicted points
    :param super_centre: whether superpoint centroids should be
      displayed
    :param super_edge: whether superedges should be displayed
      (only if super_centre=True)
    :param super_number: whether superpoint numbers should be displayed
      (only if super_centre=True)
    :param super_edge_attr: whether the edges should be colored by their
      features (only if super_edge=True)
    :param kwargs

    :return:
    """
    assert isinstance(input, (Data, NAG))

    # We work on copies of the input data, to allow modified in this
    # scope. Data_0 accounts for the lowest level of hierarchy, the
    # points themselves.
    input = input.clone()
    input = NAG([input]) if isinstance(input, Data) and input.is_sub \
        else input
    is_nag = isinstance(input, NAG)
    data_0 = input[0] if is_nag else input

    # Subsample to limit the drawing time
    # If the level-0 cloud needs to be voxelized or sampled, a NAG
    # structure will be affected too. To maintain NAG consistency, we
    # only support 'GridSampling3D' with mode='last' and random sampling
    # without replacement. To keep track of the sampled points and index
    # the NAG accordingly, we use 'SaveOriginalPosId'
    idx = torch.arange(data_0.num_points)

    # If a voxel size is specified, voxelize the level-0. We first
    # isolate the 'pos' and the input indices of data_0 and apply
    # voxelization on this. We then recover the original grid-sampled
    # points indices to be used with Data.select or NAG.select
    if voxel > 0:
        data_temp = SaveOriginalPosId()(Data(pos=data_0.pos.clone()))
        data_temp = GridSampling3D(voxel, mode='last')(data_temp)
        idx = data_temp[SaveOriginalPosId.KEY]
        del data_temp

    # If the cloud is too large with respect to required 'max_points',
    # sample without replacement
    if idx.shape[0] > max_points:
        idx = idx[torch.randperm(idx.shape[0])[:max_points]]

    # If a sampling is needed, apply it to the input Data or NAG,
    # depending on the structure
    if idx.shape[0] < data_0.num_points:
        input = input.select(0, idx) if is_nag else input.select(idx)[0]
        data_0 = input[0] if is_nag else input

    # Round to the cm for cleaner hover info
    data_0.pos = (data_0.pos * 100).round() / 100

    # Class colors initialization
    if class_colors is not None and not isinstance(class_colors[0], str):
        class_colors = np.array([f"rgb{tuple(x)}" for x in class_colors])
    else:
        class_colors = None

    # Prepare figure
    width = width if width and height else figsize
    height = height if width and height else int(figsize / 2)
    margin = int(0.02 * min(width, height))
    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(aspectmode='data', ),  # preserve aspect ratio
        margin=dict(l=margin, r=margin, b=margin, t=margin),
        uirevision=True)
    fig = go.Figure(layout=layout)

    # Prepare 3D visualization modes
    trace_modes = []

    # Draw a trace for position-colored 3D point cloud
    mini = data_0.pos.min(dim=0).values
    maxi = data_0.pos.max(dim=0).values
    data_0.pos_rgb = (data_0.pos - mini) / (maxi - mini + 1e-6)
    colors = rgb_to_plotly_rgb(data_0.pos_rgb)
    fig.add_trace(
        go.Scatter3d(
            x=data_0.pos[:, 0],
            y=data_0.pos[:, 1],
            z=data_0.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=colors, ),
            hoverinfo='x+y+z+text',
            hovertext=None,
            showlegend=False,
            visible=True, ))
    trace_modes.append({})
    trace_modes[-1]['Position RGB'] = {
        'marker.color': colors, 'hovertext': None}

    # Draw a trace for RGB 3D point cloud
    if data_0.rgb is not None:
        colors = rgb_to_plotly_rgb(data_0.rgb)
        trace_modes[-1]['RGB'] = {'marker.color': colors, 'hovertext': None}

    # Color the points with ground truth semantic labels. If labels are
    # expressed as histograms, keep the most frequent one
    if data_0.y is not None:
        y = data_0.y
        y = y.argmax(1).numpy() if y.dim() == 2 else y.numpy()
        colors = class_colors[y] if class_colors is not None else None
        if class_names is None:
            text = np.array([f'Class {i}' for i in range(y.max() + 1)])
        else:
            text = np.array([str.title(c) for c in class_names])
        text = text[y]
        trace_modes[-1]['Labels'] = {'marker.color': colors, 'hovertext': text}

    # Color the points with predicted semantic labels. If labels are
    # expressed as histograms, keep the most frequent one
    if data_0.pred is not None:
        pred = data_0.pred
        pred = pred.argmax(1).numpy() if pred.dim() == 2 else pred.numpy()
        colors = class_colors[pred] if class_colors is not None else None
        if class_names is None:
            text = np.array([f'Class {i}' for i in range(pred.max() + 1)])
        else:
            text = np.array([str.title(c) for c in class_names])
        text = text[pred]
        trace_modes[-1]['Predictions'] = {
            'marker.color': colors, 'hovertext': text}

    # Draw a trace for 3D point cloud features
    if data_0.x is not None:
        # Recover the features and convert them to an RGB format for
        # visualization.
        data_0.feat_3d = feats_to_rgb(data_0.x, normalize=True)
        colors = rgb_to_plotly_rgb(data_0.feat_3d)
        trace_modes[-1]['Features 3D'] = {
            'marker.color': colors, 'hovertext': None}

    # Draw a trace for 3D point cloud sampling (for sampling debugging)
    if 'super_sampling' in data_0.keys:
        # Recover the features and convert them to an RGB format for
        # visualization.
        colors = int_to_plotly_rgb(data_0.super_sampling)
        colors[data_0.super_sampling == -1] = 230
        trace_modes[-1]['Super sampling'] = {
            'marker.color': colors, 'hovertext': None}

    # Draw a trace for the each cluster level
    for i_level, data_i in enumerate(input if is_nag else []):

        # Exit in case the Data has no 'super_index'
        if not data_i.is_sub:
            break

        # 'Data.super_index' are expressed between levels i and i+1, but
        # we need to recover the 'super_index' between level 0 and i+1,
        # to draw clusters on the level-0 points. To this end, we
        # compute the desired 'super_index' iteratively, with a
        # bottom-up approach
        if i_level == 0:
            super_index = data_i.super_index
        else:
            super_index = data_i.super_index[super_index]

        # Note that we update the 'trace_modes' 0th element here, this
        # assumes only it is the trace holding all level-0 points and on
        # which all other colors modes are defined
        colors = int_to_plotly_rgb(super_index)
        text = [f'c: {i}' for i in super_index]
        trace_modes[0][f'Level {i_level + 1}'] = {
            'marker.color': colors, 'hovertext': text}

        # Skip to the next level if we do not need to draw the cluster
        # centroids
        if not super_centre:
            continue

        # To recover centroids of the i+1 level superpoints, we either
        # read them from the next NAG level or compute them using the
        # level i 'super_index' indices
        num_levels = input.num_levels
        is_last_level = i_level == num_levels - 1
        if is_last_level or input[i_level + 1].pos is None:
            super_pos = scatter_mean(data_0.pos, super_index, dim=0)
        else:
            super_pos = input[i_level + 1].pos

        # Round to the cm for cleaner hover info
        super_pos = (super_pos * 100).round() / 100

        # Draw the level-i+1 cluster centroids
        idx_sp = torch.arange(data_i.super_index.max() + 1)
        colors = int_to_plotly_rgb(idx_sp)
        text = [f"<b>{i}</b>" for i in idx_sp] if super_number else ''
        fig.add_trace(
            go.Scatter3d(
                x=super_pos[:, 0],
                y=super_pos[:, 1],
                z=super_pos[:, 2],
                mode='markers+text',
                marker=dict(
                    symbol='diamond',
                    size=int(pointsize * 3),
                    color=colors,
                    line_width=pointsize,
                    line_color='black'),
                textposition="bottom center",
                textfont=dict(size=16),
                hovertext=text,
                hoverinfo='x+y+z+text',
                showlegend=False,
                visible=False, ))
        trace_modes.append({})
        trace_modes[-1][f'Level {i_level + 1}'] = {
            'marker.color': colors, 'hovertext': text}

        # Do not draw superedges if not required or if the i+1 level
        # does not have any
        if not super_edge or is_last_level or not input[i_level + 1].has_edges:
            continue

        # Recover the superedge source and target positions
        se = input[i_level + 1].edge_index

        # Drawn superedges are assumed to be undirected here, so we
        # consider (i, j) and (j, i) to be duplicates. By construction,
        # we assume all superedges are represented in both directions in
        # a NAG (this assumption does not hold for drawing any type of
        # directed graph, obviously). Due to this, we can easily remove
        # duplicate superedges by only keeping egdes such that i < j
        edge_mask = se[0] < se[1]
        se = se[:, edge_mask]

        # Recover corresponding source and target coordinates using the
        # previously-computed 'super_pos' cluster centroid positions
        s_pos = super_pos[se[0]].cpu().numpy()
        t_pos = super_pos[se[1]].cpu().numpy()

        # Convert into a plotly-friendly format for 3D lines
        edges = np.full((se.shape[1] * 3, 3), None)
        edges[::3] = s_pos
        edges[1::3] = t_pos

        if super_edge_attr and input[i_level + 1].edge_attr is not None:
            # Recover edge features and convert them to RGB colors. NB:
            # edge features are assumed to be in [0, 1] or [-1, 1].
            # Since we only draw edges in one direction, we choose to
            # only represent the absolute value of the features. This
            # implies that features are either direction-independent or
            # that the edge direction only changes the sign of the
            # feature
            edge_attr = input[i_level + 1].edge_attr[edge_mask].abs()
            colors = rgb_to_plotly_rgb(feats_to_rgb(edge_attr, normalize=True))
            edge_width = pointsize * 3
        else:
            colors = 'black'
            edge_width = pointsize

        # Draw the level-i+1 superedges
        fig.add_trace(
            go.Scatter3d(
                x=edges[:, 0],
                y=edges[:, 1],
                z=edges[:, 2],
                mode='lines',
                line=dict(
                    width=edge_width,
                    color=colors,),
                hoverinfo='skip',
                showlegend=False,
                visible=False, ))
        trace_modes.append({})
        trace_modes[-1][f'Level {i_level + 1}'] = {}

    # Add a trace for prediction errors. NB: it is important that this
    # trace is created last, as the button behavior for this one is
    # particular
    has_error = data_0.y is not None and data_0.pred is not None
    if has_error:

        # Recover prediction and ground truth and deal with potential
        # histograms
        y = data_0.y
        y = y.argmax(1).numpy() if y.dim() == 2 else y.numpy()
        pred = data_0.pred
        pred = pred.argmax(1).numpy() if pred.dim() == 2 else pred.numpy()

        # Identify erroneous point indices
        indices = np.where(pred != y)[0]

        # Prepare the color for erroneous points
        error_color = 'red' if error_color is None \
            else f'rgb{tuple(error_color)}'

        # Draw the erroneous points
        fig.add_trace(
            go.Scatter3d(
                x=data_0.pos[indices, 0],
                y=data_0.pos[indices, 1],
                z=data_0.pos[indices, 2],
                mode='markers',
                marker=dict(
                    size=int(pointsize * 1.5),
                    color=error_color, ),
                showlegend=False,
                visible=False, ))

    # Recover the keys for all visualization modes, as an ordered set,
    # with respect to their order of first appearance
    modes = list(dict.fromkeys([k for m in trace_modes for k in m.keys()]))

    # Traces color for interactive point cloud coloring
    def trace_update(mode):
        # Prepare the output args for the figure update attributes. By
        # default, all traces are non visible, with no color and no
        # hover text
        n_traces = len(trace_modes)
        out = {
            'visible': [False] * (n_traces + has_error),
            'marker.color': [None] * n_traces,
            'hovertext': [''] * n_traces}

        # For each trace in 'trace_modes' see if it contains 'mode' and
        # adapt out accordingly
        for i_trace, t_modes in enumerate(trace_modes):

            # The trace has no action for the mode, skip it and leave
            # the default args for the trace
            if mode not in t_modes:
                continue

            # Note that a trace will only be visible for its modes
            # declared in trace_modes
            out['visible'][i_trace] = True
            for key, val in t_modes[mode].items():
                out[key][i_trace] = val

        return [out, list(range(len(trace_modes)))]

    # Create the buttons that will serve for toggling trace visibility
    updatemenus = [
        dict(
            buttons=[dict(
                label=mode, method='update', args=trace_update(mode))
                for mode in modes if mode.lower() != 'errors'],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02, ),]

    if has_error:
        updatemenus.append(
            dict(
                buttons=[dict(
                    method='restyle',
                    label='Errors',
                    visible=True,
                    args=[
                        {'visible': True, 'marker.color': error_color},
                        [len(trace_modes)]],
                    args2=[
                        {'visible': False,},
                        [len(trace_modes)]],)],
                pad={'r': 10, 't': 10},
                showactive=False,
                type='buttons',
                xanchor='left',
                x=1.02,
                yanchor='top',
                y=1.02, ),)

    fig.update_layout(updatemenus=updatemenus)

    # Place the legend on the left
    fig.update_layout(
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99))

    # Hide all axes and no background
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            xaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)"),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)"),
            zaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)")))

    output = {'figure': fig, 'data': data_0}

    return output


def figure_html(fig):
    # Save plotly figure to temp HTML
    fig.write_html(
        '/tmp/fig.html',
        config={'displayModeBar': False},
        include_plotlyjs='cdn',
        full_html=False)

    # Read the HTML
    with open("/tmp/fig.html", "r") as f:
        fig_html = f.read()

    # Center the figure div for cleaner display
    fig_html = fig_html.replace(
        'class="plotly-graph-div" style="',
        'class="plotly-graph-div" style="margin:0 auto;')

    return fig_html


def show(
        input, path=None, title=None, no_output=True, **kwargs):
    """Interactive data visualization.

    :param input: Data or NAG object
    :param path: path to save the visualization into a sharable HTML
    :param title: figure title
    :param no_output: set to True if you want to return the 3D and 2D
      Plotly figure objects
    :param kwargs:
    :return:
    """
    # Sanitize title and path
    if title is None:
        title = "Multimodal data"
    if path is not None:
        if osp.isdir(path):
            path = osp.join(path, f"{title}.html")
        else:
            path = osp.splitext(path)[0] + '.html'
        fig_html = f'<h1 style="text-align: center;">{title}</h1>'

    # Draw a figure for 3D data visualization
    out_3d = visualize_3d(input, **kwargs)
    if no_output:
        if path is None:
            out_3d['figure'].show(config={'displayModeBar': False})
        else:
            fig_html += figure_html(out_3d['figure'])

    if path is not None:
        with open(path, "w") as f:
            f.write(fig_html)

    if not no_output:
        return out_3d

    return
