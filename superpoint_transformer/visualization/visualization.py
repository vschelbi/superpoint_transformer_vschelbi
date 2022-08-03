from superpoint_transformer.data import Data
from superpoint_transformer.transforms import GridSampling3D
from torch_geometric.transforms import FixedPoints
import os.path as osp
import plotly
import plotly.graph_objects as go
import numpy as np
import torch
from itertools import chain

# TODO: To go further with ipwidgets :
#  - https://plotly.com/python/figurewidget-app/
#  - https://ipywidgets.readthedocs.io/en/stable/


PALETTE = plotly.colors.qualitative.Plotly


def rgb_to_plotly_rgb(rgb):
    """Convert torch.Tensor of float RGB values in [0, 1] to
    plotly-friendly RGB format.
    """
    assert isinstance(rgb, torch.Tensor) and rgb.max() <= 1.0 and rgb.dim() <= 2

    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)

    return [f"rgb{tuple(x)}" for x in (rgb * 255).int().numpy()]


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
        # If more than 3 features or more are found, project
        # features to a 3-dimensional space using N-simplex PCA
        # Heuristics for clamping
        #   - most features live in [0, 1]
        #   - most n-simplex PCA features live in [-0.5, 0.6]
        color = identity_PCA(feats, dim=3)
        color = (torch.clamp(color, -0.5, 0.6) + 0.5) / 1.1
        is_normalized = True

    if normalize and not is_normalized:
        # Unit-normalize the features in a hypercube of shared scale
        # for nicer visualizations
        if color.max() != color.min():
            color = color - color.min(dim=0).values.view(1, -1)
        color = color / (color.max(dim=0).values.view(1, -1) + 1e-6)

    return color


def identity_PCA(x, dim=3):
    """Reduce dimension of x based on PCA on the union of the n-simplex.
    This is a way of reducing the dimension of x while treating all
    input dimensions with the same importance, independently of the
    input distribution in x.
    """
    assert x.dim() == 2, f"Expected x.dim()=2 but got x.dim()={x.dim()} instead"

    # Create z the union of the N-simplex
    input_dim = x.shape[1]
    z = torch.eye(input_dim)

    # PCA on z
    z_offset = z.mean(axis=0)
    z_centered = z - z_offset
    cov_matrix = z_centered.T.mm(z_centered) / len(z_centered)
    _, eigenvectors = torch.symeig(cov_matrix, eigenvectors=True)

    # Apply the PCA on x
    x_reduced = (x - z_offset).mm(eigenvectors[:, -dim:])

    return x_reduced


def visualize_3d(
        data, figsize=800, width=None, height=None, class_names=None,
        class_colors=None, class_opacities=None, voxel=0.1, max_points=100000,
        pointsize=5, error_color=None, show_image_number=True, **kwargs):
    """3D data interactive visualization.

    :param data: Data object holding 3d points
    :param figsize: figure dimensions will be (figsize, figsize/2) if
      `width` and `height` are not specified
    :param width: figure width
    :param height: figure height
    :param class_names: names for point labels in MMData
    :param class_colors: colors for point labels in MMData
    :param class_opacities: class-wise opacities
    :param voxel: voxel size to subsample the point cloud to facilitate
      visualization
    :param max_points: maximum number of points displayed to facilitate
      visualization
    :param pointsize: size of points
    :param error_color: color used to identify mis-predicted points
    :param show_image_number: whether image numbers should be displayed
    :param kwargs:
    :return:
    """
    assert isinstance(mm_data, MMData)

    # 3D visualization modes
    modes = {'name': [], 'key': [], 'num_traces': []}

    # Make copies of the data and images to be modified in this scope
    data = mm_data.data.clone()
    has_2d = mm_data.modalities.get('image', None) is not None \
             and mm_data.modalities['image'].num_views > 0
    if has_2d:
        images = mm_data.modalities['image'].clone()

    # Convert images to ImageData for convenience
    # if isinstance(images, SameSettingImageData):
    #     images = ImageData([images])

    # Subsample to limit the drawing time
    data = GridSampling3D(voxel, mode='last')(data)
    if data.num_nodes > max_points:
        data = FixedPoints(
            max_points, replace=False, allow_duplicates=False)(data)

    # Subsample the mappings accordingly
    if has_2d and images[0].mappings is not None:
        transform = SelectMappingFromPointId()
        data, images = transform(data, images)

    # Round to the cm for cleaner hover info
    data.pos = (data.pos * 100).round() / 100
    if has_2d:
        for im in images:
            im.pos = (im.pos * 100).round() / 100

    # Class colors initialization
    if class_colors is not None and not isinstance(class_colors[0], str):
        class_colors = [f"rgb{tuple(x)}" for x in class_colors]
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
    initialized_visibility = False

    # Draw a trace for RGB 3D point cloud
    if getattr(data, 'rgb', None) is not None:
        fig.add_trace(
            go.Scatter3d(
                name='RGB',
                x=data.pos[:, 0],
                y=data.pos[:, 1],
                z=data.pos[:, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=rgb_to_plotly_rgb(data.rgb), ),
                hoverinfo='x+y+z',
                showlegend=False,
                visible=not initialized_visibility, ))
        modes['name'].append('RGB')
        modes['key'].append('rgb')
        modes['num_traces'].append(1)
        initialized_visibility = True

    # Draw a trace for labeled 3D point cloud
    if getattr(data, 'y', None) is not None:
        y = data.y.numpy()
        n_y_traces = 0

        for label in np.unique(y):
            indices = np.where(y == label)[0]

            fig.add_trace(
                go.Scatter3d(
                    name=class_names[label] if class_names else f"Class {label}",
                    opacity=class_opacities[label] if class_opacities else 1.0,
                    x=data.pos[indices, 0],
                    y=data.pos[indices, 1],
                    z=data.pos[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=pointsize,
                        color=class_colors[label] if class_colors else None, ),
                    visible=not initialized_visibility, ))
            n_y_traces += 1  # keep track of the number of traces

        modes['name'].append('Labels')
        modes['key'].append('y')
        modes['num_traces'].append(n_y_traces)
        initialized_visibility = True

    # Draw a trace for predicted labels 3D point cloud
    if getattr(data, 'pred', None) is not None:
        pred = data.pred.numpy()
        n_pred_traces = 0

        for label in np.unique(pred):
            indices = np.where(pred == label)[0]

            fig.add_trace(
                go.Scatter3d(
                    name=class_names[label] if class_names else f"Class {label}",
                    opacity=class_opacities[label] if class_opacities else 1.0,
                    x=data.pos[indices, 0],
                    y=data.pos[indices, 1],
                    z=data.pos[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=pointsize,
                        color=class_colors[label] if class_colors else None, ),
                    visible=not initialized_visibility, ))
            n_pred_traces += 1  # keep track of the number of traces

        modes['name'].append('Predictions')
        modes['key'].append('pred')
        modes['num_traces'].append(n_pred_traces)
        initialized_visibility = True

    # Draw a trace for 3D point cloud of number of images seen
    if has_2d and images[0].mappings is not None:
        n_seen = sum([
            im.mappings.pointers[1:] - im.mappings.pointers[:-1]
            for im in images])
        fig.add_trace(
            go.Scatter3d(
                name='Times seen',
                x=data.pos[:, 0],
                y=data.pos[:, 1],
                z=data.pos[:, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=n_seen,
                    colorscale='spectral',
                    colorbar=dict(
                        thickness=10, len=0.66, tick0=0,
                        dtick=max(1, int(n_seen.max() / 10.)), ), ),
                hovertext=[f"seen: {n}" for n in n_seen],
                hoverinfo='x+y+z+text',
                showlegend=False,
                visible=not initialized_visibility, ))
        modes['name'].append('Times seen')
        modes['key'].append('n_seen')
        modes['num_traces'].append(1)
        initialized_visibility = True

    # Draw a trace for position-colored 3D point cloud
    # radius = torch.norm(data.pos - data.pos.mean(dim=0), dim=1).max()
    # data.pos_rgb = (data.pos - data.pos.mean(dim=0)) / (2 * radius) + 0.5
    mini = data.pos.min(dim=0).values
    maxi = data.pos.max(dim=0).values
    data.pos_rgb = (data.pos - mini) / (maxi - mini + 1e-6)
    fig.add_trace(
        go.Scatter3d(
            name='Position RGB',
            x=data.pos[:, 0],
            y=data.pos[:, 1],
            z=data.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=rgb_to_plotly_rgb(data.pos_rgb), ),
            hoverinfo='x+y+z',
            showlegend=False,
            visible=not initialized_visibility, ))
    modes['name'].append('Position RGB')
    modes['key'].append('position_rgb')
    modes['num_traces'].append(1)
    initialized_visibility = True

    # Draw a trace for 3D point cloud features
    if getattr(data, 'x', None) is not None:
        # Recover the features and convert them to an RGB format for
        # visualization.
        data.feat_3d = feats_to_rgb(data.x, normalize=True)
        fig.add_trace(
            go.Scatter3d(
                name='Features 3D',
                x=data.pos[:, 0],
                y=data.pos[:, 1],
                z=data.pos[:, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=rgb_to_plotly_rgb(data.feat_3d), ),
                hoverinfo='x+y+z',
                showlegend=False,
                visible=not initialized_visibility, ))
        modes['name'].append('Features 3D')
        modes['key'].append('x')
        modes['num_traces'].append(1)
        initialized_visibility = True

    # Add a trace for prediction errors
    has_error = getattr(data, 'y', None) is not None \
                and getattr(data, 'pred', None) is not None
    if has_error:
        indices = np.where(data.pred.numpy() != data.y.numpy())[0]
        error_color = f"rgb{tuple(error_color)}" \
            if error_color is not None else 'rgb(255, 0, 0)'
        fig.add_trace(
            go.Scatter3d(
                name='Errors',
                opacity=1.0,
                x=data.pos[indices, 0],
                y=data.pos[indices, 1],
                z=data.pos[indices, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=error_color, ),
                showlegend=False,
                visible=False, ))
        modes['name'].append('Errors')
        modes['key'].append('error')
        modes['num_traces'].append(1)

    # Draw image positions
    if has_2d:

        img_traces = []
        if images.num_settings > 1:
            image_xyz = torch.cat([im.pos for im in images]).numpy()
            image_axes = torch.cat([im.axes for im in images]).numpy()
        else:
            image_xyz = images[0].pos.numpy()
            image_axes = images[0].axes.numpy()
        if len(image_xyz.shape) == 1:
            image_xyz = image_xyz.reshape((1, -1))
        for i, (xyz, axes) in enumerate(zip(image_xyz, image_axes)):

            # Draw image coordinate system axes
            arrow_length = 0.4
            for v, color in zip(axes, ['red', 'green', 'blue']):  # TODO: proprely rotate S3DIS equirectangular axes
                #             for v, color in zip(axes, [ 'blue', 'red', 'green',]):
                v = xyz + v * arrow_length
                fig.add_trace(
                    go.Scatter3d(
                        x=[xyz[0], v[0]],
                        y=[xyz[1], v[1]],
                        z=[xyz[2], v[2]],
                        mode='lines',
                        line=dict(
                            color=color,
                            width=pointsize + 7),
                        showlegend=False,
                        hoverinfo='none',
                        #                         visible=True,  # see all axes
                        visible=(color == 'blue'),  # see main axis
                    ))

            # Draw image position as ball
            img_traces.append(len(fig.data))
            fig.add_trace(
                go.Scatter3d(
                    name=f"Image {i}",
                    x=[xyz[0]],
                    y=[xyz[1]],
                    z=[xyz[2]],
                    mode='markers+text',
                    marker=dict(
                        line_width=2,
                        size=pointsize + 12,
                        color=PALETTE[i % len(PALETTE)], ),
                    text=f"<b>{i}</b>" if show_image_number else '',
                    textposition="bottom center",
                    textfont=dict(size=16),
                    hoverinfo='x+y+z+name',
                    showlegend=False,
                    visible=True, ))

    # Traces visibility for interactive point cloud coloring
    def trace_visibility(mode):
        visibilities = np.array([d.visible for d in fig.data], dtype='bool')

        # Traces visibility for interactive point cloud coloring
        i_mode = modes['key'].index(mode)
        a = sum(modes['num_traces'][:i_mode])
        b = sum(modes['num_traces'][:i_mode + 1])
        n_traces = sum(modes['num_traces'])

        visibilities[:n_traces] = False
        visibilities[a:b] = True

        return [{"visible": visibilities.tolist()}]

    # Create the buttons that will serve for toggling trace visibility
    updatemenus = [
        dict(
            buttons=[dict(label=name, method='update', args=trace_visibility(key))
                     for name, key in zip(modes['name'], modes['key']) if key != 'error'],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02, ),
    ]
    if has_error:
        updatemenus.append(
            dict(
                buttons=[dict(
                    method='restyle',
                    label='Error',
                    visible=True,
                    args=[{'visible': True, },
                          [sum(modes['num_traces'][:modes['key'].index('error')])]],
                    args2=[{'visible': False, },
                           [sum(modes['num_traces'][:modes['key'].index('error')])]], )],
                pad={'r': 10, 't': 10},
                showactive=False,
                type='buttons',
                xanchor='left',
                x=1.02,
                yanchor='top',
                y=1.02, ),
        )
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
                backgroundcolor="rgba(0, 0, 0, 0)"
            ),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)"
            ),
            zaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)"
            )
        )
    )

    output = {'figure': fig, 'data': data}

    if has_2d:
        output['images'] = images
        output['img_traces'] = img_traces

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
    fig_html = fig_html.replace('class="plotly-graph-div" style="',
                                'class="plotly-graph-div" style="margin:0 auto;')

    return fig_html


def show(
        data, path=None, title=None, no_output=True, **kwargs):
    """Multimodal 3D+2D data interactive visualization.

    :param data: Data object holding 3d points
    :param path: path to save the visualization into a sharable HTML
    :param title: figure title
    :param no_output: set to True if you want to return the 3D and 2D
      Plotly figure objects
    :param kwargs:
    :return:
    """
    assert isinstance(data, Data)

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
    out_3d = visualize_3d(mm_data, **kwargs)
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
