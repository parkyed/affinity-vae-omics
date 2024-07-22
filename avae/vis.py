import copy
import logging
import os.path
import random
import typing

import altair
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchvision
import umap
from PIL import Image
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from .utils import (
    colour_per_class,
    create_grid_for_plotting,
    fill_grid_for_plottting,
    pose_interpolation,
    save_imshow_png,
    save_mrc_file,
)


def _encoder(i: Image) -> str:
    """Encode PIL Image as base64 buffer.
    Parameters
    ----------
    i: PIL.Image
        Image to be encoded.

    Returns
    -------
    str
        Encoded image.

    """
    import base64
    from io import BytesIO

    with BytesIO() as buffer:
        i.thumbnail((110, 110))
        i.save(buffer, "PNG")
        data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

    return f"{data}"


def _decoder(i: str) -> Image:
    """Decode base64 buffer as PIL Image.
    Parameters
    ----------
    i: str
        Encoded image.

    Returns
    -------
    PIL.Image
        Decoded image.
    """
    import base64
    from io import BytesIO

    return Image.open(BytesIO(base64.b64decode(i)))


def format(im: Image, data_dim: int) -> str | list[str] | None:
    """Format PIL Image as Pandas compatible Altair image display.

    Parameters
    ----------
    im: PIL.Image
        Image to be formatted.
    data_dim: int
        Dimension of the data.

    Returns
    -------
    list
        Formatted images compatible Altair image display is batch is true, as we are adding a reconstruction to an input that already exist.
    str
        Formatted image compatible Altair image display if batch is not true.
    """
    if len(im.shape) == 5 and data_dim == 3:
        batch = True
        im = np.sum(
            np.copy(im.squeeze(dim=1).cpu().detach().numpy()), axis=-1
        )  # .astype(np.uint8)
    elif len(im.shape) == 4 and data_dim == 3:
        batch = False
        im = np.sum(
            np.copy(im.squeeze(dim=0).cpu().detach().numpy()), axis=-1
        )  # .astype(np.uint8)
    elif len(im.shape) == 4 and data_dim == 2:
        batch = True
        im = np.copy(im.squeeze(dim=1).cpu().detach().numpy())
        # .astype(np.uint8)
    elif len(im.shape) == 3 and data_dim == 2:
        batch = False
        im = np.copy(im.squeeze(dim=0).cpu().detach().numpy())
        # .astype(np.uint8)
    else:
        logging.warning(
            "\n\nWARNING: Wrong data format, please pass either a single "
            "unsqueezed tensor or a batch to image formatter. Exiting.\n",
        )
        return None
    im *= 255
    im = im.astype(np.uint8)
    if batch:
        # if batch we are adding a reconstruction to an input that already
        # exists in the DF, '&' is a separator.
        return ["&" + _encoder(Image.fromarray(i)) for i in im]
    else:
        return _encoder(Image.fromarray(im))


def merge(im: str) -> str | None:
    """Merge 2 base64 buffers as PIL Images and encode back to base64
    buffers.

    Parameters
    ----------
    im: str
        Input PIL Images to be merged.

    Returns
    -------
    str
        Merged image.
    """
    i = im.split("&")
    if len(i) != 2:
        logging.warning(
            "\n\nWARNING: Image format corrupt. Number of images in meta_df: {}. "
            "Exiting. \n".format(len(i)),
        )
        return None

    im1 = _decoder(i[0])
    im2 = _decoder(i[1])

    new_image = Image.new("L", (im2.size[0] + im1.size[0], im2.size[1]))
    new_image.paste(im1, (0, 0))
    new_image.paste(im2, (im1.size[0], 0))
    data = _encoder(new_image)

    return f"data:image/png;base64,{data}"


def latent_embed_plot_tsne(
    xs: npt.NDArray,
    ys: npt.NDArray,
    classes: list | None = None,
    mode: str = "",
    epoch: int = 0,
    writer: typing.Any = None,
) -> None:
    """Plot static TSNE embedding.

    Parameters
    ----------
    xs: np.ndarray
        Array of latent vectors.
    ys: np.ndarray
        Array of labels.
    classes: list
        List of classes.
    mode: str
        Added data mode to the name of the saved figure (e.g train, valid, eval).
    epoch: int
        Current epoch
    writer: SummaryWriter
        Tensorboard summary writer
    """
    logging.info(
        "################################################################",
    )
    if not mode:
        logging.info("Visualising static TSNE embedding...\n")
    else:
        logging.info("Visualising static TSNE embedding " + mode + "...\n")

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    perplexity = 40
    if len(ys) < perplexity:
        perplexity = len(ys) - 1

    if len(xs.shape) != 2:
        logging.error("Embedding only accepts 2D arrays.")
        exit(1)
    if xs.shape[-1] == 1:
        logging.warning(
            "Data contains 1 dimension, cannot create embedding,"
            " plotting histogram instead...\n"
        )
    if xs.shape[-1] == 2:
        logging.warning(
            "Data already contains 2 dimensions, cannot create"
            " embedding, plotting scatter of original data...\n"
        )

    if xs.shape[-1] > 2:
        lats = TSNE(
            n_components=2, perplexity=perplexity, random_state=42
        ).fit_transform(xs)
    elif xs.shape[-1] == 2 or xs.shape[-1] == 1:
        lats = xs

    if classes is None:
        classes = sorted(list(np.unique(ys)))
    else:
        if np.setdiff1d(ys, classes).size > 0:
            classes = list(
                np.concatenate((classes, np.setdiff1d(ys, classes)))
            )

    n_classes = len(classes)

    if n_classes < 3:
        # If the number of classes are not moe than 3 the size of the figure would be too
        # small and matplotlib would through a singularity error
        fig, ax = plt.subplots(
            figsize=(int(n_classes / 2) + 7, int(n_classes / 2) + 5)
        )
    else:
        fig, ax = plt.subplots(
            figsize=(int(n_classes / 2) + 4, int(n_classes / 2) + 2)
        )
    # When the number of classes is less than 3 the image becomes two small
    colours = colour_per_class(classes)

    if xs.shape[-1] != 1:

        for mol_id, mol in enumerate(set(ys.tolist())):
            idx = np.where(np.array(ys.tolist()) == mol)[0]

            color = colours[classes.index(mol)]

            plt.scatter(
                lats[idx, 0],
                lats[idx, 1],
                s=24,
                # label=mol[:4], # don't want to cut off the legend labels at 4 characters
                label=mol,
                facecolor=color,
                edgecolor=color,
                alpha=0.2,
            )

        ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol = 5, fontsize=10) # changed from upper left, and repositioned
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")

    if xs.shape[-1] == 1:

        for mol_id, mol in enumerate(set(ys.tolist())):
            idx = np.where(np.array(ys.tolist()) == mol)[0]
            cols = colours[classes.index(mol)]
            plt.hist(
                lats[idx],
                100,
                color=cols,
                histtype="step",
                stacked=True,
                fill=False,
                label=mol[:4],
            )
        plt.legend(
            prop={"size": 10},
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            # fontsize=16,
            fontsize=12,
        )
        plt.xlabel("dim 1")
        plt.ylabel("freq")

    plt.tight_layout()
    plt.savefig(f"plots/embedding_TSNE{mode}.png")

    if writer:
        writer.add_figure("TSNE embedding", fig, epoch)

    plt.close()


def latent_embed_plot_umap(
    xs: npt.NDArray,
    ys: npt.NDArray,
    classes: list | None = None,
    mode: str = "",
    epoch: int = 0,
    writer: typing.Any = None,
) -> None:
    """Plot static UMAP embedding.

    Parameters
    ----------
    xs: np.ndarray
        Array of latent vectors.
    ys: np.ndarray
        Array of labels.
    classes: list
        List of classes.
    mode: str
        Added data mode to the name of the saved figure (e.g train, valid, eval).
    epoch: int
        Current epoch
    writer: SummaryWriter
        Tensorboard summary writer
    """
    logging.info(
        "################################################################",
    )

    if not mode:
        logging.info("Visualising static UMAP embedding...\n")
    else:
        logging.info("Visualising static UMAP embedding " + mode + "...\n")

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if len(xs.shape) != 2:
        logging.error("Embedding only accepts 2D arrays.")
        exit(1)
    if xs.shape[-1] == 1:
        logging.warning(
            "Data contains 1 dimension, cannot create embedding,"
            " plotting histogram instead...\n"
        )
    if xs.shape[-1] == 2:
        logging.warning(
            "Data already contains 2 dimensions, cannot create"
            " embedding, plotting scatter of original data...\n"
        )

    if xs.shape[-1] > 2:
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(xs)
    elif xs.shape[-1] == 2 or xs.shape[-1] == 1:
        embedding = xs

    if classes is None:
        classes = sorted(list(np.unique(ys)))
    else:
        if np.setdiff1d(ys, classes).size > 0:
            classes = list(
                np.concatenate((classes, np.setdiff1d(ys, classes)))
            )

    n_classes = len(classes)
    if n_classes < 3:
        fig, ax = plt.subplots(
            figsize=(int(n_classes / 2) + 7, int(n_classes / 2) + 5)
        )
    else:
        fig, ax = plt.subplots(
            figsize=(int(n_classes / 2) + 4, int(n_classes / 2) + 2)
        )

    colours = colour_per_class(classes)

    if xs.shape[-1] != 1:

        for mol_id, mol in enumerate(set(ys.tolist())):
            idx = np.where(np.array(ys.tolist()) == mol)[0]
            color = colours[classes.index(mol)]

            ax.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                s=24,
                label=mol,
                # label=mol[:4],
                facecolor=color,
                edgecolor=color,
                alpha=0.2,
            )

        ax.legend(bbox_to_anchor = (0.5, -0.1), loc = "upper center", ncol = 6, fontsize = 10) # changed position and font size
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")

    if xs.shape[-1] == 1:

        for mol_id, mol in enumerate(set(ys.tolist())):
            idx = np.where(np.array(ys.tolist()) == mol)[0]
            cols = colours[classes.index(mol)]
            plt.hist(
                embedding[idx],
                100,
                color=cols,
                histtype="step",
                stacked=True,
                fill=False,
                label=mol[:4],
            )
        plt.legend(
            prop={"size": 10},
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=12,
            # fontsize=16,
        )
        plt.xlabel("dim 1")
        plt.ylabel("freq")

    plt.tight_layout()
    plt.savefig(f"plots/embedding_UMAP{mode}.png")

    if writer:
        writer.add_figure("UMAP embedding", fig, epoch)

    plt.close()


def dyn_latentembed_plot(
    df: pd.DataFrame, epoch: int, embedding: str = "umap", mode: str = ""
):
    """Plot dynamic TSNE or UMAP embedding.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the latent vectors.
    epoch: int
        Current epoch.
    embedding: str
        Type of embedding to use, either 'umap' or 'tsne'.
    mode: str
        Added data mode to the name of the saved figure (e.g train, valid, eval).
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising dynamic embedding {}...\n".format(embedding))

    epoch += 1
    latentspace = df[[col for col in df if col.startswith("lat")]].to_numpy()
    if embedding == "umap":
        lat_emb = np.array(
            umap.UMAP(random_state=42).fit_transform(latentspace)
        )
        titlex = "UMAP-1"
        titley = "UMAP-2"
    else:
        lat_emb = np.array(
            TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(
                latentspace
            )
        )
        titlex = "t-SNE-1"
        titley = "t-SNE-2"
    df["emb-x"], df["emb-y"] = np.array(lat_emb)[:, 0], np.array(lat_emb)[:, 1]

    # create column select options for radio buttons
    # add no std to radio select options - change value here for marker size!
    if "std-off" not in df.columns:
        df.insert(loc=0, column="std-off", value=np.zeros(len(lat_emb)) + 0.5)
    opts = [col for col in df.columns if col.startswith("std")]

    # create radio buttons and bind to a folded column select
    bind_checkbox = altair.binding_radio(
        options=opts,
        labels=[
            str(int(i.split("-")[-1]) + 1)
            if "off" not in i
            else i.split("-")[-1]
            for i in opts
        ],
        name="Certainty of prediction per dimension:",
    )
    column_select = altair.selection_point(
        fields=["column"], bind=bind_checkbox, name="certainty"
    )

    # mode, class and in-chart selections (also work with shft+click for multi)
    selection = altair.selection_point(fields=["id"], on="mouseover")
    selection_mode = altair.selection_point(fields=["mode"])
    selection_class = altair.selection_point(fields=["id"])

    # in-chart color condition
    color = altair.condition(
        (selection & selection_mode & selection_class),
        altair.Color("id:N", legend=None),
        altair.value("lightgray"),
    )

    # tooltip disla on-mouseover
    tooltip = ["id", "meta", "mode", "avg", "image"]  # .append(opts)

    # main scatter plot
    scatter = (
        altair.Chart(df, title="shift+click for multi-select")
        .mark_point(size=100, opacity=0.5, filled=True)
        .transform_fold(fold=opts, as_=["column", "value"])
        .transform_filter(column_select)
        .encode(
            altair.X("emb-x", title=titlex),
            altair.Y("emb-y", title=titley),
            altair.Shape(
                "mode",
                scale=altair.Scale(
                    range=["circle", "square", "triangle", "diamond"]
                ),
                legend=None,
            ),
            altair.Tooltip(tooltip),
            color=color,
            size=altair.Size("value:Q", legend=None, aggregate="mean").scale(
                type="log"
            ),
        )
        .interactive()
        .properties(width=800, height=500)
        .add_params(selection)
        .add_params(selection_class)
        .add_params(selection_mode)
        .add_params(column_select)
    )

    # interactive class legend
    legend_class = (
        altair.Chart(df, title="Class")
        .mark_point(size=100, opacity=0.5, filled=True)
        .encode(
            y=altair.Y("id:N", axis=altair.Axis(title=None, orient="right")),
            color=altair.condition(
                selection_class,
                altair.Color("id:N"),
                altair.value("lightgrey"),
            ),
        )
        .add_params(selection_class)
    )

    # interactive mode legend
    legend_mode = (
        altair.Chart(df, title="Mode")
        .mark_point(size=100, opacity=0.5, filled=True)
        .encode(
            y=altair.Y("mode:N", axis=altair.Axis(title=None, orient="right")),
            shape=altair.Shape(
                "mode",
                scale=altair.Scale(
                    range=["circle", "square", "triangle", "diamond"]
                ),
                legend=None,
            ),
            color=altair.condition(
                selection_mode,
                altair.value("steelblue"),
                altair.value("lightgrey"),
            ),
        )
        .add_params(selection_mode)
    )

    # organise charts in window and configure fonts
    chart = (
        (scatter | altair.vconcat(legend_class, legend_mode))
        .configure_axis(labelFontSize=20, titleFontSize=20)
        .configure_legend(labelFontSize=20, titleFontSize=20)
        .configure_title(fontSize=20)
    )

    # save charts and latent embedding
    if not os.path.exists("latents"):
        os.mkdir("latents")
        # save latentspace and ids
    if embedding == "umap":
        chart.save(f"latents/plt_latent_embed_epoch_{epoch}_umap{mode}.html")
    elif embedding == "tsne":
        chart.save(f"latents/plt_latent_embed_epoch_{epoch}_tsne{mode}.html")


def confidence_plot(x, y, s, suffix=None):
    logging.info(
        "################################################################",
    )
    logging.info(
        "Visualising class-average confidence metrics " + suffix + "...\n",
    )
    cmap = plt.get_cmap("jet")
    cols = [cmap(i) for i in np.linspace(0, 1, len(x[0]))]
    rows = len(np.unique(y)) // 2
    if len(np.unique(y)) % 2 != 0:
        rows += 1
    fig, ax = plt.subplots(len(np.unique(y)), sharex=True, sharey=True)
    for c, cl in enumerate(np.unique(y)):
        mu_cl = np.take(x, np.where(np.array(y) == cl)[0], axis=0)
        var_cl = np.take(s, np.where(np.array(y) == cl)[0], axis=0)
        std_cl = np.exp(0.5 * var_cl)
        mu_cl = np.mean(mu_cl, axis=0)
        std_cl = np.mean(std_cl, axis=0)

        min_mu = np.min(mu_cl)
        max_mu = np.max(mu_cl)
        max_sig = np.max(std_cl)
        step = (2 * 4 * max_sig) / 100

        xs = np.arange(min_mu - (4 * max_sig), max_mu + (4 * max_sig), step)

        for i in range(len(mu_cl)):
            ax[c].plot(
                xs,
                norm.pdf(xs, mu_cl[i], std_cl[i]),
                color=cols[i],
                label="lat" + str(i + 1),
            )
        ax[c].set_title(cl)
    name = "plots/confidence.png"
    if suffix is not None:
        name = name[:-4] + "_" + suffix + name[-4:]
    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, bbox_to_anchor=(1.06, 0.9)
    )  # , loc="upper left")
    fig.savefig(name, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close()


def accuracy_plot(
    y_train: npt.NDArray,
    ypred_train: npt.NDArray,
    y_val: npt.NDArray,
    ypred_val: npt.NDArray,
    classes: str | None = None,
    mode: str = "",
    epoch: int = 0,
    writer: typing.Any = None,
):
    """Plot confusion matrix .

    Parameters
    ----------
    y_train: np.array
        Training labels.
    ypred_train: np.array
        Predicted training labels.
    y_val: np.array
        Validation labels (unseen data).
    ypred_val: np.array
        Predicted validation labels (unseen data).
    classes: str
        Path to csv file containing classes to be used.
    mode: str
        Added data mode to the name of the saved figures (e.g train, valid, eval).
    epoch: int
        Current epoch.
    writer: SummaryWriter
        Tensorboard summary writer
    """
    logging.info(
        "################################################################",
    )

    logging.info("Visualising accuracy: confusion and F1 scores ...\n")

    if classes is not None:
        classes_list = pd.read_csv(classes).columns.tolist()
    else:
        classes_list = np.unique(np.concatenate((y_train, ypred_train)))

    cm = confusion_matrix(y_train, ypred_train, labels=classes_list)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=classes_list
    )

    avg_accuracy = cm.diagonal() / cm.sum(axis=1)

    cmn = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
    dispn = ConfusionMatrixDisplay(
        confusion_matrix=cmn, display_labels=classes_list
    )

    with plt.rc_context(
        {"font.weight": "bold", "font.size": int(len(classes_list) / 3) + 3}
    ):
        fig, ax = plt.subplots(
            figsize=(int(len(classes_list)) / 2, int(len(classes_list)) / 2)
        )

        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)

        plt.tight_layout()
        plt.title(
            "Average accuracy at epoch {}: {:.3f}%".format(
                epoch, np.mean(avg_accuracy) * 100
            ),
            fontsize=7,                                       # ECP reduced font size to fit title
        )

        if not os.path.exists("plots"):
            os.mkdir("plots")

        plt.savefig(f"plots/confusion_train{mode}.png", dpi=300)

        if writer:
            writer.add_figure("Accuracy", fig, epoch)

        plt.close()

        fig, ax = plt.subplots(
            figsize=(int(len(classes_list)) / 2, int(len(classes_list)) / 2)
        )

        dispn.plot(
            cmap=plt.cm.Blues,
            ax=ax,
            xticks_rotation=90,
            values_format=".0f",
            im_kw={"vmin": 0, "vmax": 100},
        )

        plt.tight_layout()
        plt.title(
            "Average accuracy at epoch {}: {:.3f}%".format(
                epoch, np.mean(avg_accuracy) * 100
            ),
            fontsize=7,
        )

        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.xlabel("Predicted label (%)")
        plt.ylabel("True label (%)")
        plt.savefig(f"plots/confusion_train{mode}_norm.png", dpi=300)

        if writer:
            writer.add_figure("Accuracy (Norm)", fig, epoch)

        plt.close()

    classes_list_eval = np.unique(np.concatenate((y_val, ypred_val)))

    if np.setdiff1d(classes_list_eval, classes_list).size > 0:
        ordered_class_eval = np.concatenate(
            (classes_list, np.setdiff1d(classes_list_eval, classes_list))
        )
    else:
        ordered_class_eval = classes_list

    cm_eval = confusion_matrix(y_val, ypred_val, labels=ordered_class_eval)
    disp_eval = ConfusionMatrixDisplay(
        confusion_matrix=cm_eval, display_labels=ordered_class_eval
    )
    avg_accuracy_eval = cm_eval.diagonal() / cm_eval.sum(axis=1)

    cmn_eval = (
        cm_eval.astype("float") / cm_eval.sum(axis=1)[:, np.newaxis] * 100
    )
    dispn_eval = ConfusionMatrixDisplay(
        confusion_matrix=cmn_eval, display_labels=ordered_class_eval
    )

    if mode == "_eval":
        figure_name = "plots/confusion_eval"
    elif mode == "":
        figure_name = "plots/confusion_valid"
    else:
        figure_name = f"plots/confusion_{mode}"

    with plt.rc_context(
        {
            "font.weight": "bold",
            "font.size": int(len(ordered_class_eval) / 3) + 3,
        }
    ):
        fig, ax = plt.subplots(
            figsize=(
                int(len(ordered_class_eval)) / 2,
                int(len(ordered_class_eval)) / 2,
            )
        )
        disp_eval.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
        plt.tight_layout()
        plt.title(
            "Average accuracy at epoch {}: {:.1f}%".format(
                epoch, np.mean(avg_accuracy_eval) * 100
            ),
            fontsize=7,
        )
        plt.savefig(figure_name + ".png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(
            figsize=(int(len(classes_list)) / 2, int(len(classes_list)) / 2)
        )

        dispn_eval.plot(
            cmap=plt.cm.Blues,
            ax=ax,
            xticks_rotation=90,
            values_format=".0f",
            im_kw={"vmin": 0, "vmax": 100},
        )

        plt.tight_layout()
        plt.title(
            "Average accuracy at epoch {}: {:.1}% ".format(
                epoch, np.mean(avg_accuracy_eval) * 100
            ),
            fontsize=7,
        )
        plt.xlabel("Predicted label (%)")
        plt.ylabel("True label (%)")
        plt.savefig(figure_name + "_norm.png", dpi=300)
        plt.close()


def f1_plot(
    y_train: npt.NDArray,
    ypred_train: npt.NDArray,
    y_val: npt.NDArray,
    ypred_val: npt.NDArray,
    classes: str | None = None,
    mode: str = "",
    epoch: int = 0,
    writer: typing.Any = None,
):
    """Plot F1 values. If classes list is provided, the F1 scores are calculated only for the classess in the
    list. This avoids F1 scores being affected by unseen clases that can be added in evaluation.

    Parameters
    ----------
    y_train: np.array
        Training labels.
    ypred_train: np.array
        Predicted training labels.
    y_val: np.array
        Validation labels (unseen data).
    ypred_val: np.array
        Predicted validation labels (unseen data).
    classes: str
        Path to csv file containing classes to be used.
    mode: str
        Added data mode to the name of the saved figures (e.g train, valid, eval).
    epoch: int
        Epoch number.
    writer: SummaryWriter
        Tensorboard summary writer
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising F1 scores ...\n")
    if classes is not None:
        classes_list = pd.read_csv(classes).columns.tolist()
    else:
        classes_list = np.unique(np.concatenate((y_train, ypred_train)))

    classes_list_eval = np.unique(np.concatenate((y_val, ypred_val)))

    if np.setdiff1d(classes_list_eval, classes_list).size > 0:
        logging.info(
            f"Class {np.setdiff1d(classes_list_eval, classes_list)} will not be used to compute F1 values as it was unseen in training data."
        )

        index = np.argwhere(np.isin(y_val, classes_list)).ravel()

        y_val = np.array(y_val)[index].tolist()
        ypred_val = np.array(ypred_val)[index].tolist()

    train_f1_score = f1_score(
        y_train, ypred_train, average=None, labels=classes_list
    ).tolist()
    valid_f1_score = f1_score(
        y_val, ypred_val, average=None, labels=classes_list
    ).tolist()

    if mode == "_eval":
        label = "eval"
    else:
        label = "valid"

    f1_train_file = f"plots/f1_train{mode}.csv"
    f1_valid_file = f"plots/f1_{label}.csv"

    train_df = pd.DataFrame([train_f1_score], columns=classes_list)
    valid_df = pd.DataFrame([valid_f1_score], columns=classes_list)

    train_df["epoch"] = epoch
    valid_df["epoch"] = epoch

    train_df["f1_avg"] = np.mean(train_f1_score)
    valid_df["f1_avg"] = np.mean(valid_f1_score)

    if os.path.exists(f1_train_file) and epoch != 0:
        f1_train = pd.read_csv(f1_train_file)
        pd.concat([f1_train, train_df]).to_csv(f1_train_file, index=False)
    else:
        train_df.to_csv(f1_train_file, index=False)

    if os.path.exists(f1_valid_file) and epoch != 0:
        f1_valid = pd.read_csv(f1_valid_file)
        pd.concat([f1_valid, valid_df]).to_csv(f1_valid_file, index=False)
    else:
        valid_df.to_csv(f1_valid_file, index=False)

    with plt.rc_context(
        {
            "font.weight": "bold",
            "font.size": int(len(classes_list) / 3) + 3,
        }
    ):
        fig, ax = plt.subplots(
            figsize=(
                int(len(classes_list)) / 2,
                int(len(classes_list)) / 2,
            )
        )
        plt.plot(classes_list, train_f1_score, label="train", marker="o")
        plt.plot(classes_list, valid_f1_score, label=label, marker="o")
        plt.xticks(rotation=45)
        plt.legend(loc="lower left")
        plt.title("F1 Score at epoch {}".format(epoch))
        plt.ylabel("F1 Score")
        plt.savefig(f"plots/f1{mode}.png", dpi=150)

        if writer:
            writer.add_figure("F1 score", fig, epoch)

        plt.close()


def loss_plot(
    epochs: int,
    beta: float,
    gamma: float,
    train_loss: list[float],
    val_loss: list[float] | None = None,
    p: list | None = None,
) -> None:
    """Visualise loss over epochs.

    Parameters
    ----------
    epochs: int
        Number of epochs.
    beta: list[float]
        List of beta values.
    gamma: list[float]
        List of gamma values.
    train_loss: list
        Training loss over epochs.
    val_loss: list
        Validation loss over epochs.
    p: list
        List of 7 hyperparameters: batch size, depth, "
                "channel init, latent dimension, learning rate, beta, gamma.
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising loss ...\n")

    train_loss = np.transpose(np.asarray(train_loss))
    if val_loss is not None:
        val_loss = np.transpose(np.asarray(val_loss))

    cols = ["blue", "red", "green", "orange"]
    labs = [
        "Total loss",
        "Reconstruction loss",
        "KL divergence loss x BETA",
        "Affinity loss x GAMMA",
    ]
    vlabs = [
        "VAL Total loss",
        "VAL Reconstruction loss",
        "VAL KL divergence loss x BETA",
        "VAL Affinity loss x GAMMA",
    ]

    plt.clf()
    plt.ticklabel_format(useOffset=False)

    train_loss[-2] = train_loss[-2] * beta
    train_loss[-1] = train_loss[-1] * gamma

    for i, loss in enumerate(train_loss):
        s = "-"
        plt.plot(
            range(1, epochs + 1), loss, c=cols[i], linestyle=s, label=labs[i]
        )
    if val_loss is not None:
        val_loss[-2] = val_loss[-2] * beta
        val_loss[-1] = val_loss[-1] * gamma
        for i, loss in enumerate(val_loss):
            s = "--"
            plt.plot(
                range(1, epochs + 1),
                loss,
                c=cols[i],
                linestyle=s,
                label=vlabs[i],
            )

    if p is not None:
        if len(p) != 7:
            logging.warning(
                "\n\nWARNING: Function vis.loss_plot is expecting 'p' parameter "
                "to be a list of 7 hyperparameters: batch size, depth, "
                "channel init, latent dimension, learning rate, beta, gamma. "
                "Exiting.\n",
            )
            return
        plt.title(
            "bs: %d, d: %d, ch: %d, lat: %d, lr: %.3f, beta: %.1f, "
            "gamma: %.1f" % (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        )
    else:
        plt.title("Loss")

    plt.yscale("log")
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()

    plt.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/loss.png", dpi=300)
    plt.close()

    # only training loss
    for i, loss in enumerate(train_loss):
        s = "-"
        plt.plot(
            range(1, epochs + 1), loss, c=cols[i], linestyle=s, label=labs[i]
        )

    plt.yscale("log")
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/loss_train.png", dpi=300)
    plt.close()

    # plotting only the total loss as it sometimes is a few order of magnitude higher than KLD and affinity losses
    plt.plot(
        range(1, epochs + 1),
        train_loss[0],
        c=cols[0],
        linestyle="-",
        label=labs[0],
    )
    if val_loss is not None:
        plt.plot(
            range(1, epochs + 1),
            val_loss[0],
            c=cols[0],
            linestyle="--",
            label=vlabs[0],
        )
    plt.yscale("log")
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/loss_total.png", dpi=300)
    plt.close()


def recon_plot(
    img: torch.Tensor,
    rec: torch.Tensor,
    label: list,
    data_dim: int,
    mode: str = "trn",
    epoch: int = 0,
    writer: typing.Any = None,
) -> None:
    """Visualise reconstructions.

    Parameters
    ----------
    img: torch.Tensor
        Input images.
    rec: torch.Tensor
        Reconstructed images.
    label: list
        List of labels for reconstructed images.
    data_dim: int
        Dimensionality of the data.
    mode: str
        Type of image in the training set: trn or val.
    epoch: int
        Current epoch
    writer: SummaryWriter
        Tensorboard summary writer
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising reconstructions " + mode + "...\n")

    fname_in = str(mode) + "_recon_in.png"
    fname_out = str(mode) + "_recon_out.png"

    if data_dim == 3:
        img_2d = img[:, :, :, :, img.shape[-1] // 2]
        rec_2d = rec[:, :, :, :, img.shape[-1] // 2]
    elif data_dim == 2:
        img_2d = img
        rec_2d = rec

    img_2d = torchvision.utils.make_grid(img_2d.cpu(), 10, 2).numpy()
    rec_2d = torchvision.utils.make_grid(rec_2d.detach().cpu(), 10, 2).numpy()

    save_imshow_png(
        fname_in,
        np.transpose(img_2d, (1, 2, 0)),
        writer=writer,
        figname="Recon. (In)",
        epoch=epoch,
    )
    save_imshow_png(
        fname_out,
        np.transpose(rec_2d, (1, 2, 0)),
        writer=writer,
        figname="Recon. (Out)",
        epoch=epoch,
    )

    if data_dim == 3:
        rec = rec.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        label = np.array(label)
        dsize = rec.shape[-data_dim:]

        # The number of reconstruction and input images to be displayed in the .mrc output file
        number_of_random_samples = 10
        number_of_columns = 3
        padding = 0

        if len(label) < number_of_random_samples * number_of_columns:
            # In there are not enough images for the stack, do one column only
            number_of_random_samples = len(label)
            number_of_columns = 0

        # define the dimensions for the napari grid
        grid_for_napari = np.zeros(
            (
                number_of_random_samples * dsize[0],
                2 * dsize[1] * number_of_columns + padding * number_of_columns,
                dsize[2],
            ),
            dtype=np.float32,
        )

        logging.info("Molecules in the reconstructions are  ...")
        for k in range(number_of_columns):
            # select 10 images at random
            rand_select = np.random.randint(
                0, high=img.shape[0], size=number_of_random_samples, dtype=int
            )  #
            img = img[rand_select, :, :, :, :]
            rec = rec[rand_select, :, :, :, :]
            logging.info(f"column {k} : {label[rand_select]}")

            # stack the images together with their reconstruction
            rec_img = np.hstack((img, rec))

            # Create and save the mrc file with single transversals
            for j in range(2):
                for i in range(number_of_random_samples):
                    grid_for_napari[
                        i * dsize[0] : (i + 1) * dsize[0],
                        j * dsize[1]
                        + (dsize[1] * 2 + padding) * k : (j + 1) * dsize[1]
                        + (dsize[1] * 2 + padding) * k,
                        :,
                    ] = rec_img[i, j, :, :, :]

        save_mrc_file(str(mode) + "_recon_in.mrc", grid_for_napari)
        logging.info("\n")


def latent_4enc_interpolate_plot(
    dsize: tuple,
    xs: torch.Tensor,
    ys: torch.Tensor,
    vae: torch.nn.Module,
    device: torch.device,
    plots_config: npt.NDArray,
    poses: list,
) -> None:
    """Visualise the interpolation of latent space between 4 randomly selected encodings.
    The number of plots and the number of interpolation steps is modifyable.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    x: torch.Tensor
        A sample batch. we extract 4 random images from this
    xs: list
        the list of all latent vectors
    ys: list
        the list of all labels for each latent vector in xs
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    plots_config: np.array
        Array containing the number of plots to be generated and the number of interpolation steps.
    poses: list
        List of pose vectors.
    """
    logging.info(
        "################################################################",
    )
    logging.info(
        "Visualising Latent Interpolation between 4 randomly selected encodings ...\n"
    )
    padding = 0
    data_dim = len(dsize)
    classes = np.unique(np.asarray(ys))
    latent_dim = xs.shape[1]

    # Number of plots (each have 4 random corners)
    plots_config = plots_config.replace(" ", "").split(",")

    # Number of interpolation steps
    num_steps = int(plots_config[1])

    if poses is not None:
        pose_mean = np.mean(poses)

    for num_fig in range(int(plots_config[0])):
        enc = []

        draw_four = random.sample(range(len(classes)), k=4)
        selected_classes = [classes[index] for index in draw_four]
        for idx in draw_four:
            lat = np.take(
                xs,
                random.sample(list(np.where(ys == classes[idx])[0]), k=1),
                axis=0,
            )
            enc.append(lat)

        enc = np.asarray(enc)
        alpha_values = torch.linspace(0, 1, num_steps)
        beta_values = torch.linspace(0, 1, num_steps)
        decoded_grid = []

        for h in alpha_values:
            for v in beta_values:

                # bilinear interpolation in the latent space
                interpolated_z = (
                    (1 - h) * (1 - v) * enc[0]
                    + h * (1 - v) * enc[1]
                    + (1 - h) * v * enc[2]
                    + h * v * enc[3]
                )

                # Decode the interpolated encoding to generate an image
                with torch.no_grad():
                    decoded_images = vae.decoder(
                        interpolated_z.view(-1, latent_dim).to(device=device),
                        (torch.zeros(1, poses[0].shape[0]) + pose_mean).to(
                            device=device
                        ),
                    )
                decoded_grid.append(decoded_images.cpu().squeeze().numpy())

        decoded_grid = np.reshape(
            np.array(decoded_grid), (num_steps, num_steps, *dsize)
        )

        grid_for_napari = create_grid_for_plotting(
            num_steps, num_steps, dsize, padding
        )
        grid_for_napari = fill_grid_for_plottting(
            num_steps,
            num_steps,
            grid_for_napari,
            dsize,
            decoded_grid,
            padding,
        )

        if data_dim == 3:
            save_mrc_file(
                f"latent_interpolate_{num_fig}.mrc",
                grid_for_napari,
            )
        elif data_dim == 2:
            save_imshow_png(
                f"latent_interpolate_{num_fig}.png",
                grid_for_napari,
            )


def latent_disentamglement_plot(
    dsize: tuple,
    lats: list,
    vae: torch.nn.Module,
    device: torch.device,
    poses: list | None = None,
    mode: str = "trn",
) -> None:
    """Visualise latent content disentanglement.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    lats: list
        List of latent vectors.
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    poses: list
        List of pose vectors.
    mode: str
        Mode of evaluation (trn: Training, vld: Validation, eval: Evaluation )
    writer: SummaryWriter
        Tensorboard summary writer
    """
    logging.info(
        "################################################################"
    )
    logging.info("Visualising latent content disentanglement ...\n")
    number_of_samples = 7
    padding = 0
    data_dim = len(dsize)
    latents = np.asarray(lats)
    if poses is not None:
        poses_space = np.asarray(poses)

    lat_means = np.mean(latents, axis=0)
    lat_stds = np.std(latents, axis=0)
    lat_dims = latents.shape[-1]

    if poses is not None:
        pos_means = np.mean(poses_space, axis=0)
        pos_dims = poses_space.shape[-1]

    recon_images = []

    # Generate vectors representing single transversals along each lat_dim
    for l_dim in range(lat_dims):
        for grid_spot in range(number_of_samples):
            means = copy.deepcopy(lat_means)
            # every 0.4 interval from -1.2 to 1.2 sigma
            means[l_dim] += lat_stds[l_dim] * (-1.2 + 0.4 * grid_spot)

            # Decode the current vector
            with torch.no_grad():
                current_lat_grid = torch.from_numpy(np.array([means])).to(
                    device
                )

                if poses is not None:
                    current_pos_grid = torch.from_numpy(
                        np.array([pos_means])
                    ).to(device)
                    current_recon = vae.decoder(
                        current_lat_grid, current_pos_grid
                    )
                else:
                    current_recon = vae.decoder(current_lat_grid, None)

            recon_images.append(current_recon.cpu().squeeze().numpy())

    # Combine the individual decoded images into a single array
    recon_images = np.array(recon_images)
    recon_images = np.reshape(
        recon_images, (lat_dims, number_of_samples, *dsize)
    )

    grid_for_napari = create_grid_for_plotting(
        lat_dims, number_of_samples, dsize, padding
    )
    grid_for_napari = fill_grid_for_plottting(
        lat_dims,
        number_of_samples,
        grid_for_napari,
        dsize,
        recon_images,
        padding,
    )

    if data_dim == 3:
        save_mrc_file(f"disentanglement-latent_{mode}.mrc", grid_for_napari)
    elif data_dim == 2:
        save_imshow_png(f"disentanglement-latent_{mode}.png", grid_for_napari)


def pose_class_disentanglement_plot(
    dsize: tuple,
    x: list,
    y: list,
    pose_vis_class: npt.NDArray,
    poses: list,
    vae: torch.nn.Module,
    device: torch.device,
    mode: str = "trn",
):

    """Visualise Pose interpolation per class. This function creates a pose interpolatoion
    plot for all classes listed in pose_vis_class.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    x: list
        List of latent vectors.
    y: List
        List of the labels associated with each latent vector in x
    pose_vis_class: list
        Classes to be used for pose interpolation (a seperate pose interpolation figure would be created for each class)."
    poses: list
        List of pose vectors.
    vae: torch.nn.Module
        Affinity vae model.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    device: torch.device
        Device to run the model on.
    mode: str
        Mode of evaluation (trn: Training, vld: Validation, eval: Evaluation )
    """
    logging.info(
        "Visualising pose disentanglement for each class {}...\n".format(
            "".join(pose_vis_class)
        )
    )
    if poses is None:
        logging.warning(
            "Pose interpolation cannot be done if pose dimension is not specified"
        )

    # number of pose interpolation steps, prefarably odd
    number_of_samples = 7
    padding = 0
    data_dim = len(dsize)
    x = np.asarray(x)

    poses_space = np.asarray(poses)
    pos_dims = poses_space.shape[-1]
    pose_vis_class_list = pose_vis_class.replace(" ", "").split(",")

    for i in pose_vis_class_list:
        decoded_grid = []
        class_x = np.take(x, np.where(np.array(y) == i)[0], axis=0)
        class_x_indx = np.random.choice(class_x.shape[0])
        enc = class_x[class_x_indx, :]

        class_pos = np.take(poses_space, np.where(np.array(y) == i)[0], axis=0)
        class_pos_mean = np.mean(class_pos, axis=0)
        class_pos_stds = np.std(class_pos, axis=0)

        decoded_grid = pose_interpolation(
            enc,
            pos_dims,
            class_pos_mean,
            class_pos_stds,
            dsize,
            number_of_samples,
            vae,
            device,
        )

        grid_for_napari = create_grid_for_plotting(
            pos_dims, number_of_samples, dsize, padding
        )
        grid_for_napari = fill_grid_for_plottting(
            pos_dims,
            number_of_samples,
            grid_for_napari,
            dsize,
            decoded_grid,
            padding,
        )

        if data_dim == 3:
            save_mrc_file(f"pose_interpolate_{mode}_{i}.mrc", grid_for_napari)
        elif data_dim == 2:
            save_imshow_png(
                f"pose_interpolate_{mode}_{i}.png", grid_for_napari
            )


def pose_disentanglement_plot(
    dsize: tuple,
    lats: list,
    poses: list,
    vae: torch.nn.Module,
    device: torch.device,
    label: str = "avg",
    mode: str = "trn",
):
    """Visualise pose disentanglement.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    lats: list
        List of latent vectors.
    poses: list
        List of pose vectors.
    vae: torch.nn.Module
        Affinity vae model.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    device: torch.device
        Device to run the model on.
    """
    logging.info(
        "################################################################",
    )
    if label == "avg":
        logging.info("Visualising pose disentanglement ...\n")
    else:
        logging.info(
            "Visualising pose disentanglement for class {}...\n".format(label)
        )

    number_of_samples = 7
    padding = 0
    data_dim = len(dsize)
    latents = np.asarray(lats)
    poses_space = np.asarray(poses)

    pos_means = np.mean(poses_space, axis=0)
    pos_stds = np.std(poses_space, axis=0)
    pos_dims = poses_space.shape[-1]

    lat_means = np.mean(latents, axis=0)

    recon = pose_interpolation(
        lat_means,
        pos_dims,
        pos_means,
        pos_stds,
        dsize,
        number_of_samples,
        vae,
        device,
    )

    grid_for_napari = create_grid_for_plotting(
        pos_dims, number_of_samples, dsize, padding
    )

    # Create and save the mrc file with single transversals
    grid_for_napari = fill_grid_for_plottting(
        pos_dims, number_of_samples, grid_for_napari, dsize, recon, padding
    )

    if data_dim == 3:
        save_mrc_file(
            f"disentanglement-pose_{mode}_{label}.mrc", grid_for_napari
        )
    elif data_dim == 2:
        save_imshow_png(
            f"disentanglement-pose_{mode}_{label}.png", grid_for_napari
        )


def interpolations_plot(
    dsize: tuple,
    lats: list,
    classes: list,
    vae: torch.nn.Module,
    device: torch.device,
    poses: list | None = None,
    mode: str = "trn",
) -> None:
    """Visualise interpolations.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    lats: list
        List of latent vectors.
    classes: list
        List of class labels.
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    poses: list
        List of pose vectors.
    mode: str
        Mode of evaluation (trn: Training, vld: Validation, eval: Evaluation )
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising interpolations ...\n")
    data_dim = len(dsize)
    lats = np.asarray(lats)
    classes = np.asarray(classes)

    if poses is not None:
        poses_space = np.asarray(poses)

    class_ids = np.unique(classes)
    if len(class_ids) <= 3:
        logging.warning(
            "\n\nWARNING: Interpolation plot needs at least 4 distinct classes, "
            "cannot visualise interpolations. Exiting.\n",
        )
        return

    class_reps_lats = np.take(
        lats, [np.where(classes == i)[0][0] for i in class_ids], axis=0
    )
    class_reps_lats = np.asarray(class_reps_lats)
    if poses is not None:
        class_reps_poses = np.asarray(
            np.take(
                poses_space,
                [np.where(classes == i)[0][0] for i in class_ids],
                axis=0,
            )
        )

    draw_four = random.sample(list(enumerate(class_reps_lats)), k=4)
    inds, class_rep = list(zip(*draw_four))
    class_rep_lats = np.asarray(class_rep)
    latent_dim = class_rep_lats.shape[1]

    if poses is not None:
        class_rep_poses = np.asarray([class_reps_poses[i] for i in inds])
        poses_dim = class_rep_poses.shape[1]

    # Generate a gird of latent vectors interpolated between reps of four ids

    grid_size = 6
    alpha_values = torch.linspace(0, 1, grid_size)
    beta_values = torch.linspace(0, 1, grid_size)
    decoded_grid = []
    for i, h in enumerate(alpha_values):
        for j, v in enumerate(beta_values):

            # bilinear interpolation in the latent space
            interpolated_z = (
                (1 - h) * (1 - v) * class_rep_lats[0]
                + h * (1 - v) * class_rep_lats[1]
                + (1 - h) * v * class_rep_lats[2]
                + h * v * class_rep_lats[3]
            )

            interpolated_pose = (
                (1 - h) * (1 - v) * class_reps_poses[0]
                + h * (1 - v) * class_reps_poses[1]
                + (1 - h) * v * class_reps_poses[2]
                + h * v * class_reps_poses[3]
            )
            with torch.no_grad():
                if poses is not None:
                    decoded_images = vae.decoder(
                        interpolated_z.view(-1, latent_dim).to(device=device),
                        (
                            torch.zeros(1, poses[0].shape[0])
                            + interpolated_pose
                        ).to(device=device),
                    )
                else:
                    decoded_images = vae.decoder(
                        interpolated_z.view(-1, latent_dim).to(device=device),
                        None,
                    )

            decoded_grid.append(decoded_images.cpu().squeeze().numpy())

    decoded_grid = np.reshape(
        np.array(decoded_grid), (grid_size, grid_size, *dsize)
    )

    grid_for_napari = create_grid_for_plotting(grid_size, grid_size, dsize)
    grid_for_napari = fill_grid_for_plottting(
        grid_size, grid_size, grid_for_napari, dsize, decoded_grid
    )
    # Create an mrc file with interpolations

    if data_dim == 3:
        save_mrc_file(f"interpolations_{mode}.mrc", grid_for_napari)
    elif data_dim == 2:
        save_imshow_png(f"interpolations_{mode}.png", grid_for_napari)


def plot_affinity_matrix(
    lookup: pd.DataFrame, all_classes: list, selected_classes: list
) -> None:
    """
    This function plots the Affinity matrix and highlights the
    classes selected for the given calculation.

    Parameters
    ----------
    all_classes: list
        All existing classes in the affinity matrix  affinity*.csv
    lookup: pandas.DataFrame
        The affinity matrix
    selected_classes : list
        All classes selected by the user for training in classes.csv
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising affinity matrix ...\n")

    with plt.rc_context(
        {"font.weight": "bold", "font.size": int(len(all_classes) / 3) + 3}
    ):
        fig, ax = plt.subplots(
            figsize=(int(len(all_classes)) / 2, int(len(all_classes)) / 2)
        )
    # Create the figure and gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[9, 0.4])

    # Plot the data on the left grid
    ax = plt.subplot(gs[0])
    ax.set_title("Affinity Matrix", fontsize=16)

    im = ax.imshow(lookup, vmin=-1, vmax=1, cmap=plt.cm.get_cmap("RdBu"))

    ax.set_xticks(np.arange(0, len(all_classes)))
    ax.set_xticklabels(all_classes)
    ax.set_yticks(np.arange(0, len(all_classes)))
    ax.set_yticklabels(all_classes)

    # Colour the label for selected classes
    for i, c in enumerate(all_classes):
        if c in selected_classes:
            ax.get_xticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_color("red")

    ax.tick_params(axis="x", rotation=90, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    # Create an empty plot on the right grid
    ax2 = plt.subplot(gs[1])

    plt.colorbar(im, cax=ax2)
    # Set the height of the color bar to match the height of the plot
    # cb.ax.set_position([0.96, 0.01, 0.01, 0.01])
    fig.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/affinity_matrix.png", dpi=300)
    plt.close()


def plot_classes_distribution(data: list, category: str) -> None:
    """Plot histogram with classes distribution

    Parameters
    ----------
    data : list
        List of classes
    category : str
        The category of the data (train, test, val)
    """

    logging.info(
        "################################################################",
    )
    logging.info("Visualising classes distribution " + category + "...\n")

    fig, ax = plt.subplots(figsize=(9, 9))
    labels, counts = np.unique(data, return_counts=True)
    ticks = range(len(counts))
    plt.bar(ticks, counts, align="center", color="blue", alpha=0.5)
    plt.xticks(ticks, labels)
    plt.title("Classes Distribution", fontsize=16)
    plt.xlabel("Class", fontsize=16)
    plt.ylabel("Number of Entries", fontsize=16)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/classes_distribution_" + category + ".png", dpi=300)
    plt.close()


def plot_cyc_variable(array: list, variable_name: str):
    """Plot evolution of variable from the cyclical training

    Parameters
    ----------
    array : list
        List of values for the variable
    variable_name : str
        Name of the variable
    """
    logging.info(
        "################################################################",
    )
    logging.info(f"Visualising {variable_name} ...\n")
    plt.plot(array, linewidth=3)
    plt.ylabel(rf"$\{variable_name}$", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f"plots/{variable_name}_array.png", dpi=300)
    plt.close()


def latent_space_similarity(
    latent_space: npt.NDArray,
    class_labels: npt.NDArray,
    mode: str = "",
    epoch: int = 0,
    classes_order: list = [],
) -> None:
    """
    This function calculates the similarity (affinity) between classes in the latent space and builds a matrix.
    Parameters
    ----------
    latent_space: np.ndarray
        The latent space
    class_labels: np.array
        The labels of the latent space
    mode: str
        Mode of the calculation (train, test, val)
    epoch: int
        Epoch number for title
    classes_order: list
        Order of the classes in the matrix

    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising the latent space similarity matrix ...\n")

    # get same label order as affinity matrix
    cosine_sim_matrix = cosine_similarity(latent_space)
    if len(classes_order) == 0:
        unique_classes = np.unique(class_labels)
    else:
        unique_classes_in_data = np.unique(class_labels)
        if np.setdiff1d(unique_classes_in_data, classes_order).size > 0:
            unique_classes = np.concatenate(
                (
                    classes_order,
                    np.setdiff1d(unique_classes_in_data, classes_order),
                )
            )
        else:
            unique_classes = classes_order

    # Calculate average cosine similarity for each pair of classes
    num_classes = len(unique_classes)
    avg_cosine_sim = np.zeros((num_classes, num_classes))
    std_cosine_sim = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(i, num_classes):
            class_i_indices = np.where(class_labels == unique_classes[i])[0]
            class_j_indices = np.where(class_labels == unique_classes[j])[0]
            cosine_sims = cosine_sim_matrix[class_i_indices][
                :, class_j_indices
            ]
            avg_cosine_sim[i, j] = np.mean(cosine_sims)
            avg_cosine_sim[j, i] = avg_cosine_sim[i, j]  # symmetrical matrix

            std_cosine_sim[i, j] = np.std(cosine_sims)
            std_cosine_sim[j, i] = std_cosine_sim[i, j]  # symmetrical matrix

    # Visualize average cosine similarity matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.tight_layout(pad=3)
    plt.imshow(avg_cosine_sim, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(label="Average Cosine Similarity")
    plt.xticks(ticks=np.arange(num_classes), labels=unique_classes)
    plt.yticks(ticks=np.arange(num_classes), labels=unique_classes)
    plt.title(f"Average Cosine Similarity Matrix at epoch :{epoch}")
    plt.xlabel("Class Labels")
    plt.ylabel("Class Labels")
    ax.tick_params(axis="x", rotation=90, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f"plots/similarity_mean{mode}.png", dpi=300)
    plt.close()

    # Visualize average cosine similarity matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.tight_layout(pad=3)
    plt.imshow(std_cosine_sim, cmap="RdBu")
    plt.colorbar(label="Average Cosine Similarity")
    plt.xticks(ticks=np.arange(num_classes), labels=unique_classes)
    plt.yticks(ticks=np.arange(num_classes), labels=unique_classes)
    plt.title(f"Cosine Similarity Matrix Standard Deviation at epoch :{epoch}")
    plt.xlabel("Class Labels")
    plt.ylabel("Class Labels")
    ax.tick_params(axis="x", rotation=90, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(f"plots/similarity_std{mode}.png", dpi=300)
    plt.close()
