import logging
import os

import yaml
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    ValidationError,
)

import avae.settings as settings


# Model configuration
class AffinityConfig(BaseModel):
    affinity: FilePath = Field(None, description="Path to affinity file")
    batch: PositiveInt = Field(128, description="Batch size")
    beta: float = Field(1, description="Beta value")
    beta_cycle: PositiveInt = Field(4, description="Beta cycle")
    beta_load: FilePath | None = Field(None, description="Path to beta file")
    beta_min: float = Field(0, description="Minimum betvalue")
    beta_ratio: PositiveFloat = Field(0, description="Beta value")
    channels: PositiveInt = Field(64, description="First layer channels")
    classes: FilePath = Field(None, description="Path to classes file")
    classifier: str = Field(
        "NN",
        pattern='^(KNN|NN|LR)$',
        description="Method to classify the latent space. Options "
        "are: KNN (nearest neighbour), NN (neural network), LR (Logistic Regression).",
    )
    config_file: FilePath | None = Field(
        None, description="Path to config file"
    )
    cyc_method_beta: str = Field(
        'flat',
        pattern='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    cyc_method_gamma: str = Field(
        'flat',
        pattern='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    datapath: DirectoryPath = Field(None, description="Path to data directory")
    datatype: str = Field('mrc', pattern='^npy|mrc$', description="Data type")
    debug: bool = Field(False, description="Debug mode")
    depth: PositiveInt = Field(3, description="Number of layers")
    dynamic: bool = Field(False, description="Dynamic visualisation")
    epochs: PositiveInt = Field(20, description="Number of epochs")
    eval: bool = Field(False, description="Evaluation mode")
    freq_acc: PositiveInt = Field(
        10, description="Frequency (in epochs) of accuracy plot"
    )
    freq_all: PositiveInt = Field(
        None, description="Frequency (in epochs) of all plots"
    )
    freq_dis: PositiveInt = Field(
        10, description="Frequency (in epochs) of disentanglement plot"
    )
    freq_emb: PositiveInt = Field(
        10, description="Frequency (in epochs) of embedding plot"
    )
    freq_eval: PositiveInt = Field(
        10, description="Frequency (in epochs) of evaluation"
    )
    freq_int: PositiveInt = Field(
        10, description="Frequency (in epochs) of interpolation plot"
    )
    freq_pos: PositiveInt = Field(
        10, description="Frequency (in epochs) of pose plot"
    )
    freq_rec: PositiveInt = Field(
        10, description="Frequency (in epochs) of reconstruction plot"
    )
    freq_sim: PositiveInt = Field(
        10, description="Frequency (in epochs) of similarity plot"
    )
    freq_sta: PositiveInt = Field(
        10, description="Frequency (in epochs) of states saved."
    )
    gamma: float = Field(2, description="Gamma value")
    gamma_cycle: PositiveInt = Field(4, description="Gamma cycle")
    gamma_load: FilePath | None = Field(
        None, description="Path to gamma array file"
    )
    gamma_min: float = Field(0, description="Minimum gamma value")
    gamma_ratio: float = Field(0.5, description="Gamma ratio")
    gaussian_blur: bool = Field(False, description=" Apply gaussian blur")
    gpu: bool = Field(True, description="Use GPU")
    latent_dims: PositiveInt = Field(8, description="Latent space dimensions")
    learning: PositiveFloat = Field(0.001, description="Learning rate")
    limit: PositiveInt | None = Field(
        None, description="Limit number of samples"
    )
    loss_fn: str = Field('MSE', description="Loss function")
    meta: FilePath | None = Field(None, description="Path to meta file")
    model: str = Field('a', description="Type of model to use")
    new_out: bool = Field(False, description="Create new output directory")
    no_val_drop: bool = Field(
        True,
        description="Do not drop last validation batch if is smaller than batch size",
    )
    normalise: bool = Field(False, description="Normalise data")
    opt_method: str = Field(
        'adam',
        description="Optimisation method.It can be adam/sgd/asgd",
        pattern='^(adam|sgd|asgd)$',
    )
    pose_dims: int = Field(1, description="Pose dimensions")
    rescale: float = Field(None, description="Rescale data")
    restart: bool = Field(False, description="Restart training")
    shift_min: bool = Field(
        False, description="Scale data with min-max transformation"
    )
    split: PositiveInt = Field(20, description="Split ratio")
    state: FilePath | None = Field(None, description="Path to state file")
    tensorboard: bool = Field(False, description="Use tensorboard")
    vis_acc: bool = Field(False, description="Visualise accuracy")
    vis_aff: bool = Field(False, description="Visualise affinity")
    vis_all: bool = Field(False, description="Visualise all")
    vis_cyc: bool = Field(False, description="Visualise beta/gamma cycle")
    vis_dis: bool = Field(False, description="Visualise disentanglement")
    vis_emb: bool = Field(False, description="Visualise embedding")
    vis_his: bool = Field(False, description="Visualise history")
    vis_int: bool = Field(False, description="Visualise interpolation")
    vis_los: bool = Field(False, description="Visualise loss")
    vis_pos: bool = Field(False, description="Visualise pose")
    vis_pose_class: str = Field(
        None, description="Visualise pose per class interpolation"
    )
    vis_z_n_int: str = Field(
        "0,10", description="Visualise latent space interpolation "
    )

    vis_rec: bool = Field(False, description="Visualise reconstruction")
    vis_sim: bool = Field(False, description="Visualise similarity")
    filters: list | None = Field(
        None,
        description="Comma-separated list of filters for the network. Either provide filters, or capacity and depth.",
    )
    bnorm_encoder: bool = Field(
        False, description="Use batch normalisation in encoder"
    )
    bnorm_decoder: bool = Field(
        False, description="Use batch normalisation in decoder"
    )
    klreduction: str = Field('mean', description="KL reduction method")


def load_config_params(
    config_file: str | None = None, local_vars: dict = {}
) -> dict:
    """
    Load configuration parameters from config file and command line arguments.

    Parameters
    ----------
    config_file : str
        Path to config file.
    local_vars : dict
        Dictionary of command line arguments.

    Returns
    -------
    data : dict
        Dictionary of configuration parameters.
    """

    if config_file is not None:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        try:
            data = AffinityConfig(**config_data)
            logging.info("Config file is valid!")
        except ValidationError as e:
            logging.info("Config file is invalid:")
            logging.info(e)
            raise RuntimeError("Config file is invalid:" + str(e))

    else:
        # if no config file is provided, start from default and update with command line arguments
        data = AffinityConfig()

    # check for command line input values and overwrite config file values
    for key, val in local_vars.items():
        if (val is not None and isinstance(val, (int, float, bool, str))) or (
            val is not None and getattr(data, key) is None
        ):
            logging.warning(
                "Command line argument "
                + key
                + " is overwriting config file value to: "
                + str(val)
            )
            # update model with command line arguments
            try:
                data.model_validate({key: val})
                setattr(data, key, val)
            except ValidationError as e:
                logging.info(e)
                raise ValidationError("Config file is invalid:" + str(e))
        else:
            logging.info(
                "Setting "
                + key
                + " to config file value: "
                + str(getattr(data, key))
            )

    # Check for missing values and set to default values
    for key, val in data.model_dump().items():
        if (val is None or val == "None") and key != "config_file":
            #  make sure data variables are provided
            if key == "datapath":
                logging.error(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
                raise ValueError(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
            elif key == "affinity" or key == "classes":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )

                filename_default = os.path.join(
                    getattr(data, 'datapath'), key + ".csv"
                )

                if os.path.isfile(filename_default):
                    try:
                        data.model_validate({key: filename_default})
                        setattr(data, key, val)
                    except ValidationError as e:
                        logging.info(e)
                        raise ValidationError(
                            "Affinity and classes values are invalid:" + str(e)
                        )

                else:
                    setattr(data, key, val)

                logging.info(
                    "Setting up "
                    + key
                    + " in config file to "
                    + str(getattr(data, key))
                )

            elif key == "state":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest state if in evaluation mode."
                )
            elif key == "meta":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest meta if in evaluation mode."
                )
            else:
                # set missing variables to default value
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Default values will be used."
                )

    # return data as dictionary
    return data.model_dump()


def write_config_file(time_stamp_name, data):
    # record final configuration in logger and save to yaml file
    for key, val in data.items():
        logging.info("Parameter " + key + " set to value: " + str(data[key]))

    if not os.path.exists("configs"):
        os.mkdir("configs")
    file = open("configs/avae_final_config" + time_stamp_name + ".yaml", "w")
    yaml.dump(data, file)
    file.close()

    logging.info("YAML File saved!\n")


def setup_visualisation_config(data: dict) -> None:

    if data["vis_all"]:
        settings.VIS_LOS = True
        settings.VIS_ACC = True
        settings.VIS_REC = True
        settings.VIS_CYC = True
        settings.VIS_AFF = True
        settings.VIS_EMB = True
        settings.VIS_INT = True
        settings.VIS_DIS = True
        settings.VIS_POS = True
        settings.VIS_HIS = True
        settings.VIS_SIM = True
        settings.VIS_DYN = True

    else:
        settings.VIS_LOS = data["vis_los"]
        settings.VIS_ACC = data["vis_acc"]
        settings.VIS_REC = data["vis_rec"]
        settings.VIS_CYC = data["vis_cyc"]
        settings.VIS_AFF = data["vis_aff"]
        settings.VIS_EMB = data["vis_emb"]
        settings.VIS_INT = data["vis_int"]
        settings.VIS_DIS = data["vis_dis"]
        settings.VIS_POS = data["vis_pos"]
        settings.VIS_HIS = data["vis_his"]
        settings.VIS_SIM = data["vis_sim"]
        settings.VIS_DYN = data["dynamic"]

    if data["freq_all"] is not None:
        settings.FREQ_EVAL = data["freq_all"]
        settings.FREQ_STA = data["freq_all"]
        settings.FREQ_ACC = data["freq_all"]
        settings.FREQ_REC = data["freq_all"]
        settings.FREQ_EMB = data["freq_all"]
        settings.FREQ_INT = data["freq_all"]
        settings.FREQ_DIS = data["freq_all"]
        settings.FREQ_POS = data["freq_all"]
        settings.FREQ_SIM = data["freq_all"]
    else:
        settings.FREQ_EVAL = data["freq_eval"]
        settings.FREQ_REC = data["freq_rec"]
        settings.FREQ_EMB = data["freq_emb"]
        settings.FREQ_INT = data["freq_int"]
        settings.FREQ_DIS = data["freq_dis"]
        settings.FREQ_POS = data["freq_pos"]
        settings.FREQ_ACC = data["freq_acc"]
        settings.FREQ_STA = data["freq_sta"]
        settings.FREQ_SIM = data["freq_sim"]

    settings.VIS_POSE_CLASS = data["vis_pose_class"]
    settings.VIS_Z_N_INT = data["vis_z_n_int"]
