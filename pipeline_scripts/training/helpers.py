import os
import pooch


def _get_dinov3_checkpoint(model_type: str):
    """Downloads the model checkpoint and returns filepath to it.
    """
    assert isinstance(model_type, str), "Watch out, we need the 'model_type' name."

    # Let's create a cache directory.
    save_directory = os.path.expanduser(pooch.os_cache("synapse_net/dinov3_models"))

    # The weights are stored on owncloud for now
    urls = {"vit_b": "https://owncloud.gwdg.de/index.php/s/PvTzG3fdo2pnNCz/download"}
    hashes = {"vit_b": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"}

    # Make an explicit check
    assert model_type in urls, "Oops, seems like the specific DINOv3 model is not stored on OwnCloud."

    # Download the model weights
    pooch.retrieve(
        url=urls[model_type],
        known_hash=hashes[model_type],
        fname=model_type,
        path=save_directory,
        progressbar=True,
    )

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, model_type)

    return checkpoint_path
