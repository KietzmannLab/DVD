"""Checkpoint compatibility for the DVD training recipes."""


PAPER_RECIPE = "paper"


def recipe_name(name):
    return PAPER_RECIPE if name == "p004-paper" else name


def same_dvd_protocol(saved, current):
    if isinstance(saved, dict):
        if saved.get("method") == "endpoint-linear":
            saved = dict(saved, method="dvd")
        recipe = saved.get("training_recipe")
        if isinstance(recipe, dict) and "name" in recipe and recipe["name"] != recipe_name(recipe["name"]):
            saved = dict(saved, training_recipe=dict(recipe, name=recipe_name(recipe["name"])))
    if saved == current:
        return True
    if not (isinstance(saved, dict) and current.get("method") == "dvd"
            and current.get("version") == 2 and saved.get("version") == 1):
        return False
    config = current["config"].copy()
    components = set(config.pop("speed_components"))
    if saved != dict(current, version=1, config=config):
        return False
    if config["speed"] == 1:
        return True
    # Schema 1 retimed contrast alone in forward mode and both spectral gates in flipped modes.
    flipped = current["backend"] == "reflected-batched"
    old_components = {"acuity", "contrast"} if flipped else {"contrast"}
    active = {name for name, key in (("acuity", "apply_frequency" if flipped else "apply_acuity"),
                                    ("contrast", "apply_amplitude" if flipped else "apply_contrast"),
                                    ("color", "apply_color")) if config[key]}
    return (active & components) == (active & old_components)
