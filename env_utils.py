import os
from pathlib import Path


FETCH_ENVS = (
    "FetchReach-v4",
    "FetchPush-v4",
    "FetchPickAndPlace-v4",
    "FetchSlide-v4",
)

ENV_CONFIGS = ("free", "2Staticobs", "2Dynamicobs")


def project_root():
    return Path(__file__).resolve().parent


def resolve_xml_dir(xml_dir=None, env_config=None):
    """Resolve the XML folder used by Gymnasium Robotics Fetch envs."""
    xml_dir = xml_dir or os.environ.get("HER_XML_DIR")
    env_config = env_config or os.environ.get("HER_ENV_CONFIG") or "free"

    if xml_dir:
        path = Path(xml_dir).expanduser()
        if not path.is_absolute():
            path = project_root() / path
    else:
        if env_config not in ENV_CONFIGS:
            raise ValueError(f"Unknown env config: {env_config}. Expected one of {ENV_CONFIGS}.")
        path = project_root() / "xml_envs" / env_config

    fetch_dir = path / "fetch"
    required = ("reach.xml", "push.xml", "pick_and_place.xml", "slide.xml", "robot.xml", "shared.xml")
    missing = [name for name in required if not (fetch_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"XML dir {path} is missing: {', '.join(missing)}")
    return path


def patch_fetch_xml_paths(xml_dir):
    """Point Gymnasium Robotics Fetch tasks at the selected local XML files."""
    xml_dir = str(resolve_xml_dir(xml_dir))

    import gymnasium_robotics.envs.fetch.reach as reach_mod
    import gymnasium_robotics.envs.fetch.push as push_mod
    import gymnasium_robotics.envs.fetch.pick_and_place as pick_mod
    import gymnasium_robotics.envs.fetch.slide as slide_mod

    reach_mod.MODEL_XML_PATH = os.path.join(xml_dir, "fetch", "reach.xml")
    push_mod.MODEL_XML_PATH = os.path.join(xml_dir, "fetch", "push.xml")
    pick_mod.MODEL_XML_PATH = os.path.join(xml_dir, "fetch", "pick_and_place.xml")
    slide_mod.MODEL_XML_PATH = os.path.join(xml_dir, "fetch", "slide.xml")

    return {
        "reach": reach_mod.MODEL_XML_PATH,
        "push": push_mod.MODEL_XML_PATH,
        "pick_and_place": pick_mod.MODEL_XML_PATH,
        "slide": slide_mod.MODEL_XML_PATH,
    }


def infer_env_config(xml_dir):
    parts = {part.lower() for part in Path(xml_dir).parts}
    if "2dynamicobs" in parts:
        return "2Dynamicobs"
    if "2staticobs" in parts:
        return "2Staticobs"
    if "free" in parts:
        return "free"
    return "custom"


def env_id_for(env_name, env_config):
    if env_config == "2Dynamicobs":
        return f"{env_name}-dynobs"
    if env_config == "2Staticobs":
        return f"{env_name}-2staticobs"
    if env_config == "custom":
        return f"{env_name}-customxml"
    return env_name


def run_output_dir(base_dir, env_config, env_name, seed):
    short_env = env_name.replace("Fetch", "").replace("-v4", "").lower()
    return Path(base_dir) / env_config / short_env / f"seed{seed}"
