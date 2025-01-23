import importlib

def get_item(name):
    dict = {
        "MBRL": "mbrl",
        "MBRLEval": "mbrl_eval",
        "MBRLMbpo": "mbrl_mbpo",
        "MBRLTrain": "mbrl_train",
        "MBRLDone": "mbrl_done",
        "MBRLSupcon": "mbrl_supcon",
        "MBRLSupconParallel": "mbrl_supcon_parallel",
        "MBRLDoneRobo": "mbrl_done_robo",
        "MBRLSample": "mbrl_sample",
        "MBRLSampleNew": "mbrl_sample_new",
        "MBRLSampleDog": "mbrl_sample_dog",
        "MBRLSampleRobo": "mbrl_sample_robo",
        "MBRLSampleEval": "mbrl_sample_eval",
        "MBRLSampleCase": "mbrl_sample_case",
        "LogRun": "log_run",
        "ModelRun": "model_run",
        "NoisyMBRL": "noisy_mbrl",
        "StdMBRL": "std_mbrl",
        "StepTime": "step_time",
        "PltMBRL": "plt_mbrl",
        "MFRL_DDPG": "mfrl_ddpg",
        "MFRL_SAC": "mfrl_sac",
        "MFRL_SAC_MStep": "mfrl_sac_mstep",
        "MFRL_PPO": "mfrl_ppo",
        "MFRL_DDPG_Case": "mfrl_ddpg_case",
        "MFRL_SAC_Case": "mfrl_sac_case",
        "MFRL_PPO_Case": "mfrl_ppo_case",
        "Homework": "homework",
        "USVTrain": "usv_train",
        "MBPO": "mbpo"
    }
    module = importlib.import_module("mymbrl.experiments."+dict[name])
    module_class = getattr(module, name)
    return module_class
    