from . import data, features, model_manager, models, predict, processing, train_models, hp_opt
__all__ = [data, features, model_manager, models, predict, processing, train_models, hp_opt]

def get_version():
    import git
    repo = git.Repo("../")
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    return str(tags[-1])

__version__ = get_version()
