from .clip_utils import load

def get_model_clip(args):
    model, process = load(args.model_name, args=args)
    return model, process
