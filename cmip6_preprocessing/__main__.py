"""Command line entry point.

    python -m cmip6_preprocessing MIROC6 pr          # one model, one variable
    python -m cmip6_preprocessing MIROC6             # one model, all variables
    python -m cmip6_preprocessing --all              # every model and variable
    python -m cmip6_preprocessing MIROC6 pr --plot   # show the check plots
    python -m cmip6_preprocessing MIROC6 pr --no-save
"""
import argparse
import traceback

from .config import MODELS, VARIABLES, get_settings
from .core import preprocess, save


def main():
    parser = argparse.ArgumentParser(prog="python -m cmip6_preprocessing",
                                     description="Preprocess CMIP6 historical r1i1p1f1 data.")
    parser.add_argument("model", nargs="?", help="model name, see --list")
    parser.add_argument("variables", nargs="*", help=f"variables, default all of {VARIABLES}")
    parser.add_argument("--all", action="store_true", help="run every model")
    parser.add_argument("--list", action="store_true", help="show the available models")
    parser.add_argument("--plot", action="store_true", help="show the check plots")
    parser.add_argument("--no-save", action="store_true", help="do not write the pickle file")
    args = parser.parse_args()

    if args.list:
        print("\n".join(MODELS))
        return
    if args.all:
        models = list(MODELS)
        if args.model:
            # with --all the positional arguments are variables
            args.variables.insert(0, args.model)
    elif args.model:
        models = [args.model]
    else:
        parser.error("give a model name or --all")
    variables = args.variables or VARIABLES

    failed = []
    for model in models:
        for variable in variables:
            print(f"===== {model} {variable} =====")
            try:
                settings = get_settings(model, variable)
                result = preprocess(settings)
                for key, value in result.items():
                    print(f"{key}: {value.shape}")
                if args.plot:
                    from .plot import plot_check
                    plot_check(result, model, variable)
                if args.no_save:
                    print(f"{model} {variable} has ***NOT*** been saved")
                else:
                    save(result, model, variable)
            except Exception:
                if len(models) * len(variables) == 1:
                    raise
                traceback.print_exc()
                failed.append(f"{model} {variable}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
