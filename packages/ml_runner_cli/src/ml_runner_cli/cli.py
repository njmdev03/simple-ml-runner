import argparse
import simple_config


def main():
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument("op", type=str, default="run", choices=["run", "batch", "export"])

    parser.add_argument("--config", type=str, required=False)

    args = parser.parse_args()

    if args.op == "run":
        simple_config
        print("Run")

    elif args.op == "batch":
        print("Run Batch")

    elif args.op == "export":
        print("Run one-shot task")

    else:
        raise ValueError("Invalid operation requested")

if __name__ == "__main__":
    main()